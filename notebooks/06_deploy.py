# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 06: Log, Register and Deploy
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - MLflow ResponsesAgent: wrapping a custom agent so MLflow can serve it
# MAGIC - Declaring Databricks resource dependencies (Vector Search, LLM)
# MAGIC - Logging a model run with an input example and resources
# MAGIC - Registering the model to Unity Catalog (versioned, aliased)
# MAGIC - Deploying with `agents.deploy()` — one call creates the endpoint
# MAGIC - Testing the live endpoint via the OpenAI Responses API client
# MAGIC
# MAGIC **End-to-end flow:**
# MAGIC ```
# MAGIC options_assistant_model.py   ← self-contained ResponsesAgent model file
# MAGIC          ↓
# MAGIC  mlflow.pyfunc.log_model()   ← stores model + config in an MLflow run
# MAGIC          ↓
# MAGIC  mlflow.register_model()     ← creates Unity Catalog model version (v1, v2…)
# MAGIC          ↓
# MAGIC  MlflowClient.set_alias()    ← pin  @champion  to the version to deploy
# MAGIC          ↓
# MAGIC  agents.deploy()             ← Databricks creates a serving endpoint
# MAGIC          ↓
# MAGIC  HTTPS endpoint              ← OpenAI Responses API /invocations
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q loguru openai pyyaml pydantic databricks-vectorsearch databricks-agents mlflow  # noqa: E501

# COMMAND ----------

import sys

if "../src" not in sys.path:
    sys.path.insert(0, "../src")

# COMMAND ----------

import importlib.metadata
import importlib.util
import os
import time
from datetime import datetime

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)
from pyspark.sql import SparkSession

from options_assistant.config import get_env, load_config
from options_assistant.vector_search import VectorSearchManager

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

logger.info(f"Environment  : {env}")
logger.info(f"Catalog      : {cfg.catalog}.{cfg.schema}")
logger.info(f"LLM endpoint : {cfg.llm_endpoint}")
logger.info(f"Experiment   : {cfg.experiment_name}")

# COMMAND ----------

w = WorkspaceClient()

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint,
)

mlflow.set_experiment(cfg.experiment_name)
mlflow.set_registry_uri("databricks-uc")

logger.info(f"✓ Vector search index : {vs_manager.index_name}")
logger.info("✓ MLflow registry     : databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. What is MLflow ResponsesAgent?
# MAGIC
# MAGIC `ResponsesAgent` is MLflow's purpose-built base class for LLM agents.
# MAGIC It handles the schema automatically — no manual signature inference needed.
# MAGIC
# MAGIC ```python
# MAGIC class MyAgent(mlflow.pyfunc.ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         # read config from ModelConfig (loaded from model_config.yaml
# MAGIC         # in serving; from development_config dict locally)
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         # called for every request — return the response
# MAGIC ```
# MAGIC
# MAGIC **`ModelConfig`** bridges local development and serving:
# MAGIC
# MAGIC | Context | Where config comes from |
# MAGIC |---|---|
# MAGIC | Local smoke test | `development_config` dict inside the model file |
# MAGIC | Served endpoint | `model_config.yaml` written by `log_model()` |
# MAGIC
# MAGIC We use the **file path** approach. `options_assistant_model.py` at the
# MAGIC repo root contains `OptionsAssistantModel` — a fully self-contained agent
# MAGIC with no imports from the local `src/` package.
# MAGIC
# MAGIC ### Input / output contract
# MAGIC
# MAGIC The endpoint uses the **OpenAI Responses API** format:
# MAGIC ```json
# MAGIC {
# MAGIC   "input": [{"role": "user", "content": "What is a straddle?"}],
# MAGIC   "custom_inputs": {"session_id": "s-001", "request_id": "r-001"}
# MAGIC }
# MAGIC ```
# MAGIC And every response:
# MAGIC ```json
# MAGIC {
# MAGIC   "output": [
# MAGIC     {
# MAGIC       "type": "message",
# MAGIC       "id": "<uuid>",
# MAGIC       "role": "assistant",
# MAGIC       "content": [{"type": "output_text", "text": "A straddle is …"}]
# MAGIC     }
# MAGIC   ],
# MAGIC   "custom_outputs": {"session_id": "s-001", "request_id": "r-001"}
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Smoke-Test the Model Locally
# MAGIC
# MAGIC Before logging anything to MLflow, instantiate the model exactly as the
# MAGIC serving cluster will and call `predict()` with a sample input.
# MAGIC This catches import errors, config bugs, and auth problems early.

# COMMAND ----------

# Import the model class from the file we will log
_model_path = os.path.abspath("../options_assistant_model.py")
_spec = importlib.util.spec_from_file_location("options_assistant_model", _model_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
OptionsAssistantModel = _module.OptionsAssistantModel
ResponsesAgentRequest = _module.ResponsesAgentRequest

logger.info(f"Loaded model class from: {_model_path}")

# COMMAND ----------

# In Databricks the env vars are injected automatically.
# For local smoke-testing we populate them from the WorkspaceClient.
os.environ.setdefault("DATABRICKS_HOST", w.config.host)
os.environ.setdefault(
    "DATABRICKS_TOKEN",
    w.tokens.create(lifetime_seconds=600).token_value,
)

# ModelConfig reads development_config from within the model file —
# no fake context needed. Just instantiate and call predict().
model_instance = OptionsAssistantModel()
logger.info("✓ Model loaded — running smoke test...")

# COMMAND ----------

_test_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is a straddle strategy?"}],
    custom_inputs={"session_id": "smoke-test-001", "request_id": "r-001"},
)

_result = model_instance.predict(_test_request)

_answer = _result.output[0].content[0]["text"]
logger.info("Smoke test PASSED")
logger.info(f"Answer preview: {_answer[:300]}")
logger.info(f"Custom outputs: {_result.custom_outputs}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log the Model
# MAGIC
# MAGIC `mlflow.pyfunc.log_model` stores four things in the MLflow run:
# MAGIC
# MAGIC | Artifact | Purpose |
# MAGIC |---|---|
# MAGIC | `options_assistant_model.py` | The agent code |
# MAGIC | `MLmodel` | Schema, flavour, input example |
# MAGIC | `model_config.yaml` | Config values read by `ModelConfig()` at serve time |
# MAGIC | `requirements.txt` | Pip packages the model needs at serving time |
# MAGIC
# MAGIC **`resources`** — not Python packages. These are Databricks resources
# MAGIC (Vector Search index, serving endpoints) that the model calls at
# MAGIC runtime. Declaring them lets Databricks auto-grant the correct
# MAGIC permissions to the serving endpoint's service principal.
# MAGIC
# MAGIC **No manual signature** — `ResponsesAgent` provides its schema
# MAGIC automatically. `log_model` does not need a `signature=` argument.

# COMMAND ----------

# Packages the model needs when it runs on the serving cluster
pip_requirements = [
    f"openai=={importlib.metadata.version('openai')}",
    (f"databricks-vectorsearch=={importlib.metadata.version('databricks-vectorsearch')}"),
    f"databricks-sdk=={importlib.metadata.version('databricks-sdk')}",
    f"mlflow=={importlib.metadata.version('mlflow')}",
]
logger.info("Pip requirements:")
for r in pip_requirements:
    logger.info(f"  {r}")

# COMMAND ----------

# Databricks resources the model calls at runtime.
# Declaring them here lets Databricks auto-configure permissions.
resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
    DatabricksVectorSearchIndex(index_name=vs_manager.index_name),
]

# Config values stored as model_config.yaml alongside the artifact.
# ModelConfig() reads this file in serving (development_config is the
# local fallback baked into options_assistant_model.py).
model_config = {
    "catalog": cfg.catalog,
    "schema": cfg.schema,
    "llm_endpoint": cfg.llm_endpoint,
    "embedding_endpoint": cfg.embedding_endpoint,
    "vector_search_endpoint": cfg.vector_search_endpoint,
    "system_prompt": cfg.system_prompt,
}

# A representative request used as the input example in the MLmodel file
input_example = {
    "input": [{"role": "user", "content": "What is a straddle strategy?"}],
    "custom_inputs": {"session_id": "example-001", "request_id": "r-001"},
}

# COMMAND ----------

ts = datetime.now().strftime("%Y-%m-%d")
run_name = f"options-assistant-{ts}"

logger.info(f"Starting MLflow run: {run_name}")

with mlflow.start_run(run_name=run_name) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=_model_path,
        resources=resources,
        input_example=input_example,
        model_config=model_config,
        pip_requirements=pip_requirements,
    )
    logger.info(f"✓ Model logged to run: {run.info.run_id}")
    logger.info(f"  Artifact URI : {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register to Unity Catalog
# MAGIC
# MAGIC Registering creates a **versioned entry** in Unity Catalog so you can:
# MAGIC - Track which run produced each deployed version
# MAGIC - Roll back by pointing the alias to an older version
# MAGIC - Audit who deployed what and when
# MAGIC
# MAGIC ```
# MAGIC mlops_dev.rsreeram.options_assistant
# MAGIC   ├── version 1  ← first log
# MAGIC   ├── version 2  ← second log (e.g. after tuning the prompt)
# MAGIC   └── …
# MAGIC ```
# MAGIC `@champion` is the alias that `agents.deploy()` will use.
# MAGIC Changing which version `@champion` points to is all it takes to
# MAGIC update the live endpoint.

# COMMAND ----------

model_name = f"{cfg.catalog}.{cfg.schema}.options_assistant"

logger.info(f"Registering model: {model_name}")

registered = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    tags={"run_id": run.info.run_id, "env": env},
    env_pack="databricks_model_serving",
)

logger.info(f"✓ Registered: {model_name} version {registered.version}")

# COMMAND ----------

# Pin the @champion alias to this version.
# agents.deploy() will look up @champion to find what to deploy.
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=registered.version,
)

logger.info(f"✓ Alias @champion → version {registered.version} of {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Deploy with `agents.deploy()`
# MAGIC
# MAGIC `agents.deploy()` does several things in a single call:
# MAGIC
# MAGIC 1. **Creates** (or updates) a Databricks Model Serving endpoint
# MAGIC 2. **Wires** the endpoint to the registered model version
# MAGIC 3. **Configures** auto-scaling (`scale_to_zero=True` → no cost when idle)
# MAGIC 4. **Sets up** MLflow trace forwarding to your experiment
# MAGIC 5. **Grants** the endpoint's service principal access to the declared
# MAGIC    resources (Vector Search index, LLM endpoints)
# MAGIC
# MAGIC The deployment takes **5–10 minutes**. The cell returns immediately —
# MAGIC Databricks provisions the cluster in the background.
# MAGIC You can monitor progress in the UI:
# MAGIC **Serving → options-assistant-dev** or run the monitoring cell below.

# COMMAND ----------

endpoint_name = f"options-assistant-{env}"

# Fetch the version number attached to @champion
champion_version = int(client.get_model_version_by_alias(model_name, "champion").version)

experiment = client.get_experiment_by_name(cfg.experiment_name)

logger.info(f"Deploying model    : {model_name} v{champion_version}")
logger.info(f"Endpoint name      : {endpoint_name}")
logger.info(f"MLflow experiment  : {cfg.experiment_name}")

agents.deploy(
    model_name=model_name,
    model_version=champion_version,
    endpoint_name=endpoint_name,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        # Traces from the deployed model will land in this experiment
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
    },
)

logger.info("✓ Deployment submitted — provisioning in background (~5–10 min)")
logger.info(f"  Monitor: Serving → {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Wait for the Endpoint
# MAGIC
# MAGIC Run this cell periodically until you see `READY`.
# MAGIC Typically takes 5–10 minutes on first deploy.

# COMMAND ----------


def check_endpoint_state(name: str) -> str:
    """Return the current endpoint ready-state as a string."""
    try:
        ep = w.serving_endpoints.get(name)
        config_state = ep.state.config_update.value
        ready_state = ep.state.ready.value
        return f"config_update={config_state}  ready={ready_state}"
    except Exception as e:
        return f"Error: {e}"


state = check_endpoint_state(endpoint_name)
logger.info(f"Endpoint state: {state}")
logger.info("→ Re-run this cell every minute until you see: ready=READY")

# COMMAND ----------


# Optional: poll until the endpoint is ready (blocks the cell for up to 15 min)
def wait_for_endpoint(name: str, timeout_min: int = 15) -> bool:
    """Block until endpoint is READY or timeout is reached."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        try:
            ep = w.serving_endpoints.get(name)
            config_state = ep.state.config_update.value
            ready_state = ep.state.ready.value
            logger.info(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"config={config_state}  ready={ready_state}"
            )
            if config_state == "NOT_UPDATING" and ready_state == "READY":
                logger.info(f"✓ Endpoint {name!r} is READY!")
                return True
            if config_state == "UPDATE_FAILED":
                logger.error("Deployment failed — check Serving UI for details")
                return False
        except Exception as e:
            logger.warning(f"Could not poll endpoint: {e}")
        time.sleep(30)
    logger.warning("Timeout reached — endpoint may still be provisioning")
    return False


# Uncomment to block and poll:
# wait_for_endpoint(endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test the Deployed Endpoint
# MAGIC
# MAGIC Once the endpoint shows **READY**, run the cells below.
# MAGIC The endpoint uses the **OpenAI Responses API** (`client.responses.create`),
# MAGIC which is the native interface for `ResponsesAgent` models.
# MAGIC
# MAGIC `get_open_ai_client()` returns an SDK-managed OpenAI client that handles
# MAGIC auth automatically — no token construction needed.

# COMMAND ----------

llm_client = w.serving_endpoints.get_open_ai_client()

logger.info(f"Client base URL : {w.config.host}/serving-endpoints")
logger.info(f"Target endpoint : {endpoint_name}")

# COMMAND ----------

# Single-turn test — basic concept question
_q1 = "What is the difference between a European and American option?"

logger.info("=" * 70)
logger.info(f"Q: {_q1}")
logger.info("=" * 70)

_resp1 = llm_client.responses.create(
    model=endpoint_name,
    input=[{"role": "user", "content": _q1}],
)

_a1 = _resp1.output[0].content[0].text
logger.info(f"A: {_a1}")

# COMMAND ----------

# Multi-turn test — follow-up question
_resp2 = llm_client.responses.create(
    model=endpoint_name,
    input=[
        {"role": "user", "content": "What is implied volatility?"},
        {"role": "assistant", "content": _a1},
        {"role": "user", "content": "How does it compare to historical volatility?"},
    ],
)
logger.info(f"Follow-up answer: {_resp2.output[0].content[0].text[:400]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. How to Use the Endpoint in Any Application
# MAGIC
# MAGIC Now that the endpoint is live, **any application** can query it using
# MAGIC the standard OpenAI Python SDK — no Spark, no Databricks notebook needed.
# MAGIC
# MAGIC ### Python (outside Databricks)
# MAGIC ```python
# MAGIC import os
# MAGIC from openai import OpenAI
# MAGIC
# MAGIC client = OpenAI(
# MAGIC     api_key=os.environ["DATABRICKS_TOKEN"],   # personal access token
# MAGIC     base_url="https://<your-workspace>.azuredatabricks.net/serving-endpoints",
# MAGIC )
# MAGIC
# MAGIC response = client.responses.create(
# MAGIC     model="options-assistant-dev",            # your endpoint name
# MAGIC     input=[{"role": "user", "content": "What is delta hedging?"}],
# MAGIC )
# MAGIC print(response.output[0].content[0].text)
# MAGIC ```
# MAGIC
# MAGIC ### REST (curl / Postman)
# MAGIC ```bash
# MAGIC curl -X POST \
# MAGIC   https://<host>/serving-endpoints/<endpoint-name>/invocations \
# MAGIC   -H "Authorization: Bearer $DATABRICKS_TOKEN" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{
# MAGIC     "input": [{"role": "user", "content": "Explain gamma risk."}]
# MAGIC   }'
# MAGIC ```
# MAGIC
# MAGIC ### Updating the deployed model
# MAGIC When you improve the agent (better prompt, new tool, etc.):
# MAGIC 1. Re-run the log + register cells → creates a new version (v2, v3…)
# MAGIC 2. Move the `@champion` alias to the new version
# MAGIC 3. Re-run `agents.deploy()` — it picks up `@champion` automatically
# MAGIC
# MAGIC Zero downtime: the old version keeps serving traffic until the new
# MAGIC one is fully ready.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Deployment Complete!
# MAGIC
# MAGIC | Step | What happened |
# MAGIC |---|---|
# MAGIC | `log_model` | Agent code + config stored in an MLflow run |
# MAGIC | `register_model` | Version created in Unity Catalog |
# MAGIC | `set_alias(@champion)` | Pinned the version to deploy |
# MAGIC | `agents.deploy()` | Serving endpoint created with auto-scaling |
# MAGIC | Endpoint test | Verified the live endpoint answers correctly |
# MAGIC
# MAGIC ### Where to look in the UI
# MAGIC
# MAGIC | UI location | What you'll find |
# MAGIC |---|---|
# MAGIC | **Serving → options-assistant-dev** | Endpoint status, traffic, latency |
# MAGIC | **Models → mlops_dev.rsreeram.options_assistant** | Version history |
# MAGIC | **Experiments → options-assistant-dev → Traces** | Per-request traces |

# COMMAND ----------

logger.info("✓ Notebook 06 complete!")
logger.info(f"   Model    : {model_name} v{registered.version} (@champion)")
logger.info(f"   Endpoint : {endpoint_name}")
logger.info(f"   Traces   : {cfg.experiment_name}")
logger.info("\nYour options assistant is live. Query it like any OpenAI endpoint →")
logger.info(f"  base_url = {w.config.host}/serving-endpoints")
logger.info(f"  model    = {endpoint_name}")
