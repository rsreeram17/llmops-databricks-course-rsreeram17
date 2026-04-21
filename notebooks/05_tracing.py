# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05: MLflow Tracing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is tracing and why it matters
# MAGIC - Span types: AGENT, LLM, TOOL, CHAIN
# MAGIC - Adding `@mlflow.trace` to the options agent
# MAGIC - Session and request metadata
# MAGIC - Searching and inspecting traces in MLflow
# MAGIC
# MAGIC **Trace structure we will build:**
# MAGIC ```
# MAGIC [AGENT]  TracedOptionsAgent.chat()
# MAGIC   ├── [LLM]   _call_llm()          ← first LLM call, decides to search
# MAGIC   ├── [TOOL]  _execute_tool()       ← runs search_options_book()
# MAGIC   └── [LLM]   _call_llm()          ← second LLM call, produces answer
# MAGIC ```
# MAGIC
# MAGIC Every span captures: inputs, outputs, latency, token counts.
# MAGIC All spans are searchable in the MLflow Experiments UI.

# COMMAND ----------

# MAGIC %pip install loguru openai pyyaml pydantic databricks-vectorsearch

# COMMAND ----------

import sys

if "../src" not in sys.path:
    sys.path.insert(0, "../src")

# COMMAND ----------

import json
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.entities import SpanType
from openai import OpenAI
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

logger.info(f"Environment      : {env}")
logger.info(f"Catalog / Schema : {cfg.catalog}.{cfg.schema}")
logger.info(f"LLM endpoint     : {cfg.llm_endpoint}")
logger.info(f"Experiment       : {cfg.experiment_name}")

# COMMAND ----------

w = WorkspaceClient()

llm_client = OpenAI(
    api_key=w.tokens.create(lifetime_seconds=1200).token_value,
    base_url=f"{w.config.host}/serving-endpoints",
)

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint,
)

# Point MLflow at the experiment — all traces land here
mlflow.set_experiment(cfg.experiment_name)

logger.info(f"✓ LLM client ready   → {w.config.host}/serving-endpoints")
logger.info(f"✓ Vector search index: {vs_manager.index_name}")
logger.info(f"✓ MLflow experiment  : {cfg.experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. What is Tracing?
# MAGIC
# MAGIC A **trace** is a complete record of one agent request from start to finish.
# MAGIC It is made up of **spans** — each span records one logical unit of work:
# MAGIC
# MAGIC | Span type | Used for |
# MAGIC |---|---|
# MAGIC | `AGENT` | The outermost span — the whole request |
# MAGIC | `LLM` | Each call to the language model |
# MAGIC | `TOOL` | Each tool execution (search, calculator, …) |
# MAGIC | `CHAIN` | A multi-step sub-pipeline |
# MAGIC
# MAGIC Spans nest automatically: if a traced function calls another traced
# MAGIC function, the inner one becomes a child span of the outer one.
# MAGIC
# MAGIC You don't change your logic — you just add `@mlflow.trace` decorators.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Warm-Up — Trace a Simple Function
# MAGIC
# MAGIC Before touching the agent, see how `@mlflow.trace` works on
# MAGIC a trivial function. Run this cell, then open the MLflow Experiments UI
# MAGIC (left sidebar → Experiments → options-assistant-dev) to see the trace.

# COMMAND ----------


@mlflow.trace(span_type=SpanType.CHAIN)
def demo_pipeline(question: str) -> str:
    """A fake two-step pipeline to show nested spans."""

    with mlflow.start_span("step_retrieve", span_type=SpanType.TOOL) as span:
        span.set_inputs({"query": question})
        fake_result = f"[doc about '{question}']"
        span.set_outputs({"result": fake_result})

    with mlflow.start_span("step_generate", span_type=SpanType.LLM) as span:
        span.set_inputs({"context": fake_result, "question": question})
        answer = "Based on the doc, the answer is: 42"
        span.set_outputs({"answer": answer})

    return answer


result = demo_pipeline("what is delta?")
logger.info(f"Demo result: {result}")
logger.info("✓ Open MLflow Experiments UI to see the trace →")
logger.info(f"  Experiment: {cfg.experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tool and Helper Functions
# MAGIC
# MAGIC Same as notebook 04 — no changes needed here.

# COMMAND ----------


def parse_search_results(results: dict) -> list[dict]:
    """Parse raw similarity_search() response into a list of dicts."""
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    rows = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row, strict=False)) for row in rows]


def search_options_book(query: str, num_results: int = 5) -> str:
    """Search the options trading book. Returns JSON string."""
    raw = vs_manager.search(query, num_results=num_results)
    chunks = parse_search_results(raw)
    results = [
        {
            "source": c.get("source", "options book"),
            "element_type": c.get("element_type", "text"),
            "content": c.get("text", ""),
        }
        for c in chunks
    ]
    return json.dumps({"query": query, "num_results": len(results), "results": results})


SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_options_book",
        "description": (
            "Search the options trading book for relevant information. "
            "Use this whenever the user asks about options concepts, strategies, "
            "pricing models, Greeks (delta, gamma, theta, vega), or volatility."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused search query, e.g. 'delta hedging'.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of excerpts to retrieve (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. TracedOptionsAgent
# MAGIC
# MAGIC This is identical to `OptionsAgent` from notebook 04 with three additions:
# MAGIC
# MAGIC 1. `_call_llm()` — extracted method, decorated with `@mlflow.trace(LLM)`
# MAGIC 2. `_execute_tool()` → decorated with `@mlflow.trace(TOOL)`
# MAGIC 3. `chat()` → decorated with `@mlflow.trace(AGENT)` (root span)
# MAGIC
# MAGIC `mlflow.update_current_trace()` attaches session / request metadata
# MAGIC so you can filter traces by conversation in the UI.

# COMMAND ----------


class TracedOptionsAgent:
    """Options trading agent with full MLflow tracing.

    Every request produces a nested trace:
      AGENT → (LLM → TOOL → LLM → ... → LLM)
    """

    _TOOL_SPECS = [SEARCH_TOOL_SPEC]

    def __init__(self) -> None:
        self.conversation_history: list[dict] = []
        # A stable session ID groups all turns in one conversation
        self.session_id: str = f"session-{uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    @mlflow.trace(span_type=SpanType.LLM, name="call_llm")
    def _call_llm(self, messages: list[dict]) -> dict:
        """Single LLM call. Returns a dict with role, content, tool_calls."""
        response = llm_client.chat.completions.create(
            model=cfg.llm_endpoint,
            messages=messages,
            tools=self._TOOL_SPECS,
            temperature=0.2,
        )
        msg = response.choices[0].message
        # Return a plain dict so MLflow can serialize the span output
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
        }

    # ------------------------------------------------------------------
    @mlflow.trace(span_type=SpanType.TOOL, name="execute_tool")
    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name and return the result as a JSON string."""
        if name == "search_options_book":
            return search_options_book(**args)
        return json.dumps({"error": f"Unknown tool: {name}"})

    # ------------------------------------------------------------------
    @mlflow.trace(span_type=SpanType.AGENT, name="options_agent")
    def chat(self, question: str, max_iterations: int = 10) -> str:
        """Ask the agent a question. Produces a full MLflow trace."""
        request_id = f"req-{uuid4().hex[:8]}"

        # Attach metadata to the root AGENT span
        mlflow.update_current_trace(
            tags={
                "llm_endpoint": cfg.llm_endpoint,
                "index_name": vs_manager.index_name,
                "env": env,
            },
            metadata={"mlflow.trace.session": self.session_id},
            client_request_id=request_id,
        )

        logger.info(f"[Agent] session={self.session_id}  request={request_id}")

        messages: list[dict] = (
            [{"role": "system", "content": cfg.system_prompt}]
            + self.conversation_history
            + [{"role": "user", "content": question}]
        )

        final_answer = "No answer produced."

        for iteration in range(max_iterations):
            logger.info(f"[Agent] Iteration {iteration + 1}/{max_iterations}")

            msg = self._call_llm(messages)

            if msg["tool_calls"]:
                # Add assistant message with tool call requests
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg["content"],
                        "tool_calls": msg["tool_calls"],
                    }
                )

                # Execute each tool
                for tc in msg["tool_calls"]:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])

                    logger.info(f"[Agent] → tool: {tool_name}  args: {tool_args}")
                    result = self._execute_tool(tool_name, tool_args)
                    logger.info(f"[Agent] ← result: {len(result)} chars")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )
            else:
                final_answer = msg["content"]
                logger.info(f"[Agent] Done after {iteration + 1} iteration(s)")
                break
        else:
            logger.warning("[Agent] Max iterations reached")
            final_answer = msg.get("content") or "Max iterations reached."

        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": final_answer})

        return final_answer

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset conversation and start a new session."""
        self.conversation_history = []
        self.session_id = f"session-{uuid4().hex[:8]}"
        logger.info(f"[Agent] New session: {self.session_id}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test — Watch Traces Appear
# MAGIC
# MAGIC Run each cell, then open **MLflow Experiments → options-assistant-dev**
# MAGIC in the left sidebar. Each cell creates one trace with nested spans.

# COMMAND ----------

agent = TracedOptionsAgent()
logger.info(f"Agent session: {agent.session_id}")

# Test 1 — concept (1 search + 1 final LLM call expected)
q1 = "What is the difference between a European and American option?"
logger.info("=" * 70)
logger.info(f"Q: {q1}")
a1 = agent.chat(q1)
logger.info(f"A: {a1}")

# COMMAND ----------

# Test 2 — Greeks
q2 = "Explain gamma and how it changes near expiry."
logger.info("=" * 70)
logger.info(f"Q: {q2}")
a2 = agent.chat(q2)
logger.info(f"A: {a2}")

# COMMAND ----------

# Test 3 — strategy (may trigger 2 searches)
q3 = "What is an iron condor and when does a trader use it?"
logger.info("=" * 70)
logger.info(f"Q: {q3}")
a3 = agent.chat(q3)
logger.info(f"A: {a3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Inspect Traces Programmatically
# MAGIC
# MAGIC `mlflow.search_traces()` returns a pandas DataFrame with one row per trace.
# MAGIC Useful for auditing, debugging, or feeding into an evaluation pipeline.

# COMMAND ----------

traces_df = mlflow.search_traces(
    order_by=["timestamp_ms DESC"],
    max_results=10,
)

logger.info(f"Found {len(traces_df)} recent traces")

if len(traces_df) > 0:
    logger.info(f"Columns: {list(traces_df.columns)}")
    scalar_cols = [
        c
        for c in traces_df.columns
        if c not in ("request", "response", "spans", "inputs", "outputs")
    ]
    display(traces_df[scalar_cols].head(10))  # noqa: F821

# COMMAND ----------

# Inspect the most recent trace in detail
if len(traces_df) > 0:
    t = traces_df.iloc[0]
    logger.info("Most recent trace:")
    logger.info(f"  request_id     : {t.get('request_id', 'N/A')}")
    logger.info(f"  execution_time : {t.get('execution_time_ms', 'N/A')} ms")
    logger.info(f"  status         : {t.get('status', 'N/A')}")

    tags = t.get("tags") or {}
    if tags:
        logger.info("  tags:")
        for k, v in tags.items():
            logger.info(f"    {k}: {v}")

    meta = t.get("request_metadata") or {}
    if meta:
        logger.info("  metadata:")
        for k, v in meta.items():
            logger.info(f"    {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Multi-Turn — Session Tracking
# MAGIC
# MAGIC Each turn in the same session shares `session_id` in its trace metadata.
# MAGIC This lets you filter all turns of one conversation in the MLflow UI:
# MAGIC
# MAGIC ```
# MAGIC filter: request_metadata.`mlflow.trace.session` = 'session-abc12345'
# MAGIC ```

# COMMAND ----------

agent.clear()
logger.info(f"New session: {agent.session_id}")
session = agent.session_id  # capture for search later

# Turn 1
t1 = "What is implied volatility?"
logger.info(f"[Turn 1] Q: {t1}")
r1 = agent.chat(t1)
logger.info(f"[Turn 1] A: {r1}\n")

# COMMAND ----------

# Turn 2 — follow-up
t2 = "How does it differ from historical volatility?"
logger.info(f"[Turn 2] Q: {t2}")
r2 = agent.chat(t2)
logger.info(f"[Turn 2] A: {r2}\n")

# COMMAND ----------

# Turn 3 — practical
t3 = "Which one should a trader use when pricing a short-dated option?"
logger.info(f"[Turn 3] Q: {t3}")
r3 = agent.chat(t3)
logger.info(f"[Turn 3] A: {r3}")

# COMMAND ----------

# Find all traces for this session
session_traces = mlflow.search_traces(
    filter_string=(f"request_metadata.`mlflow.trace.session` = '{session}'"),
    order_by=["timestamp_ms ASC"],
)

logger.info(f"Traces for session '{session}': {len(session_traces)}")

if len(session_traces) > 0:
    scalar_cols = [
        c
        for c in session_traces.columns
        if c not in ("request", "response", "spans", "inputs", "outputs")
    ]
    display(session_traces[scalar_cols])  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Tracing Complete!
# MAGIC
# MAGIC What every request now produces:
# MAGIC
# MAGIC | Span | Captures |
# MAGIC |---|---|
# MAGIC | `AGENT options_agent` | Full latency, session ID, request ID |
# MAGIC | `LLM call_llm` | Messages sent, response received, token count |
# MAGIC | `TOOL execute_tool` | Tool name, args, result |
# MAGIC
# MAGIC ### Where to look in the UI:
# MAGIC
# MAGIC **Experiments → options-assistant-dev → Traces tab**
# MAGIC
# MAGIC Click any trace to see the full span tree, inputs/outputs per span,
# MAGIC and latency breakdown.
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC - **Notebook 06** — Evaluation: define quality scorers and run them
# MAGIC   against the traces you just captured.

# COMMAND ----------

logger.info("✓ Notebook 05 complete!")
logger.info(f"   Experiment : {cfg.experiment_name}")
logger.info(f"   Session    : {session}")
logger.info(f"   Traces     : {len(session_traces)} for last session")
logger.info("\nOpen MLflow Experiments UI to inspect the spans →")
