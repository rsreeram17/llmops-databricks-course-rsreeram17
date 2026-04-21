"""Options Trading Assistant — MLflow ResponsesAgent model.

This file is uploaded to the MLflow artifact store when the model is logged.
It must be entirely self-contained: no imports from the local src/ package.

Auth in serving:
  WorkspaceClient() reads DATABRICKS_HOST and DATABRICKS_TOKEN automatically
  from environment variables injected by Databricks Model Serving.

Config at serving time:
  MLflow stores model_config.yaml alongside the artifact. ModelConfig reads
  that file in serving and falls back to development_config when running
  locally (e.g. in a notebook smoke test).

Input format (OpenAI Responses API):
    {
        "input": [{"role": "user", "content": "..."}],  # or a plain string
        "custom_inputs": {          # optional
            "session_id": "...",
            "request_id": "..."
        }
    }

Output format (ResponsesAgentResponse):
    {
        "output": [
            {
                "type": "message",
                "id": "<uuid>",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "<answer>"}]
            }
        ],
        "custom_outputs": {"session_id": "...", "request_id": "..."}
    }
"""

import json
import os
from typing import Any
from uuid import uuid4

import mlflow
from mlflow.models import ModelConfig
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# ---------------------------------------------------------------------------
# Helpers (module-level so they are importable without instantiating the class)
# ---------------------------------------------------------------------------


def _parse_search_results(results: dict) -> list[dict]:
    """Parse the parallel-array response returned by similarity_search()."""
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    rows = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row, strict=False)) for row in rows]


_SEARCH_TOOL_SPEC: dict = {
    "type": "function",
    "function": {
        "name": "search_options_book",
        "description": (
            "Search the options trading book for relevant information. "
            "Use this whenever the user asks about options concepts, "
            "strategies, pricing models, Greeks (delta, gamma, theta, "
            "vega), volatility, or any other topic covered in the book."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A focused search query describing what to find, "
                        "e.g. 'delta hedging mechanics' or "
                        "'Black-Scholes assumptions'."
                    ),
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

# ---------------------------------------------------------------------------
# Config — read at module load time
#
# In serving: MLflow writes model_config.yaml next to the artifact;
#             ModelConfig() reads that file automatically.
# Locally:    ModelConfig falls back to development_config below.
# ---------------------------------------------------------------------------

_config = ModelConfig(
    development_config={
        "catalog": "mlops_dev",
        "schema": "rsreeram",
        "llm_endpoint": "databricks-llama-4-maverick",
        "embedding_endpoint": "databricks-gte-large-en",
        "vector_search_endpoint": "options_vs_endpoint",
        "system_prompt": (
            "You are a helpful AI assistant that answers questions "
            "about options trading based on the provided book content."
        ),
    }
)


# ---------------------------------------------------------------------------
# ResponsesAgent model
# ---------------------------------------------------------------------------


class OptionsAssistantModel(ResponsesAgent):
    """Options trading agent wrapped as an MLflow ResponsesAgent.

    MLflow calls set_model() at the bottom of this file to register
    this class as the model. predict() is the only required method;
    predict_stream() is intentionally omitted (not needed for this use case).
    """

    def __init__(self) -> None:
        self.catalog: str = _config.get("catalog")
        self.schema: str = _config.get("schema")
        self.llm_endpoint: str = _config.get("llm_endpoint")
        self.embedding_endpoint: str = _config.get("embedding_endpoint")
        self.vector_search_endpoint: str = _config.get("vector_search_endpoint")
        self.system_prompt: str = _config.get("system_prompt") or (
            "You are a helpful AI assistant that answers questions "
            "about options trading based on the provided book content."
        )
        # Derived values
        self.index_name: str = f"{self.catalog}.{self.schema}.options_index"

    # ------------------------------------------------------------------
    def _get_clients(self) -> tuple[Any, Any]:
        """Build LLM + VectorSearch clients from injected env credentials.

        Returns:
            Tuple of (OpenAI client, VectorSearch index object)
        """
        from databricks.sdk import WorkspaceClient
        from databricks.vector_search.client import VectorSearchClient

        w = WorkspaceClient()
        # SDK-managed client: no manual token construction needed
        llm = w.serving_endpoints.get_open_ai_client()

        # VectorSearch client uses env vars injected by model serving
        host = os.environ["DATABRICKS_HOST"]
        token = os.environ["DATABRICKS_TOKEN"]
        vsc = VectorSearchClient(
            workspace_url=host,
            personal_access_token=token,
            disable_notice=True,
        )
        index = vsc.get_index(index_name=self.index_name)
        return llm, index

    # ------------------------------------------------------------------
    def _search(self, index: Any, query: str, num_results: int = 5) -> str:  # noqa: ANN401
        """Hybrid vector + BM25 search. Returns a JSON string."""
        raw = index.similarity_search(
            query_text=query,
            columns=["id", "text", "source", "element_type"],
            num_results=num_results,
            query_type="hybrid",
        )
        chunks = _parse_search_results(raw)
        results = [
            {
                "source": c.get("source", "options book"),
                "content": c.get("text", ""),
            }
            for c in chunks
        ]
        return json.dumps(
            {"query": query, "num_results": len(results), "results": results}
        )

    # ------------------------------------------------------------------
    def _run_agent(self, llm: Any, index: Any, question: str) -> str:  # noqa: ANN401
        """Agentic loop: LLM decides when to call the search tool.

        The loop runs until the LLM produces a plain text response (no
        tool calls) or until max_iterations is reached.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        msg = None
        for _ in range(10):
            response = llm.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=[_SEARCH_TOOL_SPEC],
                temperature=0.2,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # 1. Record the assistant's tool-call request
                messages.append(
                    {
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
                            for tc in msg.tool_calls
                        ],
                    }
                )
                # 2. Execute each tool and feed the result back
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = self._search(index, **args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            else:
                # LLM produced a final answer — stop the loop
                return msg.content or ""

        # Safety fallback: return whatever content we have
        return (getattr(msg, "content", "") or "") if msg else ""

    # ------------------------------------------------------------------
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Handle one request from the serving endpoint.

        Args:
            request: ResponsesAgentRequest with fields:
                input        — list of Responses API items, or a plain string
                custom_inputs — optional {"session_id": ..., ...}

        Returns:
            ResponsesAgentResponse with fields:
                output        — list with one assistant message item
                custom_outputs — echoed custom_inputs (for tracing)
        """
        # Convert Responses API input items to chat-completions message dicts
        messages = self.prep_msgs_for_cc_llm(request.input)

        # Find the most recent user turn
        user_turns = [m for m in messages if m.get("role") == "user"]
        if not user_turns:
            return ResponsesAgentResponse(
                output=[
                    self.create_text_output_item("No question received.", str(uuid4()))
                ],
                custom_outputs=request.custom_inputs or {},
            )

        question: str = user_turns[-1]["content"]

        llm, index = self._get_clients()
        answer = self._run_agent(llm, index, question)

        return ResponsesAgentResponse(
            output=[self.create_text_output_item(answer, str(uuid4()))],
            custom_outputs=request.custom_inputs or {},
        )


# Required by MLflow "Models from Code" — tells MLflow which class is the model
mlflow.models.set_model(OptionsAssistantModel())
