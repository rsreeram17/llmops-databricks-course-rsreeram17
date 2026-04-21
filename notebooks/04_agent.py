# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04: Options Trading Agent
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is the difference between RAG and an Agent?
# MAGIC - Defining tools in the OpenAI function-calling format
# MAGIC - The agentic loop: LLM decides when and what to search
# MAGIC - Building `OptionsAgent` — a stateful, tool-calling agent
# MAGIC - Multi-turn conversations with memory
# MAGIC
# MAGIC **Agent Flow:**
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC LLM (with tool specs)
# MAGIC     ↓
# MAGIC  Does LLM want to call a tool?
# MAGIC    YES                   NO
# MAGIC     ↓                     ↓
# MAGIC  Execute tool        Return answer ✅
# MAGIC  Add result to
# MAGIC  message history
# MAGIC     ↓
# MAGIC  Loop back to LLM
# MAGIC ```
# MAGIC
# MAGIC **Key difference from Notebook 03 (RAG):**
# MAGIC
# MAGIC | RAG | Agent |
# MAGIC |-----|-------|
# MAGIC | Always retrieves | Retrieves only when needed |
# MAGIC | Exactly 1 search | 0, 1, or many searches |
# MAGIC | You write the query | LLM writes the query |
# MAGIC | Single LLM call | Loop of LLM calls |

# COMMAND ----------

# MAGIC %pip install loguru openai pyyaml pydantic databricks-vectorsearch

# COMMAND ----------

import sys

if "../src" not in sys.path:
    sys.path.insert(0, "../src")

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient
from loguru import logger
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

logger.info(f"Environment       : {env}")
logger.info(f"Catalog / Schema  : {cfg.catalog}.{cfg.schema}")
logger.info(f"LLM endpoint      : {cfg.llm_endpoint}")

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

logger.info(f"✓ LLM client ready  → {w.config.host}/serving-endpoints")
logger.info(f"✓ Vector search index: {vs_manager.index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The Tool
# MAGIC
# MAGIC A tool is just a Python function + a JSON description that tells the LLM:
# MAGIC - **what the function does** (description)
# MAGIC - **what parameters it accepts** (schema)
# MAGIC - **when to use it** (the description again — this is the most important part)
# MAGIC
# MAGIC The LLM never calls the function directly. It returns a structured
# MAGIC `tool_call` in its response and **your code** executes the function,
# MAGIC then feeds the result back into the conversation.

# COMMAND ----------


def parse_search_results(results: dict) -> list[dict]:
    """Parse raw similarity_search() response into a list of dicts."""
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    rows = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row, strict=False)) for row in rows]


def search_options_book(query: str, num_results: int = 5) -> str:
    """Search the options trading book for relevant content.

    Args:
        query: What to search for (e.g. "how does delta hedging work").
        num_results: How many chunks to return (default 5).

    Returns:
        JSON string with a list of matching excerpts and their sources.
    """
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


# COMMAND ----------

# Verify the tool works on its own before wiring it into the agent
_raw = search_options_book("what is a straddle strategy?", num_results=2)
_parsed = json.loads(_raw)
logger.info(f"Tool smoke-test — query: '{_parsed['query']}'")
logger.info(f"Returned {_parsed['num_results']} results")
for i, r in enumerate(_parsed["results"], 1):
    logger.info(f"\n{i}. source={r['source']} | type={r['element_type']}")
    logger.info(f"   {r['content'][:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tool Specification
# MAGIC
# MAGIC The spec is the JSON schema the LLM reads to understand the tool.
# MAGIC The **description fields are the most important** — they tell the LLM
# MAGIC when and how to use the tool. Vague descriptions lead to missed or
# MAGIC incorrect tool calls.

# COMMAND ----------

SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_options_book",
        "description": (
            "Search the options trading book for relevant information. "
            "Use this whenever the user asks about options concepts, strategies, "
            "pricing models, Greeks (delta, gamma, theta, vega), volatility, "
            "or any topic covered in the book."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A focused search query describing what information to find. "
                        "Phrase it as a concept or question, e.g. "
                        "'delta hedging mechanics' or 'Black-Scholes assumptions'."
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of excerpts to retrieve (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

logger.info("Tool spec ready:")
logger.info(json.dumps(SEARCH_TOOL_SPEC, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The Agentic Loop
# MAGIC
# MAGIC This is the core mechanic. Each iteration:
# MAGIC
# MAGIC 1. Call the LLM with the current message history + tool specs
# MAGIC 2. If the LLM returns `tool_calls` → execute each one, add results to history
# MAGIC 3. If the LLM returns a plain text response → we're done, return it
# MAGIC
# MAGIC The LLM controls when to stop. It will keep searching until it has
# MAGIC enough information, then produce a final answer.

# COMMAND ----------


class OptionsAgent:
    """Options trading assistant with tool-calling capability.

    The agent maintains a conversation history across turns.
    Each turn runs the full agentic loop (LLM → tools → LLM → ...)
    until the model produces a final answer.
    """

    # Tools available to the LLM
    _TOOL_SPECS = [SEARCH_TOOL_SPEC]

    def __init__(self) -> None:
        self.conversation_history: list[dict] = []

    # ------------------------------------------------------------------
    def _execute_tool(self, name: str, args: dict) -> str:
        """Dispatch a tool call by name and return the result as a string."""
        if name == "search_options_book":
            return search_options_book(**args)
        return json.dumps({"error": f"Unknown tool: {name}"})

    # ------------------------------------------------------------------
    def chat(self, question: str, max_iterations: int = 10) -> str:
        """Ask the agent a question.

        The agent will call tools as many times as needed before answering.

        Args:
            question: User's question about options trading.
            max_iterations: Safety cap on the tool-calling loop.

        Returns:
            The agent's final answer as a string.
        """
        # Start with system + prior conversation + new question
        messages: list[dict] = (
            [{"role": "system", "content": cfg.system_prompt}]
            + self.conversation_history
            + [{"role": "user", "content": question}]
        )

        final_answer = "No answer produced."

        for iteration in range(max_iterations):
            logger.info(f"[Agent] Iteration {iteration + 1}/{max_iterations}")

            response = llm_client.chat.completions.create(
                model=cfg.llm_endpoint,
                messages=messages,
                tools=self._TOOL_SPECS,
                temperature=0.2,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                # ── LLM wants to call one or more tools ──────────────
                # 1. Add the assistant's tool-call request to history
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

                # 2. Execute each tool and add the result to history
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    logger.info(f"[Agent] → tool: {tool_name}  args: {tool_args}")
                    result = self._execute_tool(tool_name, tool_args)
                    logger.info(f"[Agent] ← result: {len(result)} chars")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )

            else:
                # ── LLM produced a final answer ───────────────────────
                final_answer = msg.content
                logger.info(f"[Agent] Done after {iteration + 1} iteration(s)")
                break

        else:
            logger.warning("[Agent] Max iterations reached — returning partial answer")
            final_answer = msg.content or "Max iterations reached."

        # Persist this turn (question + answer only, not tool internals)
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": final_answer})

        return final_answer

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
        logger.info("[Agent] Conversation history cleared.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test the Agent
# MAGIC
# MAGIC Watch the log output — you'll see `[Agent] → tool: search_options_book`
# MAGIC when the LLM decides to search, and `[Agent] Done after N iteration(s)`
# MAGIC when it has enough information to answer.

# COMMAND ----------

agent = OptionsAgent()

# Test 1 — concept question (expect 1 tool call)
q1 = "What is the difference between a European and American option?"
logger.info("=" * 70)
logger.info(f"Q: {q1}")
logger.info("=" * 70)
a1 = agent.chat(q1)
logger.info(f"\nA: {a1}")

# COMMAND ----------

# Test 2 — Greeks (expect 1 tool call, possibly with a focused query)
q2 = "Explain gamma and how it relates to delta hedging."
logger.info("=" * 70)
logger.info(f"Q: {q2}")
logger.info("=" * 70)
a2 = agent.chat(q2)
logger.info(f"\nA: {a2}")

# COMMAND ----------

# Test 3 — strategy question (may trigger multiple searches)
q3 = "What is an iron condor and what market conditions is it suited for?"
logger.info("=" * 70)
logger.info(f"Q: {q3}")
logger.info("=" * 70)
a3 = agent.chat(q3)
logger.info(f"\nA: {a3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Multi-Turn Conversation
# MAGIC
# MAGIC Because `OptionsAgent` stores `conversation_history`, follow-up questions
# MAGIC work naturally — the agent remembers what was said earlier in the session.
# MAGIC
# MAGIC Notice that on follow-up turns the agent still calls the search tool with
# MAGIC a *more specific* query derived from the conversation context.

# COMMAND ----------

agent.clear()  # fresh session

# Turn 1 — broad question
t1 = "What is implied volatility?"
logger.info(f"[Turn 1] Q: {t1}")
r1 = agent.chat(t1)
logger.info(f"[Turn 1] A: {r1}\n")

# COMMAND ----------

# Turn 2 — follow-up (agent knows we were discussing implied volatility)
t2 = "How does it compare to realised volatility?"
logger.info(f"[Turn 2] Q: {t2}")
r2 = agent.chat(t2)
logger.info(f"[Turn 2] A: {r2}\n")

# COMMAND ----------

# Turn 3 — practical application
t3 = "When would a trader prefer one over the other when pricing options?"
logger.info(f"[Turn 3] Q: {t3}")
r3 = agent.chat(t3)
logger.info(f"[Turn 3] A: {r3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Agent Complete!
# MAGIC
# MAGIC What you now have:
# MAGIC
# MAGIC | Component | What it does |
# MAGIC |---|---|
# MAGIC | `search_options_book()` | Tool function — hybrid search over the vector index |
# MAGIC | `SEARCH_TOOL_SPEC` | JSON schema — tells the LLM when and how to call the tool |
# MAGIC | `OptionsAgent._execute_tool()` | Dispatches tool calls by name |
# MAGIC | `OptionsAgent.chat()` | Agentic loop — LLM decides, tools execute, repeats |
# MAGIC | `conversation_history` | Multi-turn memory — follow-ups work naturally |
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC - **Notebook 05** — Add MLflow tracing so every LLM call and tool execution
# MAGIC   is recorded as a span. This lets you inspect exactly what the agent
# MAGIC   retrieved and generated on every request.

# COMMAND ----------

logger.info("✓ Notebook 04 complete!")
logger.info(f"   LLM       : {cfg.llm_endpoint}")
logger.info(f"   Index     : {vs_manager.index_name}")
logger.info("   Tool      : search_options_book (hybrid vector search)")
logger.info("   Loop      : LLM → tool → LLM → ... → answer")
logger.info("\nReady for tracing in Notebook 05!")
