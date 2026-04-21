# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03: Simple RAG — Options Q&A System
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is RAG (Retrieval-Augmented Generation)?
# MAGIC - Using our vector search index for document retrieval
# MAGIC - Enriching prompts with retrieved context
# MAGIC - Building a complete Q&A system for the options trading book
# MAGIC - Multi-turn conversation with history
# MAGIC
# MAGIC **RAG Flow:**
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC Vector Search (retrieve relevant chunks from the options book)
# MAGIC     ↓
# MAGIC Build Prompt (question + retrieved context)
# MAGIC     ↓
# MAGIC LLM (generate grounded answer)
# MAGIC     ↓
# MAGIC Answer (with source references)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install loguru openai

# COMMAND ----------

# Add the local src package to the path so options_assistant is importable
# when running interactively (jobs install the wheel automatically via the
# bundle job definition, so this is a no-op in that context).
import sys

if "../src" not in sys.path:
    sys.path.insert(0, "../src")

# COMMAND ----------

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
logger.info(f"Embedding endpoint: {cfg.embedding_endpoint}")

# COMMAND ----------

# WorkspaceClient picks up credentials automatically inside Databricks.
# We generate a short-lived token so the OpenAI-compatible client can reach
# the Databricks serving endpoint.
w = WorkspaceClient()

llm_client = OpenAI(
    api_key=w.tokens.create(lifetime_seconds=1200).token_value,
    base_url=f"{w.config.host}/serving-endpoints",
)

logger.info(f"✓ LLM client ready → {w.config.host}/serving-endpoints")

# COMMAND ----------

# Reuse the same VectorSearchManager that notebook 02 used to build the index.
vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint,
)

logger.info(f"✓ Vector search index: {vs_manager.index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Retrieval
# MAGIC
# MAGIC A single helper converts a plain-text question into a ranked list of chunks
# MAGIC from the options book via **hybrid search** (semantic cosine similarity + BM25
# MAGIC keyword matching). Hybrid gives better recall than pure semantic search for
# MAGIC domain-specific terms like "delta", "theta", or "Black-Scholes".

# COMMAND ----------


def parse_search_results(results: dict) -> list[dict]:
    """Parse the raw similarity_search() response into a list of dicts.

    Databricks Vector Search returns results as parallel arrays:
    - manifest.columns → list of column name objects
    - result.data_array → list of rows (each row is a list of values)

    The score is appended automatically as the last column.
    """
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    rows = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row, strict=False)) for row in rows]


def retrieve_chunks(query: str, num_results: int = 5) -> list[dict]:
    """Retrieve the most relevant chunks from the options book.

    Args:
        query: Natural-language question.
        num_results: Number of chunks to return.

    Returns:
        List of dicts with keys: id, text, source, element_type, score.
    """
    index = vs_manager.client.get_index(index_name=vs_manager.index_name)
    raw = index.similarity_search(
        query_text=query,
        columns=["id", "text", "source", "element_type"],
        num_results=num_results,
        query_type="hybrid",
    )
    return parse_search_results(raw)


# COMMAND ----------

# Smoke-test: verify retrieval is working before wiring up the LLM
_test_query = "What is a call option?"
_test_chunks = retrieve_chunks(_test_query, num_results=3)

logger.info(f"Retrieval smoke-test — query: '{_test_query}'")
logger.info(f"Retrieved {len(_test_chunks)} chunks")
logger.info("=" * 70)
for i, chunk in enumerate(_test_chunks, 1):
    score = chunk.get("score")
    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
    logger.info(
        f"{i}. source={chunk.get('source', 'N/A')} | "
        f"type={chunk.get('element_type', 'N/A')} | "
        f"score={score_str}"
    )
    logger.info(f"   {chunk.get('text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prompt Building
# MAGIC
# MAGIC Retrieved chunks are injected as labelled context blocks.
# MAGIC The LLM is instructed to answer **only** from that context, and to cite which
# MAGIC chunk supports each claim — this makes hallucinations visible and auditable.

# COMMAND ----------


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """Construct the user-turn prompt that carries the retrieved context.

    Args:
        question: The user's question.
        chunks: Chunks returned by retrieve_chunks().

    Returns:
        Formatted prompt string to pass as the user message.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "options book")
        element_type = chunk.get("element_type", "text")
        text = chunk.get("text", "")
        context_parts.append(
            f"[Chunk {i}] source: {source} | type: {element_type}\n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    instructions = (
        "Use the following excerpts from the options trading book to answer.\n"
        "If the context does not contain enough information, say so clearly.\n"
        'Always cite which chunk (e.g. "Chunk 2") supports each claim.'
    )
    return f"""{instructions}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# COMMAND ----------

# Inspect the assembled prompt so we can see exactly what the LLM receives
_sample_prompt = build_rag_prompt(_test_query, _test_chunks)
logger.info(f"Prompt length: {len(_sample_prompt)} characters")
logger.info(f"\nPrompt preview (first 700 chars):\n{_sample_prompt[:700]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Full RAG Query
# MAGIC
# MAGIC Wraps retrieve → augment → generate into a single function.
# MAGIC Temperature is set low (0.2) because this is a factual retrieval task —
# MAGIC we want the LLM to stay close to what the book says.

# COMMAND ----------


def rag_query(question: str, num_chunks: int = 5) -> dict:
    """Answer a single question using the RAG pipeline.

    Args:
        question: User's question about options trading.
        num_chunks: Number of chunks to retrieve.

    Returns:
        Dict with keys: question, answer, sources (list of source strings).
    """
    # 1 — Retrieve
    logger.info(f"[RAG] Retrieving for: '{question}'")
    chunks = retrieve_chunks(question, num_results=num_chunks)
    logger.info(f"[RAG] {len(chunks)} chunks retrieved")

    # 2 — Build prompt
    prompt = build_rag_prompt(question, chunks)

    # 3 — Generate
    logger.info("[RAG] Calling LLM...")
    response = llm_client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=[
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    # Deduplicate sources
    sources = sorted(
        {chunk.get("source", "unknown") for chunk in chunks if chunk.get("source")}
    )

    return {"question": question, "answer": answer, "sources": sources}


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test the RAG System
# MAGIC
# MAGIC Three questions covering fundamental concepts, the Greeks, and volatility.

# COMMAND ----------

# Question 1 — Fundamental concept
result_1 = rag_query("What is the difference between a call option and a put option?")

logger.info("=" * 70)
logger.info(f"Q: {result_1['question']}")
logger.info("=" * 70)
logger.info(f"\n{result_1['answer']}")
logger.info(f"\nSources: {result_1['sources']}")

# COMMAND ----------

# Question 2 — The Greeks
result_2 = rag_query(
    "How does delta measure an option's sensitivity to the underlying price?"
)

logger.info("=" * 70)
logger.info(f"Q: {result_2['question']}")
logger.info("=" * 70)
logger.info(f"\n{result_2['answer']}")
logger.info(f"\nSources: {result_2['sources']}")

# COMMAND ----------

# Question 3 — Time decay
result_3 = rag_query(
    "What is theta decay and how does it affect an option's value over time?"
)

logger.info("=" * 70)
logger.info(f"Q: {result_3['question']}")
logger.info("=" * 70)
logger.info(f"\n{result_3['answer']}")
logger.info(f"\nSources: {result_3['sources']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Multi-Turn Conversation — SimpleOptionsRAG
# MAGIC
# MAGIC `rag_query()` answers each question in isolation. `SimpleOptionsRAG` adds a
# MAGIC **conversation history** so follow-up questions resolve correctly without the
# MAGIC user needing to repeat context.
# MAGIC
# MAGIC **Design:**
# MAGIC - Fresh chunks are retrieved for the *current* question every turn (retrieval
# MAGIC   stays relevant even if the topic drifts).
# MAGIC - The full conversation history is included in the message list so the LLM
# MAGIC   maintains coherence across turns.
# MAGIC - Context is injected into the **system message** (not the user turn), keeping
# MAGIC   the message format clean for future agent integration.
# MAGIC
# MAGIC ```
# MAGIC Turn 1: retrieve(q1) → [system(ctx1) + user(q1)]                 → answer1
# MAGIC Turn 2: retrieve(q2) → [system(ctx2) + user(q1) + asst(a1) + user(q2)] → answer2
# MAGIC ```

# COMMAND ----------


class SimpleOptionsRAG:
    """Stateful RAG assistant for the options trading book.

    Maintains an in-memory conversation history across multiple turns.
    Each turn retrieves fresh context for the current question and passes
    the full history to the LLM so prior turns remain accessible.
    """

    def __init__(self, num_chunks: int = 5) -> None:
        self.num_chunks = num_chunks
        self.conversation_history: list[dict] = []

    # ------------------------------------------------------------------
    def chat(self, question: str) -> str:
        """Ask a question, optionally building on previous turns.

        Args:
            question: The user's current question.

        Returns:
            The assistant's answer as a string.
        """
        # 1 — Retrieve fresh context for this question
        chunks = retrieve_chunks(question, num_results=self.num_chunks)

        context = "\n\n---\n\n".join(
            f"[Chunk {i}] source: {c.get('source', 'unknown')}\n{c.get('text', '')}"
            for i, c in enumerate(chunks, 1)
        )

        # 2 — System message: assistant persona + injected context
        rag_instructions = (
            "Answer from the options book excerpts below. "
            "If context is insufficient, say so. "
            "Cite chunks when making claims."
        )
        system_content = (
            f"{cfg.system_prompt}\n\n{rag_instructions}\n\nCONTEXT:\n{context}"
        )

        # 3 — Build message list: system + history + current question
        messages = (
            [{"role": "system", "content": system_content}]
            + self.conversation_history
            + [{"role": "user", "content": question}]
        )

        # 4 — Call LLM
        response = llm_client.chat.completions.create(
            model=cfg.llm_endpoint,
            messages=messages,
            max_tokens=1000,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

        # 5 — Append this turn to history for future turns
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset conversation history to start a fresh session."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Multi-Turn Demo
# MAGIC
# MAGIC Three turns on the topic of volatility — each turn builds on the previous.

# COMMAND ----------

rag = SimpleOptionsRAG(num_chunks=5)

# Turn 1 — foundational question
q1 = "What is implied volatility and why does it matter for options pricing?"
a1 = rag.chat(q1)
logger.info(f"[Turn 1] Q: {q1}")
logger.info(f"[Turn 1] A: {a1}\n")

# COMMAND ----------

# Turn 2 — follow-up that relies on what was just answered
q2 = "How does implied volatility differ from historical volatility?"
a2 = rag.chat(q2)
logger.info(f"[Turn 2] Q: {q2}")
logger.info(f"[Turn 2] A: {a2}\n")

# COMMAND ----------

# Turn 3 — deeper follow-up; "it" refers to the conversation above
q3 = "Can you explain the volatility smile and when it typically appears?"
a3 = rag.chat(q3)
logger.info(f"[Turn 3] Q: {q3}")
logger.info(f"[Turn 3] A: {a3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Simple RAG Complete!
# MAGIC
# MAGIC You now have a fully working Q&A system grounded in the options book:
# MAGIC
# MAGIC | Component | What it does |
# MAGIC |---|---|
# MAGIC | `retrieve_chunks()` | Hybrid semantic + BM25 search over the vector index |
# MAGIC | `build_rag_prompt()` | Injects retrieved context into the LLM prompt |
# MAGIC | `rag_query()` | Single-turn RAG: retrieve → augment → generate |
# MAGIC | `SimpleOptionsRAG` | Multi-turn: adds conversation history across turns |
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC - **Notebook 04** — Build a proper agent with tool calling so the LLM can
# MAGIC   *decide* when to call vector search vs. answer from its own knowledge.
# MAGIC - **Notebook 05** — Add MLflow tracing to observe what the agent retrieves and
# MAGIC   generates on every request.

# COMMAND ----------

logger.info("✓ Notebook 03 complete!")
logger.info(f"   Index     : {vs_manager.index_name}")
logger.info(f"   LLM       : {cfg.llm_endpoint}")
logger.info("   Pipeline  : retrieve → augment → generate")
logger.info("\nReady to build the agent in Notebook 04!")
