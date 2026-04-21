# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search Setup for Options Book
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Understanding embeddings
# MAGIC - Creating vector search endpoints
# MAGIC - Creating and syncing vector search indexes
# MAGIC - Querying with similarity search
# MAGIC - Testing search quality

# COMMAND ----------

# MAGIC %pip install loguru

# COMMAND ----------

import sys

if "../src" not in sys.path:
    sys.path.insert(0, "../src")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from pyspark.sql import SparkSession

from options_assistant.config import get_env, load_config
from options_assistant.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

logger.info(f"Using catalog: {catalog}")
logger.info(f"Using schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Embeddings
# MAGIC
# MAGIC **Embeddings** are numerical representations of text that capture semantic meaning.
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC
# MAGIC - **Vector**: Array of numbers (e.g., [0.1, -0.3, 0.5, ...])
# MAGIC - **Dimension**: Length of the vector (we're using 1024 for gte-large-en)
# MAGIC - **Semantic Similarity**: Similar meanings = similar vectors
# MAGIC - **Distance Metrics**: Cosine similarity (range: -1 to 1)
# MAGIC
# MAGIC ### How it Works:
# MAGIC
# MAGIC ```
# MAGIC Text: "call option"
# MAGIC   ↓ (Embedding Model)
# MAGIC Vector: [0.23, -0.15, 0.67, ..., 0.42]  # 1024 dimensions
# MAGIC
# MAGIC Text: "buy option"
# MAGIC   ↓ (Embedding Model)
# MAGIC Vector: [0.25, -0.13, 0.65, ..., 0.40]  # Similar to above!
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vector Search Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │   Delta Table (options_chunks)          │
# MAGIC │  - id                                    │
# MAGIC │  - text                                  │
# MAGIC │  - source, element_type                 │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC                │ (Automatic sync)
# MAGIC                ↓
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │     Vector Search Index                  │
# MAGIC │  - Embeddings generated automatically    │
# MAGIC │  - Stored in optimized format            │
# MAGIC │  - Supports similarity search            │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC                │ (Query)
# MAGIC                ↓
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │     Search Results                       │
# MAGIC │  - Most similar chunks                   │
# MAGIC │  - With similarity scores                │
# MAGIC └─────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Vector Search Endpoint

# COMMAND ----------

# Using VectorSearchManager from options_assistant.vector_search
# This handles endpoint and index creation automatically

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint
)

logger.info(f"Vector Search Endpoint: {vs_manager.endpoint_name}")
logger.info(f"Embedding Model: {vs_manager.embedding_model}")
logger.info(f"Index Name: {vs_manager.index_name}")

# COMMAND ----------

# Create endpoint if it doesn't exist
vs_manager.create_endpoint_if_not_exists()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Endpoint Types:
# MAGIC
# MAGIC - **STANDARD**: General purpose, good performance (what we're using)
# MAGIC - **STANDARD_LARGE**: Higher throughput, more expensive
# MAGIC
# MAGIC For development and most workloads, STANDARD is sufficient.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Index

# COMMAND ----------

# Create or get the vector search index
# This automatically:
# - Creates the index if it doesn't exist
# - Configures it with the embedding model (databricks-gte-large-en)
# - Sets up delta sync with the options_chunks table

index = vs_manager.create_or_get_index()

logger.info("\n✓ Vector search setup complete!")
logger.info(f"  Index: {vs_manager.index_name}")
logger.info(f"  Source: {vs_manager.catalog}.{vs_manager.schema}.options_chunks")
logger.info(f"  Embedding Model: {vs_manager.embedding_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Configuration:
# MAGIC
# MAGIC - **pipeline_type**: `TRIGGERED` - Manual sync, good for batch processing
# MAGIC - **primary_key**: `id` - Unique identifier for each chunk
# MAGIC - **embedding_source_column**: `text` - The text to embed
# MAGIC - **embedding_model**: `databricks-gte-large-en` - Fast, high-quality embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sync the Index
# MAGIC
# MAGIC Trigger the index to process all chunks and generate embeddings

# COMMAND ----------

import time

# Wait for index to be ready for sync (pipeline must be in COMPLETED, FAILED, or CANCELED state)
max_wait_time = 600  # 10 minutes
wait_interval = 30  # 30 seconds
elapsed_time = 0

logger.info("Waiting for index pipeline to be ready for sync...")

while elapsed_time < max_wait_time:
    try:
        # Try to sync - if it fails because pipeline is running, wait and retry
        vs_manager.sync_index()
        logger.info("✓ Index sync triggered - embeddings are being generated")
        break
    except Exception as e:
        error_msg = str(e)
        if "not ready to sync yet" in error_msg and "RUNNING" in error_msg:
            logger.info(f"Pipeline is still RUNNING. Waiting {wait_interval}s before retry... ({elapsed_time}s elapsed)")
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        else:
            # Different error - raise it
            logger.error(f"Error syncing index: {e}")
            raise
else:
    logger.warning(f"⚠ Index pipeline did not become ready within {max_wait_time}s. You may need to sync it manually later.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Helper Function for Parsing Results

# COMMAND ----------

def parse_vector_search_results(results):
    """Parse vector search results from array format to dict format.

    Args:
        results: Raw results from similarity_search()

    Returns:
        List of dictionaries with column names as keys
    """
    columns = [col['name'] for col in results.get('manifest', {}).get('columns', [])]
    data_array = results.get('result', {}).get('data_array', [])

    return [dict(zip(columns, row_data)) for row_data in data_array]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Semantic Search
# MAGIC
# MAGIC ### How Semantic Search Works
# MAGIC
# MAGIC 1. **Query Embedding**: Convert your question to a vector
# MAGIC 2. **Similarity Calculation**: Compare query vector to all chunk vectors using cosine similarity
# MAGIC 3. **Ranking**: Return chunks with highest similarity scores
# MAGIC
# MAGIC ### Cosine Similarity Scores:
# MAGIC - **1.0**: Identical meaning
# MAGIC - **0.8-0.9**: Very similar
# MAGIC - **0.5-0.7**: Somewhat related
# MAGIC - **< 0.5**: Less relevant

# COMMAND ----------

# Test query about options
query = "What is a call option?"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "source", "element_type"],
    num_results=5
)

logger.info(f"Query: {query}\n")
logger.info("Top 5 Results:")
logger.info("=" * 80)

# Parse results using helper function
for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Source: {row.get('source', 'N/A')}")
    logger.info(f"   Element Type: {row.get('element_type', 'N/A')}")
    logger.info(f"   Chunk ID: {row.get('id', 'N/A')}")
    logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
    logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test More Queries

# COMMAND ----------

# Test query 2
query2 = "How does put option work?"

results2 = index.similarity_search(
    query_text=query2,
    columns=["text", "id"],
    num_results=3
)

logger.info(f"Query: {query2}\n")
logger.info("Top 3 Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results2), 1):
    logger.info(f"\n{i}. {row.get('text', '')[:300]}...")
    logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")

# COMMAND ----------

# Test query 3
query3 = "What is strike price?"

results3 = index.similarity_search(
    query_text=query3,
    columns=["text", "id"],
    num_results=3
)

logger.info(f"Query: {query3}\n")
logger.info("Top 3 Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results3), 1):
    logger.info(f"\n{i}. {row.get('text', '')[:300]}...")
    logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verify Index Status

# COMMAND ----------

# Get index details
client = VectorSearchClient()
index_info = client.get_index(index_name=vs_manager.index_name)

logger.info("Index Information:")
logger.info(f"  Name: {vs_manager.index_name}")

# Handle different index object structures
if hasattr(index_info, 'status'):
    if isinstance(index_info.status, dict):
        status = index_info.status.get('message', 'N/A')
    else:
        status = str(index_info.status)
    logger.info(f"  Status: {status}")

logger.info(f"  Endpoint: {vs_manager.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Vector Search Setup Complete!
# MAGIC
# MAGIC You now have:
# MAGIC - ✅ Vector search endpoint created
# MAGIC - ✅ Vector search index created and synced
# MAGIC - ✅ Embeddings generated for all chunks
# MAGIC - ✅ Semantic search working!
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC 1. **Build Q&A System** (Notebook 03): Use this vector search in a RAG pipeline
# MAGIC 2. **Try different queries**: Test various questions about options trading
# MAGIC 3. **Experiment with filters**: Filter by element_type or source
# MAGIC
# MAGIC Your options book is now searchable with semantic understanding!

# COMMAND ----------

logger.info("✓ Vector search setup complete!")
logger.info(f"   - Index: {vs_manager.index_name}")
logger.info("   - Chunks embedded: Ready for Q&A!")
logger.info("\nYou can now build your Q&A system using this vector search!")
