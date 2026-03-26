# Databricks notebook source
# MAGIC %md
# MAGIC # Quick View of Options Chunks
# MAGIC
# MAGIC Run this to quickly verify your chunks were created successfully

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import length, count, col

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load the chunks table
chunks_df = spark.table("mlops_dev.rsreeram.options_chunks")

# Get total count
total_chunks = chunks_df.count()
print(f"✅ Total chunks in table: {total_chunks:,}")

# COMMAND ----------

# Show statistics
if total_chunks > 0:
    print("\n📊 Chunk Statistics:")
    print("=" * 80)

    # Count by element type
    print("\nChunks by type:")
    chunks_df.groupBy("element_type").count().orderBy("count", ascending=False).show()

    # Text length statistics
    stats = chunks_df.select(
        length("text").alias("text_length")
    ).describe().show()

    print("\n📄 Sample Chunks:")
    print("=" * 80)

    # Show first 5 chunks
    sample_chunks = chunks_df.select("chunk_id", "element_type", "text").limit(5).collect()

    for i, row in enumerate(sample_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"ID: {row['chunk_id']}")
        print(f"Type: {row['element_type']}")
        print(f"Text preview: {row['text'][:200]}...")
        print()
else:
    print("\n⚠️  No chunks found! The ingestion may have failed.")

# COMMAND ----------

# Display full dataframe view
display(chunks_df.limit(10))
