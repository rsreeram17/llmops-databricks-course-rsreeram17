# Databricks notebook source
# MAGIC %md
# MAGIC # View Options Book Chunks
# MAGIC
# MAGIC Simple notebook to explore the chunks extracted from your options book

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load the chunks table
chunks_df = spark.table("mlops_dev.rsreeram.options_chunks")

# Show basic info
print(f"Total chunks: {chunks_df.count()}")
print("\nTable schema:")
chunks_df.printSchema()

# COMMAND ----------

# Display first 10 chunks
display(chunks_df.limit(10))

# COMMAND ----------

# Show chunk statistics
print("=== Chunk Statistics ===")
chunks_df.select("element_type").groupBy("element_type").count().show()

# COMMAND ----------

# Show some sample text from chunks
print("=== Sample Chunks ===\n")
sample_chunks = chunks_df.select("chunk_id", "text", "element_type").limit(5).collect()

for i, row in enumerate(sample_chunks, 1):
    print(f"\n{'='*80}")
    print(f"Chunk {i}:")
    print(f"ID: {row['chunk_id']}")
    print(f"Type: {row['element_type']}")
    print(f"Text:\n{row['text'][:500]}...")
    print('='*80)

# COMMAND ----------

# Search for specific terms
search_term = "call option"  # Change this to search for different terms

print(f"Searching for chunks containing: '{search_term}'\n")
search_results = chunks_df.filter(chunks_df.text.contains(search_term))
print(f"Found {search_results.count()} chunks")

display(search_results.select("chunk_id", "text", "element_type").limit(5))

# COMMAND ----------

# Show longest and shortest chunks
from pyspark.sql.functions import length

chunks_with_length = chunks_df.withColumn("text_length", length("text"))

print("=== Longest Chunks ===")
display(chunks_with_length.orderBy("text_length", ascending=False).select("chunk_id", "text_length", "text").limit(5))

print("\n=== Shortest Chunks ===")
display(chunks_with_length.orderBy("text_length", ascending=True).select("chunk_id", "text_length", "text").limit(5))
