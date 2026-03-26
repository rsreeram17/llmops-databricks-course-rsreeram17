# Databricks notebook source
# MAGIC %md
# MAGIC # Debug Ingestion - Check What Tables Exist

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# List all tables in the schema
print("=== Tables in mlops_dev.rsreeram ===")
tables = spark.sql("SHOW TABLES IN mlops_dev.rsreeram").collect()
for table in tables:
    print(f"- {table.tableName}")

# COMMAND ----------

# Check if options_parsed_docs exists and has data
print("\n=== Checking options_parsed_docs ===")
try:
    parsed_df = spark.table("mlops_dev.rsreeram.options_parsed_docs")
    parsed_count = parsed_df.count()
    print(f"✓ Table exists")
    print(f"✓ Row count: {parsed_count}")

    if parsed_count > 0:
        print("\nShowing first row:")
        display(parsed_df.limit(1))
except Exception as e:
    print(f"✗ Error: {e}")

# COMMAND ----------

# Check if options_chunks exists and has data
print("\n=== Checking options_chunks ===")
try:
    chunks_df = spark.table("mlops_dev.rsreeram.options_chunks")
    chunks_count = chunks_df.count()
    print(f"✓ Table exists")
    print(f"✓ Row count: {chunks_count}")

    if chunks_count > 0:
        print("\nShowing first row:")
        display(chunks_df.limit(1))
    else:
        print("\n⚠️ Table exists but has 0 rows!")
except Exception as e:
    print(f"✗ Error: {e}")

# COMMAND ----------

# Check volume and files
print("\n=== Checking Volume Files ===")
try:
    files = spark.sql("LIST '/Volumes/mlops_dev/rsreeram/options_files/'").collect()
    print(f"Files in volume:")
    for file in files:
        print(f"  - {file.path} ({file.size} bytes)")
except Exception as e:
    print(f"✗ Error: {e}")

# COMMAND ----------

# If parsed_docs exists but chunks doesn't, let's manually run the chunking
print("\n=== Manual Chunking Test ===")
try:
    from options_assistant.config import load_config
    from options_assistant.pdf_processor import PDFProcessor

    cfg = load_config("../project_config.yml", "dev")
    processor = PDFProcessor(spark, cfg)

    # Try to process chunks
    print("Attempting to process chunks...")
    processor.process_chunks()
    print("✓ Chunking completed!")

    # Check count again
    chunks_df = spark.table("mlops_dev.rsreeram.options_chunks")
    chunks_count = chunks_df.count()
    print(f"New chunk count: {chunks_count}")

except Exception as e:
    print(f"✗ Error during manual chunking: {e}")
    import traceback
    traceback.print_exc()
