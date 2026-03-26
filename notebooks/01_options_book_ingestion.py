# Databricks notebook source
"""
Options Book Ingestion Notebook

This notebook handles the complete ingestion pipeline for the options trading book:
1. Upload the PDF to Databricks Volume
2. Parse the PDF with AI Parse Documents
3. Extract and clean text chunks
4. Store chunks in Delta table with Change Data Feed enabled

The processed chunks will be used for vector search in subsequent notebooks.
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from loguru import logger

from options_assistant.config import get_env, load_config
from options_assistant.pdf_processor import PDFProcessor

# COMMAND ----------

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load configuration based on environment
env = get_env(spark)
logger.info(f"Running in environment: {env}")

cfg = load_config("../project_config.yml", env)
logger.info(f"Configuration loaded for catalog: {cfg.catalog}, schema: {cfg.schema}")

# COMMAND ----------

# Create catalog and schema if they don't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
logger.info(f"✓ Catalog {cfg.catalog} ready")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.full_schema_name}")
logger.info(f"✓ Schema {cfg.full_schema_name} ready")

# COMMAND ----------

# Create volume for storing the PDF
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {cfg.full_volume_path}
""")
logger.info(f"✓ Volume {cfg.full_volume_path} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Upload PDF to Volume
# MAGIC
# MAGIC Upload the options book PDF from your local machine to the Databricks Volume.
# MAGIC You can do this in two ways:
# MAGIC
# MAGIC 1. **Via Databricks UI**: Navigate to Catalog > your_catalog > your_schema > options_files
# MAGIC    and use the UI to upload `options_bok.pdf`
# MAGIC
# MAGIC 2. **Via Code** (below): Provide the local path to the PDF file

# COMMAND ----------

# Option 1: Upload via code (uncomment and set the local path)
# LOCAL_PDF_PATH = "/Users/ramessre/Downloads/options_bok.pdf"
#
# processor = PDFProcessor(spark, cfg)
# processor.upload_pdf(LOCAL_PDF_PATH)
# logger.info(f"✓ PDF uploaded to {cfg.pdf_path}")

# COMMAND ----------

# Option 2: Verify the PDF exists in the volume (if you uploaded via UI)
from pyspark.dbutils import DBUtils

dbutils = DBUtils(spark)

try:
    files = dbutils.fs.ls(f"/Volumes/{cfg.catalog}/{cfg.schema}/{cfg.volume}")
    pdf_exists = any(f.name == cfg.pdf_filename for f in files)

    if pdf_exists:
        logger.info(f"✓ PDF file found: {cfg.pdf_filename}")
    else:
        logger.warning(f"⚠ PDF file not found: {cfg.pdf_filename}")
        logger.info("Available files:")
        for f in files:
            logger.info(f"  - {f.name}")
except Exception as e:
    logger.error(f"Error checking volume: {e}")
    logger.info(f"Please ensure the volume exists: {cfg.full_volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Parse PDF with AI Parse Documents
# MAGIC
# MAGIC Use Databricks AI Parse Documents to intelligently extract text from the PDF.
# MAGIC This handles complex layouts, tables, and multi-column text better than
# MAGIC traditional PDF parsing libraries.

# COMMAND ----------

processor = PDFProcessor(spark, cfg)
processor.parse_pdf_with_ai()
logger.info("✓ PDF parsing complete")

# COMMAND ----------

# Inspect the parsed content
parsed_df = spark.table(f"{cfg.catalog}.{cfg.schema}.options_parsed_docs")
display(parsed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Extract and Clean Chunks
# MAGIC
# MAGIC Process the parsed JSON to extract text chunks, clean them, and store in a
# MAGIC Delta table ready for vector search.

# COMMAND ----------

processor.process_chunks()
logger.info("✓ Chunk processing complete")

# COMMAND ----------

# Inspect the chunks
chunks_df = spark.table(f"{cfg.catalog}.{cfg.schema}.options_chunks")
display(chunks_df)

# COMMAND ----------

# Show chunk statistics
logger.info("=== Chunk Statistics ===")
chunk_count = chunks_df.count()
logger.info(f"Total chunks: {chunk_count}")

# Average chunk length
from pyspark.sql.functions import length, avg, min as spark_min, max as spark_max

stats = chunks_df.select(
    avg(length("text")).alias("avg_length"),
    spark_min(length("text")).alias("min_length"),
    spark_max(length("text")).alias("max_length"),
).collect()[0]

logger.info(f"Average chunk length: {stats['avg_length']:.2f} characters")
logger.info(f"Min chunk length: {stats['min_length']} characters")
logger.info(f"Max chunk length: {stats['max_length']} characters")

# Chunks by element type
logger.info("\n=== Chunks by Element Type ===")
element_counts = chunks_df.groupBy("element_type").count().orderBy("count", ascending=False)
display(element_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification
# MAGIC
# MAGIC Verify that Change Data Feed is enabled for vector search sync

# COMMAND ----------

# Check table properties
table_details = spark.sql(f"""
    DESCRIBE DETAIL {cfg.catalog}.{cfg.schema}.options_chunks
""")
display(table_details)

# Verify Change Data Feed is enabled
cdf_enabled = spark.sql(f"""
    SHOW TBLPROPERTIES {cfg.catalog}.{cfg.schema}.options_chunks
""").filter("key = 'delta.enableChangeDataFeed'").collect()

if cdf_enabled and cdf_enabled[0]["value"] == "true":
    logger.info("✓ Change Data Feed is enabled")
else:
    logger.warning("⚠ Change Data Feed is not enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Create Vector Search Index**: Run notebook `02_vector_search_setup.py`
# MAGIC 2. **Build Q&A System**: Run notebook `03_options_qa_system.py`
# MAGIC
# MAGIC The chunks are now ready to be indexed for semantic search!

# COMMAND ----------

logger.info("✓ Options book ingestion complete!")
logger.info(f"   - PDF parsed: {cfg.pdf_filename}")
logger.info(f"   - Chunks created: {chunk_count}")
logger.info(f"   - Table: {cfg.catalog}.{cfg.schema}.options_chunks")
logger.info("   - Change Data Feed: Enabled")
logger.info("\nReady for vector search indexing!")
