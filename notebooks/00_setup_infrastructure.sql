-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Setup Infrastructure for Options Assistant
-- MAGIC
-- MAGIC This notebook creates the necessary volume for the Options Book project.
-- MAGIC We're using your existing **mlops_dev.rsreeram** schema.
-- MAGIC Run this first before uploading the PDF.

-- COMMAND ----------

-- Create the volume for storing PDFs
CREATE VOLUME IF NOT EXISTS mlops_dev.rsreeram.options_files;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ✅ Infrastructure Created!
-- MAGIC
-- MAGIC Now you can upload the PDF:
-- MAGIC 1. Go to Catalog → mlops_dev → rsreeram → options_files
-- MAGIC 2. Click "Upload" and select your options book PDF
-- MAGIC 3. Make sure the filename is exactly: `options_bok.pdf`

-- COMMAND ----------

-- Verify the setup
DESCRIBE SCHEMA mlops_dev.rsreeram;

-- COMMAND ----------

-- Show all objects in the schema
SHOW TABLES IN mlops_dev.rsreeram;

-- COMMAND ----------

-- Show volumes
SHOW VOLUMES IN mlops_dev.rsreeram;
