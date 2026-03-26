"""
PDF Processing Pipeline for Options Book:
   ↓ (upload_pdf)
PDF in Volume
   ↓ (parse_pdf_with_ai)
options_parsed_docs table (JSON)
   ↓ (process_chunks)
options_chunks table (clean text + metadata)
   ↓ (VectorSearchManager - separate class)
Vector Search Index (embeddings)
"""

import json
import re
from typing import Any

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col,
    current_timestamp,
    explode,
    lit,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from options_assistant.config import ProjectConfig


class PDFProcessor:
    """
    PDFProcessor handles the complete workflow of:
    - Uploading the options book PDF to Unity Catalog Volume
    - Parsing the PDF with ai_parse_document
    - Extracting and cleaning text chunks
    - Saving chunks to Delta tables
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize PDFProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.volume = config.volume

        # Table names
        self.parsed_table = f"{self.catalog}.{self.schema}.options_parsed_docs"
        self.chunks_table = f"{self.catalog}.{self.schema}.options_chunks"

        # PDF path in volume
        self.pdf_path = config.pdf_path

    def upload_pdf(self, local_pdf_path: str) -> None:
        """
        Upload PDF from local path to Databricks Volume.

        Args:
            local_pdf_path: Local path to the options book PDF

        Note:
            This will use dbutils.fs.cp to copy the file
        """
        volume_path = self.pdf_path
        logger.info(f"Uploading PDF from {local_pdf_path} to {volume_path}")

        # Use dbutils to copy file
        from pyspark.dbutils import DBUtils

        dbutils = DBUtils(self.spark)
        dbutils.fs.cp(f"file://{local_pdf_path}", volume_path)

        logger.info(f"Successfully uploaded PDF to {volume_path}")

    def parse_pdf_with_ai(self) -> None:
        """
        Parse PDF using ai_parse_document and store in options_parsed_docs table.
        """
        logger.info(f"Parsing PDF from {self.pdf_path}")

        # Create table if it doesn't exist
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                pdf_filename STRING,
                parsed_content STRING,
                parsed_at TIMESTAMP
            )
            USING DELTA
        """)

        # Check if already parsed
        existing_count = self.spark.sql(f"""
            SELECT COUNT(*) as cnt
            FROM {self.parsed_table}
            WHERE pdf_filename = '{self.cfg.pdf_filename}'
        """).collect()[0]["cnt"]

        if existing_count > 0:
            logger.info(
                f"PDF {self.cfg.pdf_filename} already parsed. "
                "Skipping parsing step."
            )
            return

        # Parse the PDF
        pdf_volume_path = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

        self.spark.sql(f"""
            INSERT INTO {self.parsed_table}
            SELECT
                '{self.cfg.pdf_filename}' as pdf_filename,
                ai_parse_document(content) AS parsed_content,
                current_timestamp() AS parsed_at
            FROM READ_FILES(
                "{pdf_volume_path}",
                format => 'binaryFile'
            )
            WHERE path LIKE '%{self.cfg.pdf_filename}'
        """)

        logger.info(f"Parsed PDF and saved to {self.parsed_table}")

    @staticmethod
    def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str, str]]:
        """
        Extract chunks from parsed_content JSON.

        Args:
            parsed_content_json: JSON string containing parsed document structure

        Returns:
            List of tuples containing (chunk_id, content, element_type)
        """
        try:
            parsed_dict = json.loads(parsed_content_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON content")
            return []

        chunks = []

        for element in parsed_dict.get("document", {}).get("elements", []):
            element_type = element.get("type", "text")
            if element_type in ["text", "paragraph", "heading"]:
                chunk_id = element.get("id", "")
                content = element.get("content", "")
                if content.strip():  # Only include non-empty chunks
                    chunks.append((chunk_id, content, element_type))

        return chunks

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """
        Clean and normalize chunk text.

        Args:
            text: Raw text content

        Returns:
            Cleaned text content
        """
        # Fix hyphenation across line breaks:
        # "docu-\nments" => "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)

        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)

        return t.strip()

    def process_chunks(self) -> None:
        """
        Process parsed documents to extract and clean chunks.
        Reads from options_parsed_docs table and saves to options_chunks table.
        """
        logger.info(f"Processing parsed documents from {self.parsed_table}")

        df = self.spark.table(self.parsed_table)

        # Define schema for the extracted chunks
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                    StructField("element_type", StringType(), True),
                ]
            )
        )

        extract_chunks_udf = udf(self._extract_chunks, chunk_schema)
        clean_chunk_udf = udf(self._clean_chunk, StringType())

        # Create the transformed dataframe
        chunks_df = (
            df.withColumn("chunks", extract_chunks_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                lit("options_bok").alias("source"),
                col("pdf_filename"),
                col("chunk.chunk_id").alias("chunk_id"),
                col("chunk.element_type").alias("element_type"),
                clean_chunk_udf(col("chunk.content")).alias("text"),
                col("chunk.chunk_id").alias("id"),
                current_timestamp().alias("processed_at"),
            )
            .filter(col("text") != "")  # Remove empty chunks
        )

        # Create table if it doesn't exist
        chunks_df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(self.chunks_table)

        logger.info(f"Saved chunks to {self.chunks_table}")

        # Enable Change Data Feed for vector search sync
        self.spark.sql(f"""
            ALTER TABLE {self.chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"Change Data Feed enabled for {self.chunks_table}")

        # Show statistics
        chunk_count = chunks_df.count()
        logger.info(f"Total chunks created: {chunk_count}")

    def process_complete_pipeline(self, local_pdf_path: str | None = None) -> None:
        """
        Complete workflow: upload PDF, parse, and process chunks.

        Args:
            local_pdf_path: Optional local path to PDF. If None, assumes PDF
                          already exists in the volume.
        """
        # Step 1: Upload PDF (if local path provided)
        if local_pdf_path:
            self.upload_pdf(local_pdf_path)

        # Step 2: Parse PDF with ai_parse_document
        self.parse_pdf_with_ai()
        logger.info("PDF parsing complete.")

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")
