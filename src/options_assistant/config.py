"""Configuration management for Options Assistant."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


class ProjectConfig(BaseModel):
    """Project configuration model."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name for storing PDFs")
    llm_endpoint: str = Field(..., description="LLM endpoint name for question answering")
    embedding_endpoint: str = Field(..., description="Embedding endpoint name for vector search")
    warehouse_id: str = Field(..., description="Warehouse ID")
    vector_search_endpoint: str = Field(..., description="Vector search endpoint name")
    pdf_filename: str = Field("options_bok.pdf", description="Name of the options book PDF file")
    system_prompt: str = Field(
        default="You are a helpful AI assistant that answers questions about options trading based on the provided book content.",
        description="System prompt for the assistant"
    )
    experiment_name: str = Field(
        default="/Shared/options-assistant-dev",
        description="MLflow experiment name for tracing and evaluation"
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        return cls(**config_data[env])

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"

    @property
    def pdf_path(self) -> str:
        """Get full path to the PDF file in the volume."""
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/{self.pdf_filename}"


class ModelConfig(BaseModel):
    """Model configuration for LLM generation."""

    temperature: float = Field(0.7, description="Model temperature")
    max_tokens: int = Field(2000, description="Maximum tokens to generate")
    top_p: float = Field(0.95, description="Top-p sampling parameter")


class VectorSearchConfig(BaseModel):
    """Vector search configuration."""

    embedding_dimension: int = Field(1024, description="Embedding dimension (gte-large)")
    similarity_metric: str = Field("cosine", description="Similarity metric")
    num_results: int = Field(5, description="Number of results to return")
    enable_hybrid_search: bool = Field(True, description="Enable hybrid search (semantic + BM25)")
    enable_reranking: bool = Field(False, description="Enable reranking with cross-encoder")


class ChunkingConfig(BaseModel):
    """Chunking configuration for PDF text."""

    chunk_size: int = Field(512, description="Chunk size in tokens")
    chunk_overlap: int = Field(50, description="Overlap between chunks in tokens")
    separator: str = Field("\n\n", description="Separator for chunking (paragraph-based)")


def load_config(config_path: str = "project_config.yml", env: str = "dev") -> ProjectConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name (dev, acc, or prd)

    Returns:
        ProjectConfig instance
    """
    # Handle relative paths from notebooks
    if not Path(config_path).is_absolute():
        # Try to find config in parent directories
        current = Path.cwd()
        for _ in range(3):  # Search up to 3 levels
            candidate = current / config_path
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent

    return ProjectConfig.from_yaml(config_path, env)


def get_env(spark: SparkSession) -> str:
    """Get current environment from dbutils widget, falling back to 'dev'.

    Args:
        spark: Active SparkSession

    Returns:
        Environment name (dev, acc, or prd)
    """
    try:
        dbutils = DBUtils(spark)
        return dbutils.widgets.get("env")
    except Exception:
        return "dev"
