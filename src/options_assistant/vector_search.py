"""Vector search management for options book content."""

from typing import Any

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from options_assistant.config import ProjectConfig


class VectorSearchManager:
    """Manages vector search endpoints and indexes for options book chunks."""

    def __init__(
        self,
        config: ProjectConfig,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
        usage_policy_id: str | None = None,
    ) -> None:
        """Initialize VectorSearchManager.

        Args:
            config: ProjectConfig object
            endpoint_name: Name of the vector search endpoint (uses config if None)
            embedding_model: Name of the embedding model endpoint (uses config if None)
            usage_policy_id: ID of the usage policy for the endpoint (optional)
        """
        self.config = config
        self.endpoint_name = endpoint_name or config.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema
        self.usage_policy_id = usage_policy_id

        self.client = VectorSearchClient()
        self.index_name = f"{self.catalog}.{self.schema}.options_index"

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (
                ep.get("name")
                if isinstance(ep, dict)
                else getattr(ep, "name", None)
            )
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name,
                endpoint_type="STANDARD",
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"✓ Vector search endpoint exists: {self.endpoint_name}")

    def create_or_get_index(self) -> Any:
        """Create or get vector search index.

        Returns:
            Vector search index object
        """
        self.create_endpoint_if_not_exists()
        source_table = f"{self.catalog}.{self.schema}.options_chunks"

        # Try to get existing index
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return index
        except Exception:
            logger.info(f"Index {self.index_name} not found, will create it")

        # Try to create the index
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search index created: {self.index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            # Index exists but get_index failed earlier (transient) — retry
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return self.client.get_index(index_name=self.index_name)

    def sync_index(self) -> None:
        """Sync the vector search index with the source table."""
        index = self.create_or_get_index()
        logger.info(f"Syncing vector search index: {self.index_name}")
        index.sync()
        logger.info("✓ Index sync triggered")

    def search(
        self, query: str, num_results: int = 5, filters: dict | None = None
    ) -> dict:
        """Search the vector index.

        Args:
            query: Search query text
            num_results: Number of results to return
            filters: Optional filters to apply

        Returns:
            Search results dictionary
        """
        index = self.client.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            columns=["id", "text", "source", "element_type"],
            num_results=num_results,
            filters=filters,
        )
        return results
