"""Vector search functionality for SolrCloud client."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pysolr
from loguru import logger

from solr_mcp.solr.interfaces import VectorSearchProvider
from solr_mcp.vector_provider import OllamaVectorProvider
from solr_mcp.vector_provider.constants import MODEL_DIMENSIONS

from ..exceptions import SchemaError, SolrError

if TYPE_CHECKING:
    from ..client import SolrClient

logger = logging.getLogger(__name__)


class VectorManager(VectorSearchProvider):
    """Vector search provider implementation."""

    def __init__(
        self,
        solr_client: "SolrClient",
        client: Optional[OllamaVectorProvider] = None,
        default_top_k: int = 10,
    ):
        """Initialize VectorManager.

        Args:
            solr_client: SolrClient instance
            client: Optional vector provider client (defaults to OllamaVectorProvider)
            default_top_k: Default number of results to return
        """
        self.solr_client = solr_client
        self.client = client or OllamaVectorProvider()
        self.default_top_k = default_top_k

    async def get_vector(
        self, text: str, vector_provider_config: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Get vector vector for text.

        Args:
            text: Text to get vector for
            vector_provider_config: Optional configuration for vector provider
                Can include 'model', 'base_url', etc.

        Returns:
            Vector as list of floats

        Raises:
            SolrError: If vector fails
        """
        if not self.client:
            raise SolrError("Vector operations unavailable - no vector provider client")

        try:
            # Create temporary client with custom config if needed
            if vector_provider_config and (
                "model" in vector_provider_config
                or "base_url" in vector_provider_config
            ):
                # Create a config with defaults from the existing client
                temp_config = {
                    "model": self.client.model,
                    "base_url": self.client.base_url,
                    "timeout": self.client.timeout,
                    "retries": self.client.retries,
                }
                # Override with provided config
                temp_config.update(vector_provider_config)

                # Create temporary client
                from solr_mcp.vector_provider import OllamaVectorProvider

                temp_client = OllamaVectorProvider(
                    model=temp_config["model"],
                    base_url=temp_config["base_url"],
                    timeout=temp_config["timeout"],
                    retries=temp_config["retries"],
                )

                # Use temporary client to get vector
                vector = await temp_client.get_vector(text)
            else:
                # Use the default client
                model = (
                    vector_provider_config.get("model")
                    if vector_provider_config
                    else None
                )
                vector = await self.client.get_vector(text, model)

            return vector
        except Exception as e:
            raise SolrError(f"Error getting vector: {str(e)}")

    def format_knn_query(
        self, vector: List[float], field: str, top_k: Optional[int] = None
    ) -> str:
        """Format KNN query for Solr.

        Args:
            vector: Query vector
            field: DenseVector field to search against
            top_k: Number of results to return (optional)

        Returns:
            Formatted KNN query string
        """
        # Format vector as string
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        # Build KNN query
        if top_k is not None:
            knn_template = "{{!knn f={field} topK={k}}}{vector}"
            return knn_template.format(field=field, k=int(top_k), vector=vector_str)
        else:
            knn_template = "{{!knn f={field}}}{vector}"
            return knn_template.format(field=field, vector=vector_str)

    async def find_vector_field(self, collection: str) -> str:
        """Find a suitable vector field for a collection.

        Args:
            collection: Collection name

        Returns:
            Name of the vector field

        Raises:
            SolrError: If no vector field is found
        """
        try:
            field = await self.solr_client.field_manager.find_vector_field(collection)
            return field
        except Exception as e:
            raise SolrError(f"Failed to find vector field: {str(e)}")

    async def validate_vector_field(
        self,
        collection: str,
        field: Optional[str],
        vector_provider_model: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Validate vector field and auto-detect if not provided.

        Args:
            collection: Collection name
            field: Optional field name, will auto-detect if None
            vector_provider_model: Optional model name

        Returns:
            Tuple of (field name, field info)

        Raises:
            SolrError: If field validation fails
        """
        try:
            # Auto-detect field if not provided
            if field is None:
                field = await self.find_vector_field(collection)

            # Validate field
            field_info = (
                await self.solr_client.field_manager.validate_vector_field_dimension(
                    collection=collection,
                    field=field,
                    vector_provider_model=vector_provider_model,
                    model_dimensions=MODEL_DIMENSIONS,
                )
            )

            return field, field_info
        except Exception as e:
            if isinstance(e, SchemaError):
                raise SolrError(str(e))
            raise SolrError(f"Failed to validate vector field: {str(e)}")

    async def execute_vector_search(
        self,
        client: pysolr.Solr,
        vector: List[float],
        field: str,
        top_k: Optional[int] = None,
        filter_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute vector similarity search.

        Args:
            client: pysolr.Solr client
            vector: Query vector
            field: DenseVector field to search against
            top_k: Number of results to return
            filter_query: Optional filter query

        Returns:
            Search results dictionary

        Raises:
            SolrError: If search fails
        """
        try:
            # Format KNN query
            knn_query = self.format_knn_query(vector, field, top_k)

            # Execute search
            results = client.search(
                knn_query,
                **{
                    "fl": "_docid_,score,_vector_distance_",  # Request _docid_ instead of id
                    "fq": filter_query if filter_query else None,
                },
            )

            # Convert pysolr Results to dict format
            if not isinstance(results, dict):
                return {
                    "responseHeader": {"QTime": getattr(results, "qtime", None)},
                    "response": {"numFound": results.hits, "docs": list(results)},
                }
            return results

        except Exception as e:
            raise SolrError(f"Vector search failed: {str(e)}")

    def extract_doc_ids(self, results: Dict[str, Any]) -> List[str]:
        """Extract document IDs from search results.

        Args:
            results: Search results dictionary

        Returns:
            List of document IDs
        """
        docs = results.get("response", {}).get("docs", [])
        return [doc["id"] for doc in docs if "id" in doc]
