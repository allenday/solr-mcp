"""Interfaces for Solr client components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CollectionProvider(ABC):
    """Interface for providing collection information."""

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all available collections.

        Returns:
            List of collection names

        Raises:
            ConnectionError: If unable to retrieve collections
        """
        pass

    @abstractmethod
    async def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists.

        Args:
            collection: Name of the collection to check

        Returns:
            True if the collection exists, False otherwise

        Raises:
            ConnectionError: If unable to check collection existence
        """
        pass


class VectorSearchProvider(ABC):
    """Interface for vector search operations."""

    @abstractmethod
    def execute_vector_search(
        self, client: Any, vector: List[float], field: str, top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a vector similarity search.

        Args:
            client: Solr client instance
            vector: Dense vector for similarity search
            field: DenseVector field to search against
            top_k: Number of top results to return

        Returns:
            Search results as a dictionary

        Raises:
            SolrError: If vector search fails
        """
        pass

    @abstractmethod
    async def get_vector(self, text: str) -> List[float]:
        """Get vector for text.

        Args:
            text: Text to convert to vector

        Returns:
            Vector as list of floats

        Raises:
            SolrError: If vector generation fails
        """
        pass
