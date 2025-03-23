"""Interfaces for Solr client components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

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

class VectorSearchProvider(ABC):
    """Interface for vector search operations."""
    
    @abstractmethod
    def execute_vector_search(
        self,
        client: Any,
        vector: List[float],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a vector similarity search.
        
        Args:
            client: Solr client instance
            vector: Dense vector for similarity search
            top_k: Number of top results to return
            
        Returns:
            Search results as a dictionary
            
        Raises:
            SolrError: If vector search fails
        """
        pass
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get vector embedding for text.
        
        Args:
            text: Text to convert to vector
            
        Returns:
            Vector embedding as list of floats
            
        Raises:
            SolrError: If embedding generation fails
        """
        pass 