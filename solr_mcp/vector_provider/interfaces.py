"""Interfaces for vector providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorProvider(ABC):
    """Interface for generating vectors for semantic search."""

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get vector embedding for a single text.
        
        Args:
            text: Text to generate vector for
            
        Returns:
            List of floats representing the vector
            
        Raises:
            VectorGenerationError: If vector generation fails
            VectorConnectionError: If connection to service fails
        """
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get vector embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate vectors for
            
        Returns:
            List of vectors (list of floats)
            
        Raises:
            VectorGenerationError: If vector generation fails
            VectorConnectionError: If connection to service fails
        """
        pass

    @property
    @abstractmethod
    def vector_dimension(self) -> int:
        """Get the dimension of vectors produced by this provider.
        
        Returns:
            Integer dimension of the vectors
            
        Raises:
            VectorConfigError: If unable to determine vector dimension
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model used by this provider.
        
        Returns:
            String name of the model
        """
        pass 