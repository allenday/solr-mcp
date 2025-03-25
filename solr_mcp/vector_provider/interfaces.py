"""Interfaces for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class EmbeddingProvider(ABC):
    """Interface for generating embeddings."""

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
            EmbeddingConnectionError: If connection to service fails
        """
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors (list of floats)
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
            EmbeddingConnectionError: If connection to service fails
        """
        pass

    @property
    @abstractmethod
    def vector_dimension(self) -> int:
        """Get the dimension of vectors produced by this provider.
        
        Returns:
            Integer dimension of the embedding vectors
            
        Raises:
            EmbeddingConfigError: If unable to determine vector dimension
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