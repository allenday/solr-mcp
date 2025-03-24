"""Ollama embedding provider implementation."""

import os
from typing import Dict, List, Optional, Any

import httpx
from loguru import logger

from ..interfaces import EmbeddingProvider
from ..exceptions import EmbeddingGenerationError, EmbeddingConnectionError, EmbeddingConfigError
from ..constants import (
    DEFAULT_OLLAMA_CONFIG,
    ENV_OLLAMA_BASE_URL,
    ENV_OLLAMA_MODEL,
    OLLAMA_EMBEDDINGS_PATH,
    MODEL_DIMENSIONS
)

class OllamaClient(EmbeddingProvider):
    """Client for generating embeddings using Ollama API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        retries: Optional[int] = None
    ):
        """Initialize the Ollama client.
        
        Args:
            base_url: Base URL of the Ollama API
            model: Model name to use for embeddings
            timeout: Request timeout in seconds
            retries: Number of retries for failed requests
            
        Raises:
            EmbeddingConfigError: If configuration is invalid
        """
        self._base_url = base_url or os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_CONFIG["base_url"])
        self._model = model or os.environ.get(ENV_OLLAMA_MODEL, DEFAULT_OLLAMA_CONFIG["model"])
        self._timeout = timeout or DEFAULT_OLLAMA_CONFIG["timeout"]
        self._retries = retries or DEFAULT_OLLAMA_CONFIG["retries"]
        self._embeddings_endpoint = f"{self._base_url}{OLLAMA_EMBEDDINGS_PATH}"
        
        logger.info(
            f"Initialized Ollama client with model={self._model} "
            f"at {self._base_url} (timeout={self._timeout}s, retries={self._retries})"
        )
    
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
        for attempt in range(self._retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self._embeddings_endpoint,
                        json={"model": self._model, "prompt": text}
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["embedding"]
            except httpx.TimeoutError as e:
                logger.warning(f"Timeout getting embedding (attempt {attempt + 1}/{self._retries})")
                if attempt == self._retries - 1:
                    raise EmbeddingConnectionError(f"Timeout after {self._retries} attempts") from e
            except httpx.HTTPError as e:
                raise EmbeddingConnectionError(f"HTTP error: {str(e)}") from e
            except KeyError as e:
                raise EmbeddingGenerationError(f"Invalid response format: {str(e)}") from e
            except Exception as e:
                raise EmbeddingGenerationError(f"Unexpected error: {str(e)}") from e
    
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
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    @property
    def vector_dimension(self) -> int:
        """Get the dimension of vectors produced by this provider.
        
        Returns:
            Integer dimension of the embedding vectors
            
        Raises:
            EmbeddingConfigError: If unable to determine vector dimension
        """
        try:
            return MODEL_DIMENSIONS[self._model]
        except KeyError:
            raise EmbeddingConfigError(f"Unknown vector dimension for model {self._model}")

    @property
    def model_name(self) -> str:
        """Get the name of the model used by this provider.
        
        Returns:
            String name of the model
        """
        return self._model 