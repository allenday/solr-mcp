"""Client for interacting with Ollama to generate embeddings."""

import os
from typing import Dict, List, Optional, Union

import httpx
from loguru import logger


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: Optional[str] = None, model: str = "nomic-embed-text"):
        """Initialize the Ollama client.
        
        Args:
            base_url: Base URL of the Ollama API, defaults to http://localhost:11434
            model: Model name to use for embeddings
        """
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.embeddings_endpoint = f"{self.base_url}/api/embeddings"
        logger.info(f"Initialized Ollama client with model {model} at {self.base_url}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If the API request fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.embeddings_endpoint,
                    json={"model": self.model, "prompt": text}
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors (list of floats)
            
        Raises:
            Exception: If any of the API requests fail
        """
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings