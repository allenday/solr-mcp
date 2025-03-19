"""Client for interacting with Ollama to generate embeddings."""

import asyncio
import os
from typing import Dict, List, Optional, Union

import httpx
from loguru import logger


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: Optional[str] = None, model: str = "nomic-embed-text", timeout: float = 30.0):
        """Initialize the Ollama client.
        
        Args:
            base_url: Base URL of the Ollama API, defaults to http://localhost:11434
            model: Model name to use for embeddings
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.embeddings_endpoint = f"{self.base_url}/api/embeddings"
        self.timeout = timeout
        logger.info(f"Initialized Ollama client with model {model} at {self.base_url}")
    
    async def get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If the API request fails after all retries
        """
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    logger.debug(f"Sending embedding request to {self.embeddings_endpoint} (attempt {retries+1}/{max_retries})")
                    
                    # If text is empty or too short, use a placeholder
                    if not text or len(text.strip()) < 3:
                        text = "placeholder text for embedding"
                    
                    response = await client.post(
                        self.embeddings_endpoint,
                        json={"model": self.model, "prompt": text}
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if "embedding" not in data:
                        logger.warning(f"No embedding in response: {data}")
                        retries += 1
                        await asyncio.sleep(1)
                        continue
                        
                    logger.debug(f"Received embedding with {len(data['embedding'])} dimensions")
                    return data["embedding"]
            except Exception as e:
                last_error = e
                logger.warning(f"Error getting embedding (attempt {retries+1}/{max_retries}): {e}")
                retries += 1
                await asyncio.sleep(1)  # Wait before retrying
        
        logger.error(f"Failed to get embedding after {max_retries} attempts. Last error: {last_error}")
        raise last_error or Exception("Failed to get embedding after multiple attempts")
    
    async def get_embeddings(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            max_retries: Maximum number of retry attempts per text
            
        Returns:
            List of embedding vectors (list of floats)
            
        Raises:
            Exception: If any of the API requests fail after all retries
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                logger.debug(f"Getting embedding for text {i+1}/{len(texts)}")
                embedding = await self.get_embedding(text, max_retries=max_retries)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to get embedding for text {i+1}/{len(texts)}: {e}")
                # Generate a fallback embedding (zeros) for failed requests
                # This allows processing to continue even if some embeddings fail
                logger.warning(f"Using fallback embedding for text {i+1}")
                # Create a zero vector with the same dimensions as expected
                # For nomic-embed-text, this is 768 dimensions
                fallback_embedding = [0.0] * 768
                embeddings.append(fallback_embedding)
        
        return embeddings