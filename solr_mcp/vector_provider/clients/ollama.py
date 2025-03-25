"""Ollama vector provider implementation."""

from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from solr_mcp.solr.interfaces import VectorSearchProvider
from solr_mcp.vector_provider.constants import MODEL_DIMENSIONS, OLLAMA_EMBEDDINGS_PATH


class OllamaVectorProvider(VectorSearchProvider):
    """Vector provider that uses Ollama to vectorize text."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        retries: int = 3,
    ):
        """Initialize the Ollama vector provider.

        Args:
            model: Name of the Ollama model to use
            base_url: Base URL of the Ollama server
            timeout: Request timeout in seconds
            retries: Number of retries for failed requests
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        logger.info(
            f"Initialized Ollama vector provider with model={model} at {base_url} (timeout={timeout}s, retries={retries})"
        )

    async def get_vector(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get vector for a single text.

        Args:
            text: Text to get vector for
            model: Optional model to use for vectorization (overrides default)

        Returns:
            List of floats representing the text vector

        Raises:
            Exception: If there is an error getting vector
        """
        url = f"{self.base_url}{OLLAMA_EMBEDDINGS_PATH}"
        data = {"model": model or self.model, "prompt": text}

        actual_model = data["model"]

        for attempt in range(self.retries + 1):
            try:
                response = requests.post(url, json=data, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["embedding"]
            except Exception as e:
                if attempt == self.retries:
                    raise Exception(
                        f"Failed to get vector with model {actual_model} after {self.retries} retries: {str(e)}"
                    )
                logger.warning(
                    f"Failed to get vector with model {actual_model} (attempt {attempt + 1}/{self.retries + 1}): {str(e)}"
                )
                continue

    async def get_vectors(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Get vector for multiple texts.

        Args:
            texts: List of texts to get vector for
            model: Optional model to use for vectorization (overrides default)

        Returns:
            List of vectors (list of floats)

        Raises:
            Exception: If there is an error getting vector
        """
        results = []
        for text in texts:
            vector = await self.get_vector(text, model)
            results.append(vector)
        return results

    async def execute_vector_search(
        self, client: Any, vector: List[float], top_k: int = 10
    ) -> Dict[str, Any]:
        """Execute vector similarity search.

        Args:
            client: Solr client instance
            vector: Query vector
            top_k: Number of results to return

        Returns:
            Dictionary containing search results

        Raises:
            Exception: If there is an error executing the search
        """
        try:
            # Build KNN query
            knn_query = {
                "q": "*:*",
                "knn": f"{{!knn f=vector topK={top_k}}}[{','.join(str(x) for x in vector)}]",
            }

            # Execute search
            results = client.search(**knn_query)
            return results

        except Exception as e:
            raise Exception(f"Vector search failed: {str(e)}")

    @property
    def vector_dimension(self) -> int:
        """Get the dimension of vectors produced by this provider.

        Returns:
            Integer dimension of the vectors
        """
        return MODEL_DIMENSIONS.get(
            self.model, 768
        )  # Default to 768 if model not found

    @property
    def model_name(self) -> str:
        """Get the name of the model used by this provider.

        Returns:
            String name of the model
        """
        return self.model
