"""Vector search functionality for SolrCloud client."""

import logging
from typing import List, Optional, Dict, Any

from loguru import logger
import numpy as np
import pysolr

from solr_mcp.embeddings import OllamaClient
from solr_mcp.solr.exceptions import SolrError
from solr_mcp.solr.interfaces import VectorSearchProvider

logger = logging.getLogger(__name__)

class VectorManager(VectorSearchProvider):
    """Vector search provider implementation."""

    def __init__(self, solr, client=None):
        """Initialize VectorManager.
        
        Args:
            solr: Solr client instance
            client: Optional client for vector operations (e.g. Ollama)
        """
        self.solr = solr
        self.client = client or OllamaClient()
        self.embedding_field = "embedding"
        self.default_top_k = 10
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get vector embedding for text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Vector embedding as list of floats
            
        Raises:
            SolrError: If embedding fails
        """
        if not self.client:
            raise SolrError("Vector operations unavailable - no Ollama client")
            
        try:
            embedding = await self.client.get_embedding(text)
            return embedding
        except Exception as e:
            raise SolrError(f"Error getting embedding: {str(e)}")
            
    def format_knn_query(self, vector: List[float], top_k: Optional[int] = None) -> str:
        """Format KNN query for vector similarity search.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            
        Returns:
            Formatted KNN query string
        """
        vector_str = "[" + ",".join(str(x) for x in vector) + "]"
        k = top_k or self.default_top_k
        return f"{{!knn f={self.embedding_field} topK={k}}}{vector_str}"
        
    def execute_vector_search(
        self,
        client: pysolr.Solr,
        vector: List[float],
        top_k: Optional[int] = None,
        filter_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute vector similarity search.
        
        Args:
            client: pysolr.Solr client
            vector: Query vector
            top_k: Number of results to return
            filter_query: Optional filter query
            
        Returns:
            Search results dictionary
            
        Raises:
            SolrError: If search fails
        """
        try:
            # Format KNN query
            knn_query = self.format_knn_query(vector, top_k)
            
            # Execute search
            results = client.search(
                knn_query,
                fq=filter_query,
                fl=f"id,score,{self.embedding_field}"
            )
            
            if not isinstance(results, dict):
                results = results.raw_response
                
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