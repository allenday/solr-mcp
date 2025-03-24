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
        """Format KNN query for Solr.
        
        Args:
            vector: Query vector
            top_k: Number of results to return (optional)
            
        Returns:
            Formatted KNN query string
        """
        # Format vector as string
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        
        # Build KNN query
        if top_k is not None:
            knn_template = "{{!knn f={field} topK={k}}}{vector}"
            return knn_template.format(field=self.embedding_field, k=int(top_k), vector=vector_str)
        else:
            knn_template = "{{!knn f={field}}}{vector}"
            return knn_template.format(field=self.embedding_field, vector=vector_str)
        
    async def execute_vector_search(
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
                **{
                    'fl': '_docid_,score,_vector_distance_',  # Request _docid_ instead of id
                    'fq': filter_query if filter_query else None
                }
            )
            
            # Convert pysolr Results to dict format
            if not isinstance(results, dict):
                return {
                    'responseHeader': {'QTime': getattr(results, 'qtime', None)},
                    'response': {
                        'numFound': results.hits,
                        'docs': list(results)
                    }
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