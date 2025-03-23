"""SolrCloud client implementation."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import pysolr
import requests
from loguru import logger

from solr_mcp.solr.config import SolrConfig
from solr_mcp.solr.exceptions import SolrError, ConnectionError, QueryError
from solr_mcp.solr.schema import FieldManager
from solr_mcp.solr.query import QueryBuilder
from solr_mcp.solr.utils.formatting import format_search_results, format_sql_response
from solr_mcp.solr.vector import VectorManager, VectorSearchResults
from solr_mcp.solr.interfaces import CollectionProvider, VectorSearchProvider
from solr_mcp.solr.zookeeper import ZooKeeperCollectionProvider

logger = logging.getLogger(__name__)

class SolrClient:
    """Client for interacting with SolrCloud."""
    
    def __init__(self, 
                 config: SolrConfig,
                 collection_provider: Optional[CollectionProvider] = None,
                 solr_client: Optional[pysolr.Solr] = None,
                 field_manager: Optional[FieldManager] = None,
                 vector_provider: Optional[VectorSearchProvider] = None,
                 query_builder: Optional[QueryBuilder] = None):
        """Initialize the SolrClient with the given configuration and optional dependencies.
        
        Args:
            config: Configuration for the client
            collection_provider: Optional collection provider implementation
            solr_client: Optional pre-configured Solr client
            field_manager: Optional pre-configured field manager
            vector_provider: Optional vector search provider implementation
            query_builder: Optional pre-configured query builder
        """
        self.config = config
        self.base_url = config.solr_base_url.rstrip('/')
        
        # Initialize collection provider
        self.collection_provider = collection_provider or ZooKeeperCollectionProvider(
            hosts=self.config.zookeeper_hosts,
            default_collection=self.config.default_collection
        )

        # Initialize or use provided Solr client
        self.solr = solr_client or self._get_or_create_client(self.config.default_collection)
        
        # Initialize or use provided managers
        self.field_manager = field_manager or FieldManager(self.base_url)
        self.vector_provider = vector_provider or VectorManager(self.solr)
        self.query_builder = query_builder or QueryBuilder(field_manager=self.field_manager)

    def _get_or_create_client(self, collection: str) -> pysolr.Solr:
        """Get or create a Solr client for the specified collection."""
        return pysolr.Solr(f"{self.base_url}/{collection}")
    
    async def list_collections(self) -> List[str]:
        """List all available collections."""
        try:
            return await self.collection_provider.list_collections()
        except ConnectionError:
            # Re-raise ConnectionError as it's the expected type
            raise
        except Exception as e:
            raise ConnectionError(f"Failed to list collections: {str(e)}")
    
    def _format_search_results(self, results: pysolr.Results, start: int = 0) -> str:
        """Format Solr search results for LLM consumption."""
        return format_search_results(results, start)

    async def execute_select_query(self, query: str) -> Dict[str, Any]:
        """Execute a SQL SELECT query against Solr using the SQL interface."""
        try:
            # Parse and validate query
            ast, collection, _ = self.query_builder.parse_and_validate_select(query)
            
            # Build SQL endpoint URL
            sql_url = f"{self.config.solr_base_url}/{collection}/sql"
            
            # Execute SQL query
            response = requests.post(sql_url, json={
                "stmt": query
            })
            
            if response.status_code != 200:
                raise QueryError(f"SQL query failed: {response.text}")
                
            return format_sql_response(response.json())
            
        except QueryError as e:
            raise QueryError(f"SQL query failed: {str(e)}")
        except Exception as e:
            raise QueryError(f"SQL query failed: {str(e)}")

    async def execute_vector_select_query(
        self,
        query: str,
        vector: List[float]
    ) -> Dict[str, Any]:
        """Execute SQL query filtered by vector similarity search."""
        try:
            # Parse and validate query
            ast, collection, _ = self.query_builder.parse_and_validate_select(query)
            
            # Get limit and offset from query
            limit = ast.args.get("limit")
            offset = ast.args.get("offset", 0)
            top_k = limit + offset if limit else None
            
            # Execute vector search
            client = self._get_or_create_client(collection)
            results = self.vector_provider.execute_vector_search(
                client=client,
                vector=vector,
                top_k=top_k
            )
            
            # Convert to VectorSearchResults
            vector_results = VectorSearchResults.from_solr_response(
                response=results,
                top_k=top_k or 10  # Use default top_k if not specified
            )
            
            # Build final query with vector results
            query_params = self.query_builder.build_vector_query(
                query,
                vector_results.get_doc_ids()
            )
            
            # Build SQL endpoint URL
            sql_url = f"{self.config.solr_base_url}/{collection}/sql"
            
            # Execute SQL query
            response = requests.post(sql_url, json=query_params)
            
            if response.status_code != 200:
                raise QueryError(f"SQL query failed: {response.text}")
                
            return format_sql_response(response.json())
            
        except Exception as e:
            if isinstance(e, (QueryError, SolrError)):
                raise
            raise QueryError(f"Error executing vector query: {str(e)}")

    async def execute_semantic_select_query(
        self,
        query: str,
        text: str
    ) -> Dict[str, Any]:
        """Execute SQL query filtered by semantic similarity."""
        try:
            # Get vector embedding
            vector = await self.vector_provider.get_embedding(text)
            
            # Reuse vector query logic
            return await self.execute_vector_select_query(query, vector)
        except Exception as e:
            if isinstance(e, (QueryError, SolrError)):
                raise
            raise SolrError(f"Semantic search failed: {str(e)}")