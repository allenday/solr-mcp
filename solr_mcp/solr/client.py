"""SolrCloud client implementation."""

import json
import logging
from typing import Any, Dict, List, Optional

import pysolr
import requests
from loguru import logger
import aiohttp

from solr_mcp.solr.config import SolrConfig
from solr_mcp.solr.exceptions import (
    SolrError, ConnectionError, QueryError,
    DocValuesError, SQLParseError, SQLExecutionError
)
from solr_mcp.solr.schema import FieldManager
from solr_mcp.solr.query import QueryBuilder
from solr_mcp.solr.utils.formatting import format_search_results, format_sql_response
from solr_mcp.solr.vector import VectorManager, VectorSearchResults
from solr_mcp.solr.interfaces import CollectionProvider, VectorSearchProvider
from solr_mcp.solr.zookeeper import ZooKeeperCollectionProvider
from solr_mcp.vector_provider import OllamaVectorProvider

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
            hosts=self.config.zookeeper_hosts
        )

        # Initialize field manager
        self.field_manager = field_manager or FieldManager(self.base_url)

        # Initialize vector provider
        self.vector_provider = vector_provider or OllamaVectorProvider()

        # Initialize query builder
        self.query_builder = query_builder or QueryBuilder(field_manager=self.field_manager)

        # Initialize vector manager
        self.vector_manager = VectorManager(
            self,
            self.vector_provider,
            self.config.embedding_field,
            self.config.default_top_k
        )

        # Initialize Solr client
        self._solr_client = solr_client
        self._default_collection = self.config.default_collection

    async def _get_or_create_client(self, collection: Optional[str] = None) -> pysolr.Solr:
        """Get or create a Solr client for the given collection.
        
        Args:
            collection: Optional collection name to use. If not provided, uses default collection.
            
        Returns:
            Configured Solr client
            
        Raises:
            SolrError: If no collection is specified and no default collection is configured
        """
        if not collection:
            if not self._default_collection:
                raise SolrError("No collection specified and no default collection configured")
            collection = self._default_collection

        if not self._solr_client:
            self._solr_client = pysolr.Solr(
                f"{self.base_url}/{collection}",
                timeout=self.config.connection_timeout
            )

        return self._solr_client
    
    async def list_collections(self) -> List[str]:
        """List all available collections."""
        try:
            response = requests.get(f"{self.base_url}/admin/collections?action=LIST")
            if response.status_code != 200:
                raise SolrError(f"Failed to list collections: {response.text}")
            
            collections = response.json()['collections']
            return collections
            
        except Exception as e:
            raise SolrError(f"Failed to list collections: {str(e)}")

    async def list_fields(self, collection: str) -> List[Dict[str, Any]]:
        """List all fields in a collection with their properties."""
        try:
            # Verify collection exists
            collections = await self.list_collections()
            if collection not in collections:
                raise SolrError(f"Collection '{collection}' does not exist")
            
            # Get schema fields and copyFields
            schema_response = requests.get(f"{self.base_url}/{collection}/schema")
            if schema_response.status_code != 200:
                raise SolrError(f"Failed to get schema for collection '{collection}': {schema_response.text}")
            
            schema = schema_response.json()
            fields = schema['schema']['fields']
            copy_fields = schema['schema'].get('copyFields', [])
            
            # Build map of destination fields to their source fields
            copies_from = {}
            for copy_field in copy_fields:
                dest = copy_field['dest']
                source = copy_field['source']
                if dest not in copies_from:
                    copies_from[dest] = []
                copies_from[dest].append(source)
            
            # Add copyField information to field properties
            for field in fields:
                if field['name'] in copies_from:
                    field['copies_from'] = copies_from[field['name']]
            
            return fields
            
        except SolrError:
            raise
        except Exception as e:
            raise SolrError(f"Failed to list fields for collection '{collection}': {str(e)}")

    def _format_search_results(self, results: pysolr.Results, start: int = 0) -> str:
        """Format Solr search results for LLM consumption."""
        return format_search_results(results, start)

    async def execute_select_query(self, query: str) -> Dict[str, Any]:
        """Execute a SQL SELECT query against Solr using the SQL interface."""
        try:
            # Parse and validate query
            logger.debug(f"Original query: {query}")
            preprocessed_query = self.query_builder.parser.preprocess_query(query)
            logger.debug(f"Preprocessed query: {preprocessed_query}")
            ast, collection, _ = self.query_builder.parse_and_validate_select(preprocessed_query)
            logger.debug(f"Parsed collection: {collection}")
            
            # Build SQL endpoint URL with aggregationMode
            sql_url = f"{self.base_url}/{collection}/sql?aggregationMode=facet"
            logger.debug(f"SQL URL: {sql_url}")
            
            # Execute SQL query with URL-encoded form data
            payload = {'stmt': preprocessed_query.strip()}
            logger.debug(f"Request payload: {payload}")
            
            response = requests.post(
                sql_url,
                data=payload,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response text: {response.text}")
            
            if response.status_code != 200:
                raise SQLExecutionError(f"SQL query failed with status {response.status_code}: {response.text}")
            
            response_json = response.json()
            
            # Check for Solr SQL exception in response
            if 'result-set' in response_json and 'docs' in response_json['result-set']:
                docs = response_json['result-set']['docs']
                if docs and 'EXCEPTION' in docs[0]:
                    exception_msg = docs[0]['EXCEPTION']
                    response_time = docs[0].get('RESPONSE_TIME')
                    
                    # Raise appropriate exception type based on error message
                    if 'must have DocValues to use this feature' in exception_msg:
                        raise DocValuesError(exception_msg, response_time)
                    elif 'parse failed:' in exception_msg:
                        raise SQLParseError(exception_msg, response_time)
                    else:
                        raise SQLExecutionError(exception_msg, response_time)
            
            return format_sql_response(response_json)
            
        except (DocValuesError, SQLParseError, SQLExecutionError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise SQLExecutionError(f"SQL query failed: {str(e)}")

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
            limit = 10  # Default limit
            if ast.args.get('limit'):
                try:
                    limit_expr = ast.args['limit']
                    if hasattr(limit_expr, 'expression'):
                        # Handle case where expression is a Literal
                        if hasattr(limit_expr.expression, 'this'):
                            limit = int(limit_expr.expression.this)
                        else:
                            limit = int(limit_expr.expression)
                    else:
                        limit = int(limit_expr)
                except (ValueError, AttributeError):
                    limit = 10  # Fallback to default
            
            offset = ast.args.get('offset', 0)
            
            # For KNN search, we need to fetch limit + offset results to account for pagination
            top_k = limit + offset
            
            # Execute vector search
            client = await self._get_or_create_client(collection)
            results = await self.vector_manager.execute_vector_search(
                client=client,
                vector=vector,
                top_k=top_k
            )
            
            # Convert to VectorSearchResults
            vector_results = VectorSearchResults.from_solr_response(
                response=results,
                top_k=top_k
            )
            
            # Build final query with vector results
            query_params = self.query_builder.build_vector_query(
                query,
                vector_results.get_doc_ids()
            )
            
            # Build SQL endpoint URL
            sql_url = f"{self.config.solr_base_url}/{collection}/sql?aggregationMode=facet"
            
            # Execute SQL query using aiohttp
            async with aiohttp.ClientSession() as session:
                # Convert query_params to SQL statement
                stmt = query  # Start with original query
                if 'fq' in query_params:
                    # Add filter query if present
                    doc_ids = query_params['fq']
                    if doc_ids:
                        stmt = f"{stmt} WHERE id IN ({','.join(doc_ids)})"
                    else:
                        # No vector search results, return empty result set
                        stmt = f"{stmt} WHERE 1=0"  # Always false condition
                
                if 'rows' in query_params:
                    # Add limit if present and not already in query
                    if 'LIMIT' not in stmt.upper():
                        stmt = f"{stmt} LIMIT {query_params['rows']}"
                
                logger.debug(f"Executing SQL query: {stmt}")
                async with session.post(
                    sql_url,
                    data={'stmt': stmt},
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise QueryError(f"SQL query failed: {error_text}")
                    
                    content_type = response.headers.get('Content-Type', '')
                    response_text = await response.text()
                    
                    try:
                        if 'application/json' in content_type:
                            response_json = json.loads(response_text)
                        else:
                            # For text/plain responses, try to parse as JSON first
                            try:
                                response_json = json.loads(response_text)
                            except json.JSONDecodeError:
                                # If not JSON, wrap in a basic result structure
                                response_json = {
                                    'result-set': {
                                        'docs': [],
                                        'numFound': 0,
                                        'start': 0
                                    }
                                }
                        
                        return format_sql_response(response_json)
                    except Exception as e:
                        raise QueryError(f"Failed to parse response: {str(e)}, Response: {response_text[:200]}")
            
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
            vector = await self.vector_manager.get_embedding(text)
            
            # Reuse vector query logic
            return await self.execute_vector_select_query(query, vector)
        except Exception as e:
            if isinstance(e, (QueryError, SolrError)):
                raise
            raise SolrError(f"Semantic search failed: {str(e)}")
