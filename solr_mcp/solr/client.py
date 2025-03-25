"""SolrCloud client implementation."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

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
from solr_mcp.vector_provider.constants import MODEL_DIMENSIONS

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

        # Initialize vector manager with default top_k of 10
        self.vector_manager = VectorManager(
            self,
            self.vector_provider,
            10  # Default value for top_k
        )

        # Initialize Solr client
        self._solr_client = solr_client
        self._default_collection = None

    async def _get_or_create_client(self, collection: str) -> pysolr.Solr:
        """Get or create a Solr client for the given collection.
        
        Args:
            collection: Collection name to use.
            
        Returns:
            Configured Solr client
            
        Raises:
            SolrError: If no collection is specified
        """
        if not collection:
            raise SolrError("No collection specified")

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
        vector: List[float],
        field: str
    ) -> Dict[str, Any]:
        """Execute SQL query filtered by vector similarity search."""
        try:
            # Parse and validate query
            ast, collection, _ = self.query_builder.parse_and_validate_select(query)
            
            # Validate field exists and is a dense_vector type
            fields = await self.list_fields(collection)
            field_info = next((f for f in fields if f['name'] == field), None)
            if not field_info:
                raise SolrError(f"Field '{field}' does not exist in collection '{collection}'")
            if field_info.get('type') != 'dense_vector':
                raise SolrError(f"Field '{field}' is not a dense_vector field")
            
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
                field=field,
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
        text: str,
        field: str,
        vector_provider_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute SQL query filtered by semantic similarity.
        
        Args:
            query: SQL query to execute
            text: Search text to convert to vector
            field: Name of the DenseVector field to search against
            vector_provider_config: Optional configuration for the vector provider
                                    Can include 'model', 'base_url', etc.
            
        Returns:
            Query results
            
        Raises:
            SolrError: If search fails
            QueryError: If query execution fails
        """
        try:
            # Parse and validate query to get collection name
            ast, collection, _ = self.query_builder.parse_and_validate_select(query)
            
            # Extract model from config if present
            model = vector_provider_config.get("model") if vector_provider_config else None
            
            # Validate field exists and get its dimension
            field_info = await self._validate_vector_field_dimension(collection, field, model)
            
            # Get vector using the vector provider configuration
            vector = await self.vector_manager.get_vector(text, vector_provider_config)
            
            # Reuse vector query logic
            return await self.execute_vector_select_query(query, vector, field)
        except Exception as e:
            if isinstance(e, (QueryError, SolrError)):
                raise
            raise SolrError(f"Semantic search failed: {str(e)}")
            
    async def _validate_vector_field_dimension(self, collection: str, field: str, vector_provider_model: Optional[str] = None) -> Dict[str, Any]:
        """Validate that the vector field exists and its dimension matches the vectorizer.
        
        Args:
            collection: Collection name
            field: Field name to validate
            vectorizer: Optional vectorizer model name
            
        Returns:
            Field information dictionary
            
        Raises:
            SolrError: If validation fails
        """
        # Initialize cache if needed
        if not hasattr(self.field_manager, '_vector_field_cache'):
            self.field_manager._vector_field_cache = {}
            
        # Check cache first
        cache_key = f"{collection}:{field}"
        if cache_key in self.field_manager._vector_field_cache:
            field_info = self.field_manager._vector_field_cache[cache_key]
            logger.debug(f"Using cached field info for {cache_key}")
            return field_info
            
        try:
            # Get collection fields
            fields = await self.list_fields(collection)
            
            # Find the specified field
            field_info = next((f for f in fields if f['name'] == field), None)
            if not field_info:
                raise SolrError(f"Field '{field}' does not exist in collection '{collection}'")
                
            # Check if field is a dense_vector type
            field_type = field_info.get('type')
            if field_type != 'dense_vector':
                raise SolrError(f"Field '{field}' is not a dense_vector field (type: {field_type})")
                
            # Get field dimension
            vector_dimension = None
            if 'vectorDimension' in field_info:
                vector_dimension = field_info['vectorDimension']
            else:
                # Look for the field type in schema
                field_types = [f for f in fields if f.get('class') == 'solr.DenseVectorField' and f.get('name') == field_type]
                if field_types and 'vectorDimension' in field_types[0]:
                    vector_dimension = field_types[0]['vectorDimension']
            
            if not vector_dimension:
                raise SolrError(f"Could not determine vector dimension for field '{field}'")
                
            # Get vector provider model dimension
            model_name = vector_provider_model or self.vector_manager.client.model
            model_dimension = MODEL_DIMENSIONS.get(model_name)
            if not model_dimension:
                raise SolrError(f"Unknown vector dimension for model '{model_name}'")
                
            # Validate dimensions match
            if int(vector_dimension) != model_dimension:
                raise SolrError(
                    f"Vector dimension mismatch: field '{field}' has dimension {vector_dimension}, "
                    f"but model '{model_name}' produces vectors with dimension {model_dimension}"
                )
                
            # Cache the result
            self.field_manager._vector_field_cache[cache_key] = field_info
            return field_info
            
        except SolrError:
            raise
        except Exception as e:
            raise SolrError(f"Error validating vector field dimension: {str(e)}")
