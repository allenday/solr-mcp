"""SolrCloud client implementation."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import urllib.parse
import re

import numpy as np
import pysolr
import requests
from kazoo.client import KazooClient
from loguru import logger
from pydantic import BaseModel, Field
import sqlglot
from sqlglot import parse_one, exp
from sqlglot.expressions import Select, From, Column, Identifier, Where, Binary

from solr_mcp.utils import SolrUtils
from solr_mcp.embeddings.client import OllamaClient

logger = logging.getLogger(__name__)

class SolrError(Exception):
    """Custom exception for Solr-related errors."""
    pass

# Field type mapping for sorting
FIELD_TYPE_MAPPING = {
    "string": "string",
    "text_general": "text",
    "text_en": "text",
    "int": "numeric",
    "long": "numeric",
    "float": "numeric",
    "double": "numeric",
    "date": "date",
    "boolean": "boolean"
}

# Synthetic fields that can be used for sorting
SYNTHETIC_SORT_FIELDS = {
    "score": {
        "type": "numeric",
        "directions": ["asc", "desc"],
        "default_direction": "desc",
        "searchable": True
    },
    "_docid_": {
        "type": "numeric",
        "directions": ["asc", "desc"],
        "default_direction": "asc",
        "searchable": False,
        "warning": "Internal Lucene document ID. Not stable across restarts or reindexing."
    }
}

class SolrConfig(BaseModel):
    """Solr configuration model."""
    
    zookeeper_hosts: List[str] = Field(
        default_factory=lambda: ["localhost:2181"],
        description="ZooKeeper ensemble hosts"
    )
    solr_base_url: str = Field(
        default="http://localhost:8983/solr",
        description="Solr base URL"
    )
    default_collection: str = Field(
        default="unified",
        description="Default Solr collection"
    )
    connection_timeout: int = Field(
        default=10,
        description="Connection timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )


class SolrClient:
    """Client for interacting with SolrCloud."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the SolrClient.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize client cache
        self.solr_clients: Dict[str, pysolr.Solr] = {}
        
        # Initialize ZooKeeper client
        try:
            self.zk_client = KazooClient(
                hosts=",".join(self.config.zookeeper_hosts),
                timeout=self.config.connection_timeout
            )
            self.zk_client.start()
            logger.info(f"Connected to ZooKeeper: {self.config.zookeeper_hosts}")
        except Exception as e:
            logger.error(f"Error connecting to ZooKeeper: {e}")
            raise
        
        # Initialize Ollama client if available
        try:
            self.ollama = OllamaClient()
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client: {e}")
            logger.warning("Vector operations will be unavailable")
            self.ollama = None
        
        # Initialize Solr client for default collection
        try:
            self._get_or_create_client(self.config.default_collection)
            logger.info(f"Connected to Solr collection: {self.config.default_collection}")
        except Exception as e:
            logger.error(f"Error connecting to Solr: {e}")
            raise
            
        # Initialize cache for field information per collection
        self._field_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"SolrClient initialized with collections: {self.list_collections()}")

    def _load_config(self, config_path: Optional[str] = None) -> SolrConfig:
        """Load configuration from file or environment.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            SolrConfig object
        """
        config_data = {}
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
        
        # Override with environment variables
        env_zk = os.environ.get("SOLR_MCP_ZK_HOSTS")
        env_solr = os.environ.get("SOLR_MCP_SOLR_URL")
        env_collection = os.environ.get("SOLR_MCP_DEFAULT_COLLECTION")
        
        if env_zk:
            config_data["zookeeper_hosts"] = env_zk.split(",")
        if env_solr:
            config_data["solr_base_url"] = env_solr
        if env_collection:
            config_data["default_collection"] = env_collection
        
        return SolrConfig(**config_data)
    
    def _get_or_create_client(self, collection: str) -> pysolr.Solr:
        """Get or create a Solr client for the specified collection.
        
        Args:
            collection: Solr collection name
            
        Returns:
            pysolr.Solr client
        """
        if collection not in self.solr_clients:
            solr_url = f"{self.config.solr_base_url}/{collection}"
            self.solr_clients[collection] = pysolr.Solr(
                solr_url,
                timeout=self.config.connection_timeout,
                always_commit=False
            )
        
        return self.solr_clients[collection]
    
    def list_collections(self) -> List[str]:
        """List available Solr collections.
        
        Returns:
            List of collection names
        """
        if self.zk_client and self.zk_client.exists("/collections"):
            collections = self.zk_client.get_children("/collections")
            return collections
        return []
    
    def _format_search_results(self, results: pysolr.Results, start: int = 0) -> str:
        """Format Solr search results for LLM consumption.
        
        Args:
            results: pysolr Results object
            start: Start offset used in the search
            
        Returns:
            Formatted results as JSON string
        """
        try:
            formatted = {
                "numFound": results.hits,
                "start": start,  # Always include the start parameter
                "maxScore": getattr(results, "max_score", None),
                "docs": list(results.docs) if hasattr(results, "docs") else [],
            }
            
            if hasattr(results, "facets") and results.facets:
                formatted["facets"] = results.facets
            
            if hasattr(results, "highlighting") and results.highlighting:
                formatted["highlighting"] = results.highlighting
            
            try:
                return json.dumps(formatted, default=str)
            except TypeError as e:
                logger.error(f"JSON serialization error: {e}")
                # Fall back to basic result format
                return json.dumps({
                    "numFound": results.hits,
                    "start": start,
                    "docs": [str(doc) for doc in results.docs] if hasattr(results, "docs") else []
                })
        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            return json.dumps({"error": str(e)})
    
    def _get_collection_fields(self, collection: str) -> Dict[str, Any]:
        """Get or load field information for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Dict containing searchable and sortable fields for the collection
        """
        if collection not in self._field_cache:
            try:
                searchable_fields = self._get_searchable_fields(collection)
                sortable_fields = self._get_sortable_fields(collection)
                
                self._field_cache[collection] = {
                    "searchable_fields": searchable_fields,
                    "sortable_fields": sortable_fields,
                    "last_updated": time.time()
                }
                
                logger.info(f"Loaded field information for collection {collection}")
                logger.debug(f"Searchable fields: {searchable_fields}")
                logger.debug(f"Sortable fields: {sortable_fields}")
            except Exception as e:
                logger.error(f"Error loading field information for collection {collection}: {e}")
                # Use safe defaults
                self._field_cache[collection] = {
                    "searchable_fields": ['_text_'],
                    "sortable_fields": {
                        "score": {
                            "type": "numeric",
                            "directions": ["asc", "desc"],
                            "default_direction": "desc",
                            "searchable": True
                        }
                    },
                    "last_updated": time.time()
                }
        
        return self._field_cache[collection]

    def _get_searchable_fields(self, collection: str) -> List[str]:
        """Get list of searchable fields for a collection.

        Args:
            collection: Collection name

        Returns:
            List of field names that can be searched
        """
        try:
            # Try schema API first
            schema_url = f"{collection}/schema/fields?wt=json"
            self.logger.debug(f"Getting searchable fields from schema URL: {schema_url}")
            full_url = f"{self.config.solr_base_url}/{schema_url}"
            self.logger.debug(f"Full URL: {full_url}")
            
            response = requests.get(full_url)
            fields_data = response.json()
            
            searchable_fields = []
            for field in fields_data.get("fields", []):
                field_name = field.get("name")
                field_type = field.get("type")
                
                # Skip special fields
                if field_name.startswith("_") and field_name not in ["_text_"]:
                    continue
                    
                # Add text and string fields
                if field_type in ["text_general", "string"] or "text" in field_type:
                    self.logger.debug(f"Found searchable field: {field_name}, type: {field_type}")
                    searchable_fields.append(field_name)
            
            # Add known content fields
            content_fields = ["content", "title", "_text_"]
            for field in content_fields:
                if field not in searchable_fields:
                    searchable_fields.append(field)
                    
            self.logger.info(f"Using searchable fields for collection {collection}: {searchable_fields}")
            return searchable_fields
            
        except Exception as e:
            self.logger.warning(f"Error getting schema fields: {str(e)}")
            self.logger.info("Fallback: trying direct URL with query that returns field info")
            
            try:
                client = self._get_or_create_client(collection)
                direct_url = f"{client.url}/select?q=*:*&rows=0&wt=json"
                self.logger.debug(f"Trying direct URL: {direct_url}")
                
                response = requests.get(direct_url)
                response_data = response.json()
                
                # Extract fields from response header
                fields = []
                if "responseHeader" in response_data:
                    header = response_data["responseHeader"]
                    if "params" in header and "fl" in header["params"]:
                        fields = header["params"]["fl"].split(",")
                
                # Add known searchable fields
                fields.extend(["content", "title", "_text_"])
                searchable_fields = list(set(fields))  # Remove duplicates
                
            except Exception as e2:
                self.logger.error(f"Error getting searchable fields: {str(e2)}")
                self.logger.info("Using fallback searchable fields: ['content', 'title', '_text_']")
                searchable_fields = ["content", "title", "_text_"]
                
            self.logger.info(f"Using searchable fields for collection {collection}: {searchable_fields}")
            return searchable_fields

    def _get_sortable_fields(self, collection: str) -> Dict[str, Dict[str, Any]]:
        """Get list of sortable fields and their properties for a collection.

        Args:
            collection: Collection name

        Returns:
            Dict mapping field names to their properties
        """
        try:
            # Try schema API first
            schema_url = f"{collection}/schema/fields?wt=json"
            self.logger.debug(f"Getting sortable fields from schema URL: {schema_url}")
            full_url = f"{self.config.solr_base_url}/{schema_url}"
            self.logger.debug(f"Full URL: {full_url}")
            
            response = requests.get(full_url)
            fields_data = response.json()
            
            sortable_fields = {}
            for field in fields_data.get("fields", []):
                field_name = field.get("name")
                field_type = field.get("type", "")
                
                # Skip special fields except _docid_
                if field_name.startswith("_") and field_name != "_docid_":
                    continue
                
                # Determine field type category
                field_category = "text"
                if "date" in field_type or field_type == "pdate" or field_name.endswith("_dt"):
                    field_category = "date"
                elif "int" in field_type or "long" in field_type or "float" in field_type or "double" in field_type or field_type.startswith("p"):
                    field_category = "numeric"
                elif field_type == "string":
                    field_category = "string"
                
                # Add field to sortable fields
                self.logger.debug(f"Found sortable field: {field_name}, type: {field_type}")
                sortable_fields[field_name] = {
                    "type": field_category,
                    "directions": ["asc", "desc"],
                    "default_direction": "asc" if field_category in ["string", "date"] else "desc",
                    "searchable": True
                }
            
            # Add special fields
            sortable_fields["score"] = {
                "type": "numeric",
                "directions": ["asc", "desc"],
                "default_direction": "desc",
                "searchable": True
            }
            
            if "_docid_" not in sortable_fields:
                sortable_fields["_docid_"] = {
                    "type": "numeric",
                    "directions": ["asc", "desc"],
                    "default_direction": "asc",
                    "searchable": False,
                    "warning": "Internal Lucene document ID. Not stable across restarts or reindexing."
                }
            
            # Add known date fields if not already present
            known_date_fields = ["date_indexed_dt", "created_dt", "modified_dt"]
            for field in known_date_fields:
                if field not in sortable_fields:
                    sortable_fields[field] = {
                        "type": "date",
                        "directions": ["asc", "desc"],
                        "default_direction": "asc",
                        "searchable": True
                    }
            
            self.logger.info(f"Using detected and known sortable fields for collection {collection}: {list(sortable_fields.keys())}")
            return sortable_fields
            
        except Exception as e:
            self.logger.warning(f"Error getting schema fields for sorting: {str(e)}")
            self.logger.info("Fallback: trying direct URL with query that returns field info")
            
            try:
                client = self._get_or_create_client(collection)
                direct_url = f"{client.url}/select?q=*:*&rows=0&wt=json"
                self.logger.debug(f"Trying direct URL: {direct_url}")
                
                response = requests.get(direct_url)
                response_data = response.json()
                
                # Extract fields from response header
                fields = []
                if "responseHeader" in response_data:
                    header = response_data["responseHeader"]
                    if "params" in header and "fl" in header["params"]:
                        fields = header["params"]["fl"].split(",")
                
                # Create sortable fields dict with basic properties
                sortable_fields = {}
                for field in fields:
                    if field.startswith("_") and field != "_docid_":
                        continue
                    sortable_fields[field] = {
                        "type": "string",  # Default to string type
                        "directions": ["asc", "desc"],
                        "default_direction": "asc",
                        "searchable": True
                    }
                
                # Add special fields
                sortable_fields["score"] = {
                    "type": "numeric",
                    "directions": ["asc", "desc"],
                    "default_direction": "desc",
                    "searchable": True
                }
                
                if "_docid_" not in sortable_fields:
                    sortable_fields["_docid_"] = {
                        "type": "numeric",
                        "directions": ["asc", "desc"],
                        "default_direction": "asc",
                        "searchable": False,
                        "warning": "Internal Lucene document ID. Not stable across restarts or reindexing."
                    }
                
            except Exception as e2:
                self.logger.error(f"Error getting sortable fields: {str(e2)}")
                # Return only score as sortable field
                sortable_fields = {
                    "score": {
                        "type": "numeric",
                        "directions": ["asc", "desc"],
                        "default_direction": "desc",
                        "searchable": True
                    }
                }
            
            self.logger.info(f"Using detected and known sortable fields for collection {collection}: {list(sortable_fields.keys())}")
            return sortable_fields

    def _validate_sort(self, sort: Optional[str]) -> Optional[str]:
        """Validate and normalize sort parameter.
        
        Args:
            sort: Sort string in format "field direction" or just "field"
            
        Returns:
            Validated sort string or None if sort is None
            
        Raises:
            ValueError: If sort field is invalid or direction is invalid
        """
        if not sort:
            return None
            
        parts = sort.strip().split()
        if len(parts) == 1:
            field = parts[0]
            direction = None
        elif len(parts) == 2:
            field, direction = parts
        else:
            raise ValueError(f"Invalid sort format: {sort}")
            
        # Get sortable fields for the collection
        sortable_fields = self._get_collection_fields(self.config.default_collection)["sortable_fields"]
        
        # Check if field is sortable
        if field not in sortable_fields:
            raise ValueError(f"Field '{field}' is not sortable")
            
        # Validate direction if provided
        if direction:
            valid_directions = sortable_fields[field]["directions"]
            if direction.lower() not in [d.lower() for d in valid_directions]:
                raise ValueError(f"Invalid sort direction '{direction}' for field '{field}'")
        else:
            # Use default direction for field
            direction = sortable_fields[field]["default_direction"]
            
        return f"{field} {direction}"

    def _validate_fields(self, collection: str, fields: List[str]) -> None:
        """Validate that the requested fields exist in the collection.
        
        Args:
            collection: Collection name
            fields: List of field names to validate
            
        Raises:
            SolrError: If any field is not valid for the collection
        """
        collection_info = self._get_collection_fields(collection)
        searchable_fields = collection_info["searchable_fields"]
        sortable_fields = collection_info["sortable_fields"]
        
        # Combine all valid fields
        valid_fields = set(searchable_fields) | set(sortable_fields.keys())
        
        # Check each requested field
        invalid_fields = [f for f in fields if f not in valid_fields]
        if invalid_fields:
            raise SolrError(f"Invalid fields for collection {collection}: {', '.join(invalid_fields)}")

    def _validate_sort_fields(self, collection: str, sort_fields: List[str]) -> None:
        """Validate that the requested sort fields are sortable in the collection.
        
        Args:
            collection: Collection name
            sort_fields: List of field names to validate for sorting
            
        Raises:
            SolrError: If any field is not sortable in the collection
        """
        collection_info = self._get_collection_fields(collection)
        sortable_fields = collection_info["sortable_fields"]
        
        # Check each sort field
        invalid_fields = [f for f in sort_fields if f not in sortable_fields]
        if invalid_fields:
            raise SolrError(f"Fields not sortable in collection {collection}: {', '.join(invalid_fields)}")

    def _extract_sort_fields(self, sort_spec: str) -> List[str]:
        """Extract field names from a sort specification.
        
        Args:
            sort_spec: Sort specification string (e.g. "field1 asc, field2 desc")
            
        Returns:
            List of field names used in sorting
        """
        fields = []
        for part in sort_spec.split(","):
            field = part.strip().split()[0]  # Get field name before direction
            fields.append(field)
        return fields

    def _preprocess_solr_query(self, query: str) -> str:
        """Preprocess a Solr query to make it SQL-compatible.
        
        Args:
            query: SQL query with potential Solr syntax
            
        Returns:
            SQL-compatible query string
        """
        # Replace Solr field:value syntax with field = 'value'
        def replace_field_value(match):
            field = match.group(1)
            value = match.group(2)
            return f"{field} = '{value}'"
            
        # First pass: handle basic field:value syntax
        query = re.sub(r'(\w+):(\w+)', replace_field_value, query)
        
        return query

    async def execute_select_query(self, query: str) -> Dict[str, Any]:
        """Execute a SQL SELECT query against Solr using the SQL interface.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Search results as a dictionary with result-set containing docs and metadata
            
        Raises:
            SolrError: If there is a syntax error or other Solr-related error
        """
        try:
            # Parse and validate SQL query
            ast = parse_one(self._preprocess_solr_query(query))
            if not isinstance(ast, Select):
                raise SolrError("Query must be a SELECT statement")
            
            # Extract collection from FROM clause
            from_clause = ast.args.get("from")
            if not from_clause or not isinstance(from_clause, From):
                raise SolrError("Query must have a FROM clause")
            
            collection = from_clause.this.name
            
            # Extract selected fields
            selected_fields = []
            for expr in ast.expressions:
                if isinstance(expr, Column):
                    selected_fields.append(expr.alias_or_name)
                elif isinstance(expr, Identifier):
                    selected_fields.append(expr.name)
            
            # Validate fields if any are specified
            if selected_fields and not all(field == "*" for field in selected_fields):
                self._validate_fields(collection, selected_fields)
            
            # Extract and validate sort fields if ORDER BY is present
            sort_clause = ast.args.get("order")
            if sort_clause:
                sort_fields = []
                for expr in sort_clause:
                    if isinstance(expr, exp.Ordered):
                        if isinstance(expr.this, exp.Column):
                            sort_fields.append(expr.this.name)
                        elif isinstance(expr.this, exp.Identifier):
                            sort_fields.append(expr.this.name)
                self._validate_sort_fields(collection, sort_fields)
            
            # Build SQL endpoint URL
            sql_url = f"{self.config.solr_base_url}/{collection}/sql"
            
            # Execute SQL query
            response = requests.post(sql_url, json={
                "stmt": query
            })
            
            if response.status_code != 200:
                raise SolrError(f"SQL query failed: {response.text}")
            
            # Format response to match expected structure
            raw_response = response.json()
            
            # Transform response to expected format
            result = {
                "rows": raw_response.get("docs", []),
                "numFound": raw_response.get("numFound", 0),
                "offset": raw_response.get("start", 0)
            }
            
            return result
            
        except Exception as e:
            if isinstance(e, SolrError):
                raise
            raise SolrError(f"Error executing SQL query: {str(e)}")

    async def execute_semantic_select_query(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Execute a semantic search by converting text query to vector embedding.
        
        Args:
            query: Text query to convert to vector embedding
            limit: Maximum number of results to return
            
        Returns:
            Search results as a dictionary with rows, numFound, and offset
            
        Raises:
            SolrError: If Ollama client is not initialized or other Solr-related error
        """
        if not self.ollama:
            raise SolrError("Ollama client not initialized. Vector operations unavailable.")
            
        try:
            # Get vector embedding for query
            embedding = await self.ollama.get_embedding(query)
            
            # Execute vector search with the embedding
            return await self.execute_vector_select_query(embedding)
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise SolrError(f"Semantic search failed: {e}")

    async def execute_vector_select_query(self, vector: List[float]) -> Dict[str, Any]:
        """Execute a vector similarity search using a raw vector.
        
        Args:
            vector: Vector to search for similar documents
            
        Returns:
            Search results as a dictionary with rows, numFound, and offset
            
        Raises:
            SolrError: If there is a Solr-related error
        """
        try:
            # Format vector for KNN query
            vector_str = "[" + ",".join(map(str, vector)) + "]"
            knn_query = f"{{!knn f=embedding topK=10}}{vector_str}"
            
            # Execute search
            client = self._get_or_create_client(self.config.default_collection)
            results = client.search(knn_query)
            
            return {
                "rows": list(results.docs),
                "numFound": results.hits,
                "offset": 0
            }
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise SolrError(f"Vector search failed: {e}")


        """DEPRECATED: Use execute_semantic_select instead.
        
        This method is kept for backward compatibility and will be removed in a future version.
        """
        import warnings
        warnings.warn(
            "vector_search is deprecated. Use execute_semantic_select instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return await self.execute_semantic_select(query, limit)