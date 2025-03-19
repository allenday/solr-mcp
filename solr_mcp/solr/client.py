"""SolrCloud client implementation."""

import json
import os
from typing import Any, Dict, List, Optional, Union
import urllib.parse

import numpy as np
import pysolr
from kazoo.client import KazooClient
from loguru import logger
from pydantic import BaseModel, Field

from solr_mcp.embeddings.client import OllamaClient
from solr_mcp.utils import SolrUtils, FIELD_TYPE_MAPPING, SYNTHETIC_SORT_FIELDS

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
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.solr_clients: Dict[str, pysolr.Solr] = {}
        
        # Try to initialize ZooKeeper client
        try:
            self.zk_client = self._init_zookeeper()
            logger.info(f"Connected to ZooKeeper: {','.join(self.config.zookeeper_hosts)}")
        except Exception as e:
            logger.warning(f"Failed to connect to ZooKeeper: {e}")
            logger.warning("Will continue without ZooKeeper connectivity - some features may be limited")
            # Create a dummy client to avoid failures later
            self.zk_client = None
        
        # Initialize Ollama client for embeddings
        try:
            self.ollama_client = OllamaClient()
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client: {e}")
            logger.warning("Vector operations will be unavailable")
            self.ollama_client = None
        
        # Validate default collection
        if not self.config.default_collection:
            logger.warning("No default collection specified in config")
        
        # Initialize default collection
        try:
            self._get_or_create_client(self.config.default_collection)
            logger.info(f"Connected to Solr collection: {self.config.default_collection}")
        except Exception as e:
            logger.error(f"Failed to connect to default collection '{self.config.default_collection}': {e}")
            logger.warning("Will continue but search operations may fail")
        
        # Cache searchable and sortable fields
        try:
            self.searchable_fields = self._get_searchable_fields()
            self.sortable_fields = self._get_sortable_fields()
            
            logger.info(f"SolrClient initialized with collections: {self.list_collections()}")
            logger.info(f"Searchable fields: {self.searchable_fields}")
            logger.info(f"Sortable fields: {self.sortable_fields}")
        except Exception as e:
            logger.error(f"Error initializing field information: {e}")
            # Set defaults as fallback
            self.searchable_fields = ['_text_']
            self.sortable_fields = {
                "score": {
                    "type": "numeric",
                    "directions": ["asc", "desc"],
                    "default_direction": "desc",
                    "searchable": True
                }
            }
    
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
    
    def _init_zookeeper(self) -> KazooClient:
        """Initialize ZooKeeper client.
        
        Returns:
            KazooClient instance
        """
        zk_hosts = ",".join(self.config.zookeeper_hosts)
        client = KazooClient(hosts=zk_hosts)
        client.start()
        return client
    
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
    
    def _format_search_results(self, results: pysolr.Results) -> str:
        """Format Solr search results for LLM consumption.
        
        Args:
            results: pysolr Results object
            
        Returns:
            Formatted results as JSON string
        """
        try:
            # Get the start parameter from the response header params
            response_header = getattr(results, "responseHeader", {})
            params = response_header.get("params", {})
            start = int(params.get("start", 0))

            formatted = {
                "numFound": results.hits,
                "start": start,
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
            logger.exception(f"Error formatting search results: {e}")
            return json.dumps({"error": str(e)})
    
    async def search(
        self,
        query: str,
        collection: Optional[str] = None,
        fields: Optional[List[str]] = None,
        filters: Optional[Union[str, List[str]]] = None,
        sort: Optional[str] = None,
        start: int = 0,
        rows: int = 10,
        **kwargs: Any,
    ) -> str:
        """Perform a search query against Solr.
        
        Args:
            query: The search query string
            collection: Collection to search (defaults to config default)
            fields: List of fields to return
            filters: Filter query(s) to apply
            sort: Sort string in format "field direction" or just "field"
            start: Start offset for pagination
            rows: Number of results to return
            **kwargs: Additional parameters to pass to Solr
            
        Returns:
            JSON string containing search results and metadata
        """
        client = self._get_or_create_client(collection or self.config.default_collection)
        
        # Sanitize inputs
        sanitized_fields = SolrUtils.sanitize_fields(fields)
        sanitized_filters = SolrUtils.sanitize_filters(filters)
        sanitized_sort = SolrUtils.sanitize_sort(sort, self.sortable_fields)
        
        # Build search kwargs
        search_kwargs = {
            "defType": "edismax",
            "mm": "100%",
            "tie": 0.1,
            "qf": " ".join(self.searchable_fields),
            "fl": "*,score",
            "start": max(0, int(start)),  # Ensure non-negative
            "rows": max(1, min(int(rows), 1000))  # Limit range
        }
        
        if sanitized_fields:
            search_kwargs["fl"] = ",".join(sanitized_fields)
        if sanitized_filters:
            search_kwargs["fq"] = sanitized_filters
        if sanitized_sort:
            search_kwargs["sort"] = sanitized_sort
            
        # Add any additional kwargs (consider sanitizing these too)
        search_kwargs.update(kwargs)
        
        try:
            results = client.search(query, **search_kwargs)
            return self._format_search_results(results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def get_suggestions(
        self,
        query: str,
        collection: Optional[str] = None,
        suggestion_field: str = "suggest",
        count: int = 5,
        **kwargs: Any,
    ) -> str:
        """Get search suggestions from Solr.
        
        Args:
            query: Partial query to get suggestions for
            collection: Collection name
            suggestion_field: Suggestion handler name
            count: Number of suggestions to return
            **kwargs: Additional parameters
            
        Returns:
            JSON string with suggestions
        """
        collection = collection or self.config.default_collection
        client = self._get_or_create_client(collection)
        
        handler = f"{collection}/{suggestion_field}"
        params = {
            "q": query,
            "count": count,
            "wt": "json",
            **kwargs
        }
        
        logger.debug(f"Getting suggestions for: {query}")
        
        try:
            response = client._send_request("get", handler, params)
            return json.dumps(response)
        except Exception as e:
            logger.exception(f"Error getting suggestions: {e}")
            raise
    
    async def get_facets(
        self,
        query: str,
        facet_fields: List[str],
        collection: Optional[str] = None,
        facet_limit: int = 10,
        facet_mincount: int = 1,
        **kwargs: Any,
    ) -> str:
        """Get facet information from Solr.
        
        Args:
            query: Search query
            facet_fields: Fields to facet on
            collection: Collection name
            facet_limit: Maximum number of facet values
            facet_mincount: Minimum count for facet values
            **kwargs: Additional parameters
            
        Returns:
            JSON string with facet information
        """
        collection = collection or self.config.default_collection
        client = self._get_or_create_client(collection)
        
        search_kwargs = {
            "facet": "on",
            "facet.field": facet_fields,
            "facet.limit": facet_limit,
            "facet.mincount": facet_mincount,
            "rows": 0,  # We only need facets, not results
            **kwargs
        }
        
        logger.debug(f"Getting facets for query: {query} on fields: {facet_fields}")
        
        try:
            results = client.search(query, **search_kwargs)
            return json.dumps({"facets": results.facets})
        except Exception as e:
            logger.exception(f"Error getting facets: {e}")
            raise
    
    async def vector_search(
        self,
        vector: List[float],
        vector_field: str = "embedding",
        collection: Optional[str] = None,
        k: int = 10,
        filter_query: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Perform a vector search using KNN in Solr.
        
        Args:
            vector: Dense vector embedding to search with
            vector_field: Field containing vector embeddings
            collection: Collection name
            k: Number of nearest neighbors to retrieve
            filter_query: Optional filter query
            return_fields: Fields to return
            **kwargs: Additional parameters
            
        Returns:
            JSON string with vector search results
        """
        collection = collection or self.config.default_collection
        client = self._get_or_create_client(collection)
        
        # Format vector as expected by Solr
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        
        # Build the KNN query
        knn_query = f"{{!knn f={vector_field} topK={k}}}{vector_str}"
        
        search_kwargs = {}
        
        if return_fields:
            search_kwargs["fl"] = ",".join(return_fields)
        
        if filter_query:
            search_kwargs["fq"] = filter_query
        
        # Add any additional kwargs
        search_kwargs.update(kwargs)
        
        logger.debug(f"Solr vector search with params: {search_kwargs}")
        
        try:
            results = client.search(knn_query, **search_kwargs)
            return self._format_search_results(results)
        except Exception as e:
            logger.exception(f"Error in vector search: {e}")
            raise
    
    async def index_document_with_vector(
        self,
        document: Dict[str, Any],
        vector: Optional[List[float]] = None,
        vector_field: str = "embedding",
        text_field: str = "text",
        collection: Optional[str] = None,
        commit: bool = True,
    ) -> bool:
        """Index a document with vector embedding.
        
        Args:
            document: Document fields
            vector: Vector embedding (if None, will be generated)
            vector_field: Field name for the vector
            text_field: Field name to use for generating embedding if vector is None
            collection: Collection name
            commit: Whether to commit immediately
            
        Returns:
            Success status
        """
        collection = collection or self.config.default_collection
        client = self._get_or_create_client(collection)
        
        # Add the vector to the document
        doc = document.copy()
        
        # Generate embedding if not provided
        if vector is None:
            if text_field not in doc:
                raise ValueError(f"Document must contain '{text_field}' field to generate embedding")
            
            try:
                vector = await self.ollama_client.get_embedding(doc[text_field])
                logger.info(f"Generated embedding for document with id: {doc.get('id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                raise
        
        doc[vector_field] = vector
        
        try:
            client.add([doc], commit=commit)
            logger.info(f"Document indexed with vector in {collection}")
            return True
        except Exception as e:
            logger.exception(f"Error indexing document with vector: {e}")
            raise
            
    async def index_document_with_generated_embedding(
        self,
        document: Dict[str, Any],
        text_field: str = "text",
        vector_field: str = "embedding",
        collection: Optional[str] = None,
        commit: bool = True,
    ) -> bool:
        """Index a document with automatically generated vector embedding.
        
        Args:
            document: Document fields
            text_field: Field name to use for generating embedding
            vector_field: Field name for the vector
            collection: Collection name
            commit: Whether to commit immediately
            
        Returns:
            Success status
        """
        return await self.index_document_with_vector(
            document=document,
            vector=None,
            vector_field=vector_field,
            text_field=text_field,
            collection=collection,
            commit=commit
        )
    
    async def batch_index_with_vectors(
        self,
        documents: List[Dict[str, Any]],
        vectors: Optional[List[List[float]]] = None,
        vector_field: str = "embedding",
        text_field: str = "text",
        collection: Optional[str] = None,
        commit: bool = True,
    ) -> bool:
        """Batch index documents with vector embeddings.
        
        Args:
            documents: List of documents
            vectors: List of vector embeddings (if None, will be generated)
            vector_field: Field name for vectors
            text_field: Field name to use for generating embeddings if vectors is None
            collection: Collection name
            commit: Whether to commit immediately
            
        Returns:
            Success status
        """
        collection = collection or self.config.default_collection
        client = self._get_or_create_client(collection)
        
        # Generate embeddings if not provided
        if vectors is None:
            try:
                # Extract text from documents
                texts = []
                for doc in documents:
                    if text_field not in doc:
                        raise ValueError(f"All documents must contain '{text_field}' field to generate embeddings")
                    texts.append(doc[text_field])
                
                # Generate embeddings
                vectors = await self.ollama_client.get_embeddings(texts)
                logger.info(f"Generated embeddings for {len(texts)} documents")
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise
        elif len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
        
        # Add vectors to documents
        docs_with_vectors = []
        for doc, vec in zip(documents, vectors):
            doc_copy = doc.copy()
            doc_copy[vector_field] = vec
            docs_with_vectors.append(doc_copy)
        
        try:
            client.add(docs_with_vectors, commit=commit)
            logger.info(f"Batch indexed {len(docs_with_vectors)} documents with vectors in {collection}")
            return True
        except Exception as e:
            logger.exception(f"Error batch indexing documents with vectors: {e}")
            raise
    
    async def batch_index_with_generated_embeddings(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        vector_field: str = "embedding",
        collection: Optional[str] = None,
        commit: bool = True,
    ) -> bool:
        """Batch index documents with automatically generated vector embeddings.
        
        Args:
            documents: List of documents
            text_field: Field name to use for generating embeddings
            vector_field: Field name for vectors
            collection: Collection name
            commit: Whether to commit immediately
            
        Returns:
            Success status
        """
        return await self.batch_index_with_vectors(
            documents=documents,
            vectors=None,
            vector_field=vector_field,
            text_field=text_field,
            collection=collection,
            commit=commit
        )

    def _get_sortable_fields(self) -> Dict[str, Dict[str, Any]]:
        """Get all fields that can be used for sorting.
        
        Returns:
            Dict mapping field names to their sort properties:
            {
                "field_name": {
                    "type": "string|numeric|date|boolean",
                    "directions": ["asc", "desc"],
                    "default_direction": "asc",
                    "searchable": bool,  # Whether field can be used in search queries
                    "warning": str,      # Optional warning about field usage
                }
            }
        """
        collection = self.config.default_collection
        client = self._get_or_create_client(collection)
        
        try:
            # Try getting schema using Solr admin API
            try:
                schema_url = f"admin/collections/{collection}/schema/fields"
                logger.debug(f"Getting sortable fields from schema URL: {schema_url}")
                response = client._send_request('get', schema_url)
            except Exception as e:
                logger.warning(f"Error getting schema fields for sorting: {e}")
                logger.info("Fallback: trying direct URL with query that returns field info")
                
                # Fallback - use direct select query to get the fields
                import requests
                direct_url = f"{client.url}/select?q=*:*&rows=0&wt=json"
                logger.debug(f"Trying direct URL: {direct_url}")
                response = requests.get(direct_url, timeout=self.config.connection_timeout).json()
            
            sortable_fields = {}
            
            # Process schema fields
            for field in response.get('fields', []):
                field_name = field.get('name')
                field_type = field.get('type')
                multi_valued = field.get('multiValued', False)
                doc_values = field.get('docValues', False)
                
                # Skip special fields, multi-valued fields, and fields without a recognized type
                if (field_name.startswith('_') and field_name not in SYNTHETIC_SORT_FIELDS) or \
                   multi_valued or \
                   field_type not in FIELD_TYPE_MAPPING:
                    continue
                
                # Add field to sortable fields
                sortable_fields[field_name] = {
                    "type": FIELD_TYPE_MAPPING[field_type],
                    "directions": ["asc", "desc"],
                    "default_direction": "asc" if FIELD_TYPE_MAPPING[field_type] in ["string", "numeric", "date"] else "desc",
                    "searchable": True  # Regular schema fields are searchable
                }
            
            # Add synthetic fields
            sortable_fields.update(SYNTHETIC_SORT_FIELDS)
            
            return sortable_fields
            
        except Exception as e:
            logger.error(f"Error getting sortable fields: {e}")
            # Return only the guaranteed score field - _docid_ is not recommended as a fallback
            return {
                "score": {
                    "type": "numeric",
                    "directions": ["asc", "desc"],
                    "default_direction": "desc",
                    "searchable": True
                }
            }

    def _validate_sort(self, sort: Optional[str]) -> Optional[str]:
        """Validate and normalize sort parameter.
        
        Args:
            sort: Sort string in format "field direction" or "field"
            
        Returns:
            Normalized sort string or None if invalid
            
        Raises:
            ValueError: If sort field or direction is invalid
        """
        if not sort:
            return None
        
        parts = sort.strip().split()
        field = parts[0]
        direction = parts[1].lower() if len(parts) > 1 else None
        
        # Check if field is sortable
        if field not in self.sortable_fields:
            raise ValueError(f"Field '{field}' is not sortable. Available sort fields: {list(self.sortable_fields.keys())}")
        
        field_info = self.sortable_fields[field]
        
        # Use specified direction or default
        if direction:
            if direction not in field_info["directions"]:
                raise ValueError(f"Invalid sort direction '{direction}' for field '{field}'. Allowed directions: {field_info['directions']}")
        else:
            direction = field_info["default_direction"]
        
        return f"{field} {direction}"

    def _get_searchable_fields(self) -> List[str]:
        """Get all text and string fields from the schema.
        
        Returns:
            List of field names that can be searched
        """
        collection = self.config.default_collection
        client = self._get_or_create_client(collection)
        
        try:
            # Try getting schema using Solr admin API
            try:
                schema_url = f"admin/collections/{collection}/schema/fields"
                logger.debug(f"Getting searchable fields from schema URL: {schema_url}")
                response = client._send_request('get', schema_url)
            except Exception as e:
                logger.warning(f"Error getting schema fields: {e}")
                logger.info("Fallback: trying direct URL with query that returns field info")
                
                # Fallback - use direct select query to get the fields
                import requests
                direct_url = f"{client.url}/select?q=*:*&rows=0&wt=json"
                logger.debug(f"Trying direct URL: {direct_url}")
                response = requests.get(direct_url, timeout=self.config.connection_timeout).json()
            
            # Extract text and string fields
            fields = []
            for field in response.get('fields', []):
                field_type = field.get('type', '')
                # Check if field type maps to text or string in our simplified type system
                if field_type in FIELD_TYPE_MAPPING:
                    mapped_type = FIELD_TYPE_MAPPING[field_type]
                    if mapped_type in ['text', 'string']:
                        fields.append(field['name'])
            
            if not fields:
                # If no fields found, return catch-all field
                return ['_text_']
            
            return fields
        except Exception as e:
            logger.error(f"Error getting searchable fields: {e}")
            # Return only the guaranteed catch-all field
            return ['_text_']