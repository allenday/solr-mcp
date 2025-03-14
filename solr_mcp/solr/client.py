"""SolrCloud client implementation."""

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pysolr
from kazoo.client import KazooClient
from loguru import logger
from pydantic import BaseModel, Field

from solr_mcp.embeddings.client import OllamaClient


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
        self.zk_client = self._init_zookeeper()
        self.solr_clients: Dict[str, pysolr.Solr] = {}
        
        # Initialize Ollama client for embeddings
        self.ollama_client = OllamaClient()
        
        # Initialize default collection
        self._get_or_create_client(self.config.default_collection)
        
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
        if self.zk_client.exists("/collections"):
            collections = self.zk_client.get_children("/collections")
            return collections
        return []
    
    async def search(
        self,
        query: str,
        collection: Optional[str] = None,
        fields: Optional[List[str]] = None,
        filters: Optional[List[str]] = None,
        rows: int = 10,
        start: int = 0,
        sort: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Search Solr collection.
        
        Args:
            query: Search query
            collection: Collection name (uses default if not specified)
            fields: Fields to return
            filters: Filter queries
            rows: Number of rows to return
            start: Start offset
            sort: Sort order
            **kwargs: Additional parameters
            
        Returns:
            JSON string with search results
        """
        collection = collection or self.config.default_collection
        client = self._get_or_create_client(collection)
        
        search_kwargs = {
            "rows": rows,
            "start": start,
        }
        
        if fields:
            search_kwargs["fl"] = ",".join(fields)
        
        if filters:
            search_kwargs["fq"] = filters
        
        if sort:
            search_kwargs["sort"] = sort
        
        # Add any additional kwargs
        search_kwargs.update(kwargs)
        
        # Modify the query to improve search results
        # If it's a simple query with no field specifiers, target the content field
        # and use phrase proximity search
        if query and ":" not in query:
            # Check if it's multiple words - if so, use phrase proximity
            words = query.split()
            if len(words) > 1:
                # Use content field with phrase proximity
                query = f'content:"{query}"~5'
            else:
                # Single word, just use content field
                query = f'content:{query}'
        
        logger.debug(f"Solr search: {query} with params: {search_kwargs}")
        
        try:
            results = client.search(query, **search_kwargs)
            return self._format_search_results(results)
        except Exception as e:
            logger.exception(f"Error searching Solr: {e}")
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
    
    async def hybrid_search(
        self,
        query: str,
        collection: Optional[str] = None,
        vector_field: str = "embedding",
        blend_factor: float = 0.5,
        fields: Optional[List[str]] = None,
        filter_query: Optional[str] = None,
        rows: int = 10,
        **kwargs: Any,
    ) -> str:
        """Perform a hybrid search combining keyword and vector search results.
        
        Args:
            query: Search query text
            collection: Solr collection name
            vector_field: Name of the vector field
            blend_factor: Blending factor between keyword and vector results (0-1)
            fields: Fields to return
            filter_query: Optional filter query
            rows: Number of results to return
            **kwargs: Additional parameters
            
        Returns:
            JSON string with hybrid search results
        """
        collection = collection or self.config.default_collection
        
        if not fields:
            fields = ["id", "title", "content", "source", "score"]
        
        # Run both keyword and vector searches
        keyword_results_json = await self.search(
            query=query,
            collection=collection,
            fields=fields,
            filters=[filter_query] if filter_query else None,
            rows=rows * 2  # Request more results to ensure we have enough after blending
        )
        
        # Generate embedding for vector search
        embedding = await self.ollama_client.get_embedding(query)
        
        vector_results_json = await self.vector_search(
            vector=embedding,
            vector_field=vector_field,
            collection=collection,
            k=rows * 2,  # Request more results to ensure we have enough after blending
            filter_query=filter_query,
            return_fields=fields
        )
        
        # Parse results
        keyword_results = json.loads(keyword_results_json)
        vector_results = json.loads(vector_results_json)
        
        # Extract docs from both result sets
        keyword_docs = keyword_results.get('docs', [])
        vector_docs = vector_results.get('docs', [])
        
        # Create a hybrid result set
        hybrid_docs = {}
        max_keyword_score = max([doc.get('score', 0) for doc in keyword_docs]) if keyword_docs else 1
        max_vector_score = max([doc.get('score', 0) for doc in vector_docs]) if vector_docs else 1
        
        # Process keyword results
        for doc in keyword_docs:
            doc_id = doc['id']
            # Normalize score to 0-1 range
            normalized_score = doc.get('score', 0) / max_keyword_score if max_keyword_score > 0 else 0
            hybrid_docs[doc_id] = {
                **doc,
                'keyword_score': normalized_score,
                'vector_score': 0,
                'hybrid_score': normalized_score * (1 - blend_factor)
            }
        
        # Process vector results
        for doc in vector_docs:
            doc_id = doc['id']
            # Normalize score to 0-1 range
            normalized_score = doc.get('score', 0) / max_vector_score if max_vector_score > 0 else 0
            if doc_id in hybrid_docs:
                # Update existing doc with vector score
                hybrid_docs[doc_id]['vector_score'] = normalized_score
                hybrid_docs[doc_id]['hybrid_score'] += normalized_score * blend_factor
            else:
                hybrid_docs[doc_id] = {
                    **doc,
                    'keyword_score': 0,
                    'vector_score': normalized_score,
                    'hybrid_score': normalized_score * blend_factor
                }
        
        # Sort by hybrid score
        sorted_docs = sorted(hybrid_docs.values(), key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        # Create a hybrid result
        hybrid_result = {
            "numFound": len(sorted_docs),
            "start": 0,
            "maxScore": 1.0,
            "docs": sorted_docs[:rows]
        }
        
        return json.dumps(hybrid_result, default=str)
    
    def _format_search_results(self, results: pysolr.Results) -> str:
        """Format Solr search results for LLM consumption.
        
        Args:
            results: pysolr Results object
            
        Returns:
            Formatted results as JSON string
        """
        formatted = {
            "numFound": results.hits,
            "start": getattr(results, "start", 0),
            "maxScore": getattr(results, "max_score", None),
            "docs": list(results.docs),
        }
        
        if hasattr(results, "facets") and results.facets:
            formatted["facets"] = results.facets
        
        if hasattr(results, "highlighting") and results.highlighting:
            formatted["highlighting"] = results.highlighting
        
        return json.dumps(formatted, default=str)