"""Tests for the SolrClient."""

import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from kazoo.client import KazooClient
from pysolr import Solr, Results

from solr_mcp.solr.client import SolrClient, SolrConfig


class TestSolrClient:
    """Test suite for SolrClient."""
    
    @pytest.fixture
    def mock_kazoo(self):
        """Mock KazooClient."""
        with patch("solr_mcp.solr.client.KazooClient") as mock:
            kazoo_instance = MagicMock(spec=KazooClient)
            mock.return_value = kazoo_instance
            kazoo_instance.exists.return_value = True
            kazoo_instance.get_children.return_value = ["collection1", "collection2"]
            yield kazoo_instance
    
    @pytest.fixture
    def mock_solr(self):
        """Mock pysolr.Solr."""
        with patch("solr_mcp.solr.client.pysolr.Solr") as mock:
            solr_instance = MagicMock(spec=Solr)
            mock.return_value = solr_instance
            
            # Mock search results
            results = MagicMock(spec=Results)
            results.hits = 10
            results.docs = [
                {"id": "1", "title": "Document 1", "content": "Content 1"},
                {"id": "2", "title": "Document 2", "content": "Content 2"},
            ]
            results.facets = {"facet_fields": {"category": ["books", 5, "articles", 3]}}
            
            solr_instance.search.return_value = results
            
            # Mock _send_request for suggestions
            suggestion_response = {
                "suggest": {
                    "suggest": {
                        "doc": {
                            "numFound": 2,
                            "suggestions": [
                                {"term": "document", "weight": 10},
                                {"term": "documentation", "weight": 5},
                            ]
                        }
                    }
                }
            }
            solr_instance._send_request.return_value = suggestion_response
            
            yield solr_instance
    
    @pytest.fixture
    def client(self, mock_kazoo, mock_solr):
        """Create a SolrClient instance with mocked dependencies."""
        with patch("solr_mcp.solr.client.SolrClient._load_config") as mock_load:
            config = SolrConfig(
                zookeeper_hosts=["localhost:2181"],
                solr_base_url="http://localhost:8983/solr",
                default_collection="collection1"
            )
            mock_load.return_value = config
            client = SolrClient()
            yield client
    
    async def test_search(self, client, mock_solr):
        """Test search functionality."""
        # Arrange
        query = "test query"
        fields = ["id", "title"]
        filters = ["category:books"]
        
        # Act
        result = await client.search(
            query=query,
            fields=fields,
            filters=filters,
            rows=10,
            start=0
        )
        
        # Assert
        mock_solr.search.assert_called_once()
        assert isinstance(result, str)
        
        # Parse result and check content
        result_data = json.loads(result)
        assert result_data["numFound"] == 10
        assert len(result_data["docs"]) == 2
        assert "facets" in result_data
    
    async def test_get_suggestions(self, client, mock_solr):
        """Test suggestions functionality."""
        # Arrange
        query = "doc"
        
        # Act
        result = await client.get_suggestions(query=query, count=5)
        
        # Assert
        mock_solr._send_request.assert_called_once()
        assert isinstance(result, str)
        
        # Parse result and check content
        result_data = json.loads(result)
        assert "suggest" in result_data
    
    async def test_get_facets(self, client, mock_solr):
        """Test facets functionality."""
        # Arrange
        query = "test query"
        facet_fields = ["category", "author"]
        
        # Act
        result = await client.get_facets(
            query=query,
            facet_fields=facet_fields,
            facet_limit=10
        )
        
        # Assert
        mock_solr.search.assert_called_once()
        assert isinstance(result, str)
        
        # Parse result and check content
        result_data = json.loads(result)
        assert "facets" in result_data
    
    def test_list_collections(self, client, mock_kazoo):
        """Test listing collections."""
        # Act
        collections = client.list_collections()
        
        # Assert - note that get_children is called twice:
        # Once during initialization and once when method is called
        mock_kazoo.get_children.assert_called_with("/collections")
        assert mock_kazoo.get_children.call_count >= 1
        assert len(collections) == 2
        assert "collection1" in collections
        assert "collection2" in collections
    
    async def test_vector_search(self, client, mock_solr):
        """Test vector search functionality."""
        # Arrange
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        vector_field = "embedding"
        
        # Act
        result = await client.vector_search(
            vector=vector,
            vector_field=vector_field,
            k=5
        )
        
        # Assert
        mock_solr.search.assert_called_once()
        assert isinstance(result, str)
        
        # Verify the KNN query format
        call_args = mock_solr.search.call_args
        assert call_args is not None
        query = call_args[0][0]
        assert "{!knn" in query
        assert "topK=5" in query
        
        # Verify results format
        result_data = json.loads(result)
        assert "numFound" in result_data
        assert "docs" in result_data
    
    async def test_index_document_with_vector(self, client, mock_solr):
        """Test indexing document with vector."""
        # Arrange
        document = {"id": "doc1", "title": "Test Document"}
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Act
        result = await client.index_document_with_vector(
            document=document,
            vector=vector,
            commit=True
        )
        
        # Assert
        mock_solr.add.assert_called_once()
        assert result is True
        
        # Verify document format
        call_args = mock_solr.add.call_args
        assert call_args is not None
        docs = call_args[0][0]
        assert len(docs) == 1
        assert docs[0]["id"] == "doc1"
        assert docs[0]["embedding"] == vector
    
    async def test_batch_index_with_vectors(self, client, mock_solr):
        """Test batch indexing documents with vectors."""
        # Arrange
        documents = [
            {"id": "doc1", "title": "Document 1"},
            {"id": "doc2", "title": "Document 2"}
        ]
        vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        # Act
        result = await client.batch_index_with_vectors(
            documents=documents,
            vectors=vectors,
            commit=True
        )
        
        # Assert
        mock_solr.add.assert_called_once()
        assert result is True
        
        # Verify documents format
        call_args = mock_solr.add.call_args
        assert call_args is not None
        docs = call_args[0][0]
        assert len(docs) == 2
        assert docs[0]["id"] == "doc1"
        assert docs[0]["embedding"] == vectors[0]
        assert docs[1]["id"] == "doc2"
        assert docs[1]["embedding"] == vectors[1]