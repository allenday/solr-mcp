"""Unit tests for vector search functionality."""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

import pysolr

from solr_mcp.solr.vector import VectorManager
from solr_mcp.solr.exceptions import SolrError

class TestVectorManager:
    """Test suite for VectorManager."""

    def test_init(self, mock_ollama, mock_solr_instance):
        """Test VectorManager initialization."""
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        assert manager.client == mock_ollama
        assert manager.solr_client == mock_solr_instance

    @pytest.mark.asyncio
    async def test_get_vector_success(self, mock_ollama, mock_solr_instance):
        """Test successful vector generation."""
        mock_ollama.get_vector = AsyncMock(return_value=[0.1, 0.2, 0.3])
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        result = await manager.get_vector("test text")
        assert result == [0.1, 0.2, 0.3]
        mock_ollama.get_vector.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_get_vector_error(self, mock_ollama, mock_solr_instance):
        """Test vector generation error handling."""
        mock_ollama.get_vector = AsyncMock(side_effect=Exception("Test error"))
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        with pytest.raises(SolrError) as exc_info:
            await manager.get_vector("test text")
        assert "Error getting vector" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_vector_no_client(self, mock_solr_instance):
        """Test vector generation with no client."""
        manager = VectorManager(solr_client=mock_solr_instance)
        manager.client = None  # Override the default client
        with pytest.raises(SolrError) as exc_info:
            await manager.get_vector("test text")
        assert "Vector operations unavailable" in str(exc_info.value)

    def test_format_knn_query(self, mock_ollama, mock_solr_instance):
        """Test KNN query formatting."""
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        vector = [0.1, 0.2, 0.3]

        # Test with default top_k
        query = manager.format_knn_query(vector, "vector_field")
        assert query == "{!knn f=vector_field}[0.1,0.2,0.3]"

        # Test with specified top_k
        query = manager.format_knn_query(vector, "vector_field", top_k=5)
        assert query == "{!knn f=vector_field topK=5}[0.1,0.2,0.3]"

    @pytest.mark.asyncio
    async def test_execute_vector_search_success(self, mock_ollama, mock_solr_instance):
        """Test successful vector search execution."""
        mock_solr_instance.search.return_value = {
            "responseHeader": {
                "status": 0,
                "QTime": 10
            },
            "response": {
                "docs": [{"_docid_": "1", "score": 0.95, "_vector_distance_": 0.05}],
                "numFound": 1,
                "maxScore": 0.95
            }
        }
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        vector = [0.1, 0.2, 0.3]

        # Test without filter query
        results = await manager.execute_vector_search(mock_solr_instance, vector, "vector_field")
        assert mock_solr_instance.search.call_count == 1
        assert mock_solr_instance.search.call_args[0][0] == \
            "{!knn f=vector_field}[0.1,0.2,0.3]"

        # Test with filter query
        results = await manager.execute_vector_search(
            mock_solr_instance,
            vector,
            "vector_field",
            filter_query="field:value"
        )
        assert mock_solr_instance.search.call_count == 2
        assert mock_solr_instance.search.call_args[0][0] == \
            "{!knn f=vector_field}[0.1,0.2,0.3]"
        assert mock_solr_instance.search.call_args[1]["fq"] == "field:value"

    @pytest.mark.asyncio
    async def test_execute_vector_search_error(self, mock_ollama, mock_solr_instance):
        """Test error handling in vector search."""
        mock_solr_instance.search.side_effect = Exception("Search error")
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        vector = [0.1, 0.2, 0.3]
        with pytest.raises(SolrError, match="Vector search failed"):
            await manager.execute_vector_search(mock_solr_instance, vector, "vector_field")

def test_vector_manager_init():
    """Test VectorManager initialization."""
    manager = VectorManager(solr_client=None)
    assert manager.client is not None  # Should create default OllamaVectorProvider
    assert manager.solr_client == None 