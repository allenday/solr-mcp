"""Unit tests for VectorManager."""

import pytest
from solr_mcp.solr.vector.manager import VectorManager
from solr_mcp.solr.vector.results import VectorSearchResults, VectorSearchResult
from solr_mcp.solr.exceptions import SolrError

class TestVectorManager:
    """Test cases for VectorManager."""

    def test_init(self, mock_ollama, mock_solr_instance):
        """Test VectorManager initialization."""
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        assert manager.client == mock_ollama
        assert manager.solr_client == mock_solr_instance
        assert manager.embedding_field == "embedding"
        assert manager.default_top_k == 10

    def test_init_no_client(self, mock_solr_instance):
        """Test VectorManager initialization without client."""
        manager = VectorManager(solr_client=mock_solr_instance)
        assert manager.client is not None  # Should create default OllamaVectorProvider
        assert manager.solr_client == mock_solr_instance

    @pytest.mark.asyncio
    async def test_get_embedding_success(self, mock_ollama, mock_solr_instance):
        """Test successful embedding generation."""
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        result = await manager.get_embedding("test text")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_embedding_error(self, mock_ollama, mock_solr_instance):
        """Test error handling in embedding generation."""
        mock_ollama.get_embedding = lambda _: (_ for _ in ()).throw(Exception("Embedding error"))
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        with pytest.raises(SolrError, match="Error getting embedding"):
            await manager.get_embedding("test text")

    def test_format_knn_query(self, mock_ollama, mock_solr_instance):
        """Test KNN query formatting."""
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        vector = [0.1, 0.2, 0.3]
        
        # Test with default top_k
        query = manager.format_knn_query(vector)
        assert query == "{!knn f=embedding}[0.1,0.2,0.3]"
        
        # Test with custom top_k
        query = manager.format_knn_query(vector, top_k=5)
        assert query == "{!knn f=embedding topK=5}[0.1,0.2,0.3]"

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
        results = await manager.execute_vector_search(mock_solr_instance, vector)
        mock_solr_instance.search.assert_called_with(
            "{!knn f=embedding}[0.1,0.2,0.3]",
            fq=None,
            fl="_docid_,score,_vector_distance_"
        )
        assert results["response"]["docs"][0]["score"] == 0.95
        
        # Test with filter query
        filter_query = "type:document"
        results = await manager.execute_vector_search(mock_solr_instance, vector, filter_query=filter_query)
        mock_solr_instance.search.assert_called_with(
            "{!knn f=embedding}[0.1,0.2,0.3]",
            fq=filter_query,
            fl="_docid_,score,_vector_distance_"
        )

    @pytest.mark.asyncio
    async def test_execute_vector_search_error(self, mock_ollama, mock_solr_instance):
        """Test error handling in vector search."""
        mock_solr_instance.search.side_effect = Exception("Search error")
        manager = VectorManager(solr_client=mock_solr_instance, client=mock_ollama)
        vector = [0.1, 0.2, 0.3]
        with pytest.raises(SolrError, match="Vector search failed"):
            await manager.execute_vector_search(mock_solr_instance, vector)

class TestVectorSearchResults:
    """Test cases for VectorSearchResults."""

    def test_from_solr_response(self):
        """Test creating results from Solr response."""
        response = {
            "responseHeader": {
                "QTime": 10,
                "status": 0
            },
            "response": {
                "docs": [
                    {"_docid_": "1", "score": 0.95, "_vector_distance_": 0.05},
                    {"_docid_": "2", "score": 0.85, "_vector_distance_": 0.15}
                ],
                "numFound": 2,
                "maxScore": 0.95
            }
        }
        
        results = VectorSearchResults.from_solr_response(response, top_k=2)
        assert len(results.results) == 2
        assert results.results[0].docid == "1"
        assert results.results[0].score == 0.95
        assert results.results[0].distance == 0.05
        assert results.total_found == 2
        assert results.query_time_ms == 10

    def test_from_solr_response_no_vector_distance(self):
        """Test creating results without vector distances."""
        response = {
            "responseHeader": {
                "QTime": 10,
                "status": 0
            },
            "response": {
                "docs": [
                    {"_docid_": "1", "score": 0.95},
                    {"_docid_": "2", "score": 0.85}
                ],
                "numFound": 2,
                "maxScore": 0.95
            }
        }
        
        results = VectorSearchResults.from_solr_response(response, top_k=2)
        assert len(results.results) == 2
        assert results.results[0].docid == "1"
        assert results.results[0].score == 0.95
        assert results.results[0].distance is None

    def test_get_doc_ids(self):
        """Test getting document IDs from results."""
        results = VectorSearchResults(
            results=[
                VectorSearchResult(docid="1", score=0.95),
                VectorSearchResult(docid="2", score=0.85)
            ],
            total_found=2,
            top_k=2
        )
        
        doc_ids = results.get_doc_ids()
        assert len(doc_ids) == 2
        assert "1" in doc_ids
        assert "2" in doc_ids

def test_vector_manager_init():
    """Test VectorManager initialization."""
    manager = VectorManager(solr_client=None)
    assert manager.client is not None  # Should create default OllamaVectorProvider
    assert manager.solr_client == None
    assert manager.embedding_field == "embedding"
    assert manager.default_top_k == 10 