"""Tests for solr_mcp.solr.vector.results module."""

import pytest
from typing import Dict, Any

from solr_mcp.solr.vector.results import VectorSearchResult, VectorSearchResults

@pytest.fixture
def sample_result_data() -> Dict[str, Any]:
    """Create sample result data."""
    return {
        "docid": "123",
        "score": 0.95,
        "distance": 0.05,
        "metadata": {"title": "Test Document", "author": "Test Author"}
    }

@pytest.fixture
def sample_solr_response() -> Dict[str, Any]:
    """Create sample Solr response."""
    return {
        "responseHeader": {
            "QTime": 50
        },
        "response": {
            "numFound": 2,
            "docs": [
                {
                    "_docid_": "123",
                    "score": 0.95,
                    "_vector_distance_": 0.05,
                    "title": "Test Document 1",
                    "author": "Test Author 1"
                },
                {
                    "_docid_": "456",
                    "score": 0.85,
                    "_vector_distance_": 0.15,
                    "title": "Test Document 2",
                    "author": "Test Author 2"
                }
            ]
        }
    }

def test_vector_search_result_creation(sample_result_data):
    """Test VectorSearchResult creation and properties."""
    result = VectorSearchResult(**sample_result_data)
    assert result.docid == "123"
    assert result.score == 0.95
    assert result.distance == 0.05
    assert result.metadata == {"title": "Test Document", "author": "Test Author"}

def test_vector_search_result_subscript(sample_result_data):
    """Test VectorSearchResult subscript access."""
    result = VectorSearchResult(**sample_result_data)
    assert result["docid"] == "123"
    assert result["score"] == 0.95
    assert result["distance"] == 0.05
    assert result["metadata"] == {"title": "Test Document", "author": "Test Author"}

def test_vector_search_result_invalid_key(sample_result_data):
    """Test VectorSearchResult invalid key access."""
    result = VectorSearchResult(**sample_result_data)
    with pytest.raises(KeyError, match="Invalid key: invalid_key"):
        _ = result["invalid_key"]

def test_vector_search_results_creation(sample_solr_response):
    """Test VectorSearchResults creation from Solr response."""
    results = VectorSearchResults.from_solr_response(sample_solr_response, top_k=10)
    assert len(results.results) == 2
    assert results.total_found == 2
    assert results.top_k == 10
    assert results.query_time_ms == 50

def test_vector_search_results_docs_property(sample_solr_response):
    """Test VectorSearchResults docs property."""
    results = VectorSearchResults.from_solr_response(sample_solr_response, top_k=10)
    docs = results.docs
    assert len(docs) == 2
    assert isinstance(docs[0], VectorSearchResult)
    assert docs[0].docid == "123"
    assert docs[1].docid == "456"

def test_vector_search_results_alternate_docid_fields():
    """Test VectorSearchResults with alternate docid field names."""
    response = {
        "response": {
            "numFound": 1,
            "docs": [
                {
                    "[docid]": "123",
                    "score": 0.95
                }
            ]
        }
    }
    results = VectorSearchResults.from_solr_response(response, top_k=10)
    assert results.results[0].docid == "123"

    response["response"]["docs"][0] = {"docid": "456", "score": 0.85}
    results = VectorSearchResults.from_solr_response(response, top_k=10)
    assert results.results[0].docid == "456"

def test_vector_search_results_missing_docid():
    """Test VectorSearchResults with missing docid field."""
    response = {
        "response": {
            "numFound": 1,
            "docs": [
                {
                    "score": 0.95
                }
            ]
        }
    }
    results = VectorSearchResults.from_solr_response(response, top_k=10)
    assert results.results[0].docid == "0"

def test_vector_search_results_to_dict(sample_solr_response):
    """Test VectorSearchResults to_dict method."""
    results = VectorSearchResults.from_solr_response(sample_solr_response, top_k=10)
    result_dict = results.to_dict()
    
    assert "results" in result_dict
    assert "metadata" in result_dict
    assert len(result_dict["results"]) == 2
    assert result_dict["metadata"]["total_found"] == 2
    assert result_dict["metadata"]["top_k"] == 10
    assert result_dict["metadata"]["query_time_ms"] == 50

def test_vector_search_results_get_methods(sample_solr_response):
    """Test VectorSearchResults getter methods."""
    results = VectorSearchResults.from_solr_response(sample_solr_response, top_k=10)
    
    assert results.get_doc_ids() == ["123", "456"]
    assert results.get_scores() == [0.95, 0.85]
    assert results.get_distances() == [0.05, 0.15]

def test_vector_search_results_empty_response():
    """Test VectorSearchResults with empty response."""
    empty_response = {
        "responseHeader": {},
        "response": {
            "numFound": 0,
            "docs": []
        }
    }
    results = VectorSearchResults.from_solr_response(empty_response, top_k=10)
    assert len(results.results) == 0
    assert results.total_found == 0
    assert results.get_doc_ids() == []
    assert results.get_scores() == []
    assert results.get_distances() == []

def test_vector_search_results_minimal_response():
    """Test VectorSearchResults with minimal response."""
    minimal_response = {
        "response": {
            "docs": [{"_docid_": "123"}]
        }
    }
    results = VectorSearchResults.from_solr_response(minimal_response, top_k=10)
    assert len(results.results) == 1
    assert results.total_found == 0  # Default when numFound is missing
    assert results.query_time_ms is None  # Default when QTime is missing 