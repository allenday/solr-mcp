"""Tests for solr_mcp.solr.utils.formatting module."""

import json
from unittest.mock import Mock, patch

import pytest
from pysolr import Results

from solr_mcp.solr.exceptions import QueryError, SolrError
from solr_mcp.solr.utils.formatting import (
    format_error_response,
    format_search_results,
    format_sql_response,
)


@pytest.fixture
def mock_results():
    """Create a mock pysolr Results object."""
    results = Mock(spec=Results)
    results.hits = 10
    results.docs = [{"id": "1", "title": "Test"}, {"id": "2", "title": "Test 2"}]
    results.max_score = 1.5
    results.facets = {"category": {"test": 5}}
    results.highlighting = {"1": {"title": ["<em>Test</em>"]}}
    return results


def test_format_search_results_basic(mock_results):
    """Test basic search results formatting."""
    formatted = json.loads(format_search_results(mock_results, start=0))
    assert "result-set" in formatted
    assert formatted["result-set"]["numFound"] == 10
    assert formatted["result-set"]["start"] == 0
    assert len(formatted["result-set"]["docs"]) == 2


def test_format_search_results_with_score(mock_results):
    """Test search results formatting with score."""
    formatted = json.loads(format_search_results(mock_results, include_score=True))
    assert formatted["result-set"]["maxScore"] == 1.5


def test_format_search_results_with_facets(mock_results):
    """Test search results formatting with facets."""
    formatted = json.loads(format_search_results(mock_results, include_facets=True))
    assert "facets" in formatted["result-set"]
    assert formatted["result-set"]["facets"] == {"category": {"test": 5}}


def test_format_search_results_with_highlighting(mock_results):
    """Test search results formatting with highlighting."""
    formatted = json.loads(
        format_search_results(mock_results, include_highlighting=True)
    )
    assert "highlighting" in formatted["result-set"]
    assert formatted["result-set"]["highlighting"] == {
        "1": {"title": ["<em>Test</em>"]}
    }


def test_format_search_results_without_optional_fields(mock_results):
    """Test search results formatting without optional fields."""
    formatted = json.loads(
        format_search_results(
            mock_results,
            include_score=False,
            include_facets=False,
            include_highlighting=False,
        )
    )
    assert "maxScore" not in formatted["result-set"]
    assert "facets" not in formatted["result-set"]
    assert "highlighting" not in formatted["result-set"]


def test_format_search_results_json_error(mock_results):
    """Test handling of JSON serialization errors."""

    # Create an object that can't be JSON serialized
    class UnserializableObject:
        pass

    mock_results.docs = [UnserializableObject()]

    formatted = json.loads(format_search_results(mock_results))
    assert "result-set" in formatted
    assert isinstance(formatted["result-set"]["docs"][0], str)


def test_format_search_results_general_error():
    """Test handling of general errors."""
    results = None  # This will cause an attribute error
    formatted = json.loads(format_search_results(results))
    assert "error" in formatted


def test_format_sql_response_success():
    """Test SQL response formatting with successful response."""
    raw_response = {"result-set": {"docs": [{"id": "1"}, {"id": "2"}]}}
    formatted = format_sql_response(raw_response)
    assert formatted["result-set"]["docs"] == [{"id": "1"}, {"id": "2"}]
    assert formatted["result-set"]["numFound"] == 2
    assert formatted["result-set"]["start"] == 0


def test_format_sql_response_with_exception():
    """Test SQL response formatting with exception in response."""
    raw_response = {"result-set": {"docs": [{"EXCEPTION": "Test error"}]}}
    with pytest.raises(QueryError, match="Test error"):
        format_sql_response(raw_response)


def test_format_sql_response_error():
    """Test SQL response formatting with general error."""
    raw_response = None  # This will cause an attribute error
    with pytest.raises(QueryError, match="Error formatting SQL response"):
        format_sql_response(raw_response)


def test_format_search_results_missing_docs(mock_results):
    """Test formatting results when docs attribute is missing."""
    delattr(mock_results, "docs")
    formatted = json.loads(format_search_results(mock_results))
    assert formatted["result-set"]["docs"] == []


def test_format_search_results_missing_hits(mock_results):
    """Test formatting results when hits attribute is missing."""
    delattr(mock_results, "hits")
    formatted = json.loads(format_search_results(mock_results))
    assert "error" in formatted
    assert "Mock object has no attribute 'hits'" in formatted["error"]


def test_format_search_results_complex_json_error(mock_results):
    """Test handling of complex JSON serialization errors."""

    # Create a more complex unserializable object
    class ComplexObject:
        def __init__(self):
            self.circular = self

    mock_results.docs = [{"complex": ComplexObject()}]
    formatted = json.loads(format_search_results(mock_results))
    assert "result-set" in formatted
    # The entire document should be converted to a string due to JSON serialization error
    assert all(isinstance(doc, dict) for doc in formatted["result-set"]["docs"])
    assert isinstance(formatted["result-set"]["docs"][0]["complex"], str)


def test_format_sql_response_empty_response():
    """Test SQL response formatting with empty response."""
    raw_response = {}
    formatted = format_sql_response(raw_response)
    assert formatted["result-set"]["docs"] == []
    assert formatted["result-set"]["numFound"] == 0
    assert formatted["result-set"]["start"] == 0


def test_format_sql_response_missing_docs():
    """Test SQL response formatting with missing docs."""
    raw_response = {"result-set": {}}
    formatted = format_sql_response(raw_response)
    assert formatted["result-set"]["docs"] == []
    assert formatted["result-set"]["numFound"] == 0


def test_format_error_response_query_error():
    """Test formatting QueryError response."""
    error = QueryError("Invalid query")
    formatted = json.loads(format_error_response(error))
    assert formatted["error"]["code"] == "QUERY_ERROR"
    assert formatted["error"]["message"] == "Invalid query"


def test_format_error_response_solr_error():
    """Test formatting SolrError response."""
    error = SolrError("Solr connection failed")
    formatted = json.loads(format_error_response(error))
    assert formatted["error"]["code"] == "SOLR_ERROR"
    assert formatted["error"]["message"] == "Solr connection failed"


def test_format_error_response_generic_error():
    """Test formatting generic error response."""
    error = ValueError("Invalid value")
    formatted = json.loads(format_error_response(error))
    assert formatted["error"]["code"] == "INTERNAL_ERROR"
    assert formatted["error"]["message"] == "Invalid value"


def test_format_search_results_with_empty_facets(mock_results):
    """Test formatting results with empty facets."""
    mock_results.facets = {}
    formatted = json.loads(format_search_results(mock_results))
    assert "facets" not in formatted["result-set"]


def test_format_search_results_with_empty_highlighting(mock_results):
    """Test formatting results with empty highlighting."""
    mock_results.highlighting = {}
    formatted = json.loads(format_search_results(mock_results))
    assert "highlighting" not in formatted["result-set"]
