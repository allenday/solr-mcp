"""Unit tests for formatting utilities."""

import json
from unittest.mock import Mock

from solr_mcp.solr.exceptions import QueryError, SolrError
from solr_mcp.solr.utils.formatting import (
    format_error_response,
    format_search_results,
    format_sql_response,
)


class TestFormatting:
    """Test cases for formatting utilities."""

    def test_format_search_results(self):
        """Test formatting Solr search results."""
        # Create mock pysolr Results
        mock_results = Mock()
        mock_results.docs = [
            {"id": "1", "title": "Test 1"},
            {"id": "2", "title": "Test 2"},
        ]
        mock_results.hits = 2
        mock_results.raw_response = {
            "response": {
                "docs": mock_results.docs,
                "numFound": mock_results.hits,
                "start": 0,
            }
        }

        formatted = format_search_results(mock_results, start=0)
        result_dict = json.loads(formatted)

        assert "result-set" in result_dict
        assert result_dict["result-set"]["docs"] == mock_results.docs
        assert result_dict["result-set"]["numFound"] == mock_results.hits
        assert result_dict["result-set"]["start"] == 0

    def test_format_search_results_empty(self):
        """Test formatting empty search results."""
        mock_results = Mock()
        mock_results.docs = []
        mock_results.hits = 0
        mock_results.raw_response = {
            "response": {"docs": [], "numFound": 0, "start": 0}
        }

        formatted = format_search_results(mock_results, start=0)
        result_dict = json.loads(formatted)

        assert "result-set" in result_dict
        assert result_dict["result-set"]["docs"] == []
        assert result_dict["result-set"]["numFound"] == 0
        assert result_dict["result-set"]["start"] == 0

    def test_format_sql_response(self):
        """Test formatting SQL query response."""
        response = {
            "result-set": {
                "docs": [
                    {"id": "1", "title": "Test 1"},
                    {"id": "2", "title": "Test 2"},
                ],
                "numFound": 2,
                "start": 0,
            }
        }

        formatted = format_sql_response(response)

        assert formatted == response
        assert "result-set" in formatted
        assert formatted["result-set"]["numFound"] == 2
        assert len(formatted["result-set"]["docs"]) == 2

    def test_format_sql_response_empty(self):
        """Test formatting empty SQL query response."""
        response = {"result-set": {"docs": [], "numFound": 0, "start": 0}}

        formatted = format_sql_response(response)

        assert formatted == response
        assert "result-set" in formatted
        assert formatted["result-set"]["numFound"] == 0
        assert formatted["result-set"]["docs"] == []

    def test_format_error_response_query_error(self):
        """Test formatting QueryError response."""
        error = QueryError("Invalid SQL syntax")
        formatted = format_error_response(error)
        error_dict = json.loads(formatted)

        assert "error" in error_dict
        assert error_dict["error"]["code"] == "QUERY_ERROR"
        assert error_dict["error"]["message"] == "Invalid SQL syntax"

    def test_format_error_response_solr_error(self):
        """Test formatting SolrError response."""
        error = SolrError("Connection failed")
        formatted = format_error_response(error)
        error_dict = json.loads(formatted)

        assert "error" in error_dict
        assert error_dict["error"]["code"] == "SOLR_ERROR"
        assert error_dict["error"]["message"] == "Connection failed"

    def test_format_error_response_generic_error(self):
        """Test formatting generic error response."""
        error = Exception("Unknown error")
        formatted = format_error_response(error)
        error_dict = json.loads(formatted)

        assert "error" in error_dict
        assert error_dict["error"]["code"] == "INTERNAL_ERROR"
        assert "Unknown error" in error_dict["error"]["message"]
