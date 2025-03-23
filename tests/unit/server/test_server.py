"""Tests for Solr MCP server."""

import json
import pytest
from unittest.mock import AsyncMock
from mcp.server.fastmcp.exceptions import ToolError

from solr_mcp.server import SolrMCPServer
from solr_mcp.solr.exceptions import ConnectionError, QueryError

class TestSolrMCPServer:
    """Test cases for SolrMCPServer."""

    @pytest.mark.asyncio
    async def test_handle_list_collections_success(self, mock_server):
        """Test successful list collections handling."""
        result = await mock_server.mcp.call_tool("execute_list_collections", {"mcp": mock_server})
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_handle_list_collections_with_string_server(self, mock_server):
        """Test that string server names are handled correctly."""
        # Call with string server name
        result = await mock_server.mcp.call_tool("execute_list_collections", {"mcp": "solr_mcp"})
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_handle_list_collections_error(self, mock_error_server):
        """Test list collections error handling."""
        with pytest.raises(ToolError, match="Error executing tool execute_list_collections: Test error"):
            await mock_error_server.mcp.call_tool("execute_list_collections", {"mcp": mock_error_server})

    @pytest.mark.asyncio
    async def test_handle_execute_select_query_success(self, mock_server):
        """Test successful select query handling."""
        query = "SELECT * FROM collection1"
        result = await mock_server.mcp.call_tool("execute_select_query", {
            "mcp": mock_server,
            "query": query
        })
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].type == "text"

    @pytest.mark.asyncio
    async def test_handle_execute_select_query_missing_query(self, mock_server):
        """Test select query with missing query."""
        with pytest.raises(Exception, match="Input should be a valid string"):
            await mock_server.mcp.call_tool("execute_select_query", {
                "mcp": mock_server,
                "query": None
            })

    @pytest.mark.asyncio
    async def test_handle_execute_select_query_invalid_query(self, mock_error_server):
        """Test select query with invalid query."""
        with pytest.raises(ToolError, match="Error executing tool execute_select_query: Test error"):
            await mock_error_server.mcp.call_tool("execute_select_query", {
                "mcp": mock_error_server,
                "query": "INVALID"
            })

    @pytest.mark.asyncio
    async def test_handle_execute_vector_select_query_success(self, mock_server):
        """Test successful vector select query handling."""
        query = "SELECT * FROM collection1"
        vector = [0.1, 0.2, 0.3]
        result = await mock_server.mcp.call_tool("execute_vector_select_query", {
            "mcp": mock_server,
            "query": query,
            "vector": vector
        })
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].type == "text"

    @pytest.mark.asyncio
    async def test_handle_execute_vector_select_query_missing_params(self, mock_server):
        """Test vector select query with missing parameters."""
        with pytest.raises(Exception, match="Input should be a valid"):
            await mock_server.mcp.call_tool("execute_vector_select_query", {
                "mcp": mock_server,
                "query": None,
                "vector": None
            })

    @pytest.mark.asyncio
    async def test_handle_execute_vector_select_query_invalid_vector(self, mock_server):
        """Test vector select query with invalid vector."""
        mock_server.solr_client.execute_vector_select_query.side_effect = Exception("Invalid vector")
        with pytest.raises(Exception, match="Input should be a valid list"):
            await mock_server.mcp.call_tool("execute_vector_select_query", {
                "mcp": mock_server,
                "query": "SELECT * FROM collection1",
                "vector": "not a vector"
            })

    @pytest.mark.asyncio
    async def test_handle_execute_semantic_select_query_success(self, mock_server):
        """Test successful semantic select query handling."""
        query = "SELECT * FROM collection1"
        text = "sample search text"
        result = await mock_server.mcp.call_tool("execute_semantic_select_query", {
            "mcp": mock_server,
            "query": query,
            "text": text
        })
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].type == "text"

    @pytest.mark.asyncio
    async def test_handle_execute_semantic_select_query_missing_params(self, mock_server):
        """Test semantic select query with missing parameters."""
        with pytest.raises(Exception, match="Input should be a valid"):
            await mock_server.mcp.call_tool("execute_semantic_select_query", {
                "mcp": mock_server,
                "query": None,
                "text": None
            }) 