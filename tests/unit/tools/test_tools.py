"""Tests for Solr MCP tools."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from solr_mcp.tools.solr_list_collections import execute_list_collections
from solr_mcp.tools.solr_select import execute_select_query
from solr_mcp.tools.solr_vector_select import execute_vector_select_query
from solr_mcp.tools.solr_semantic_select import execute_semantic_select_query

@pytest.mark.asyncio
class TestListCollectionsTool:
    """Test list collections tool."""
    
    async def test_execute_list_collections(self, mock_server_instance):
        """Test list collections execution."""
        # Setup mock
        mock_solr_client = AsyncMock()
        mock_solr_client.list_collections.return_value = ["collection1", "collection2"]
        mock_server_instance.solr_client = mock_solr_client
        
        # Execute tool
        result = await execute_list_collections(mock_server_instance)
        
        # Verify result
        assert result == ["collection1", "collection2"]
        mock_solr_client.list_collections.assert_called_once()
        
@pytest.mark.asyncio
class TestSelectQueryTool:
    """Test select query tool."""
    
    async def test_execute_select_query(self, mock_server_instance):
        """Test select query execution."""
        # Setup mock
        mock_solr_client = AsyncMock()
        mock_solr_client.execute_select_query.return_value = {
            "rows": [{"id": "1"}]
        }
        mock_server_instance.solr_client = mock_solr_client
        
        # Execute tool
        query = "SELECT * FROM collection1"
        result = await execute_select_query(mock_server_instance, query)
        
        # Verify result
        assert result == {"rows": [{"id": "1"}]}
        mock_solr_client.execute_select_query.assert_called_once_with(query)
        
@pytest.mark.asyncio
class TestVectorSelectTool:
    """Test vector select tool."""
    
    async def test_execute_vector_select_query(self, mock_server_instance):
        """Test vector select query execution."""
        # Setup mock
        mock_solr_client = AsyncMock()
        mock_solr_client.execute_vector_select_query.return_value = {
            "rows": [{"id": "1"}]
        }
        mock_server_instance.solr_client = mock_solr_client
        
        # Execute tool
        query = "SELECT * FROM collection1"
        vector = [0.1, 0.2, 0.3]
        result = await execute_vector_select_query(mock_server_instance, query, vector)
        
        # Verify result
        assert result == {"rows": [{"id": "1"}]}
        mock_solr_client.execute_vector_select_query.assert_called_once_with(query, vector)
        
@pytest.mark.asyncio
class TestSemanticSelectTool:
    """Test semantic select tool."""
    
    async def test_execute_semantic_select_query(self, mock_server_instance):
        """Test semantic select query execution."""
        # Setup mock
        mock_solr_client = AsyncMock()
        mock_solr_client.execute_semantic_select_query.return_value = {
            "rows": [{"id": "1"}]
        }
        mock_server_instance.solr_client = mock_solr_client
        
        # Execute tool
        query = "SELECT * FROM collection1"
        text = "sample search text"
        result = await execute_semantic_select_query(mock_server_instance, query, text)
        
        # Verify result
        assert result == {"rows": [{"id": "1"}]}
        mock_solr_client.execute_semantic_select_query.assert_called_once_with(query, text)
        
class TestToolMetadata:
    """Test tool metadata."""
    
    def test_list_collections_metadata(self):
        """Test list collections tool metadata."""
        assert hasattr(execute_list_collections, "_tool_name")
        assert execute_list_collections._tool_name == "solr_list_collections"
        
    def test_select_query_metadata(self):
        """Test select query tool metadata."""
        assert hasattr(execute_select_query, "_tool_name")
        assert execute_select_query._tool_name == "solr_select"
        
    def test_vector_select_metadata(self):
        """Test vector select tool metadata."""
        assert hasattr(execute_vector_select_query, "_tool_name")
        assert execute_vector_select_query._tool_name == "solr_vector_select"
        
    def test_semantic_select_metadata(self):
        """Test semantic select tool metadata."""
        assert hasattr(execute_semantic_select_query, "_tool_name")
        assert execute_semantic_select_query._tool_name == "solr_semantic_select" 