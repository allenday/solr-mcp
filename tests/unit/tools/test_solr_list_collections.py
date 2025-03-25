"""Tests for Solr list collections tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from solr_mcp.server import SolrMCPServer
from solr_mcp.tools.solr_list_collections import execute_list_collections


@pytest.mark.asyncio
class TestListCollectionsTool:
    """Test list collections tool."""

    async def test_execute_list_collections_requires_server_instance(self):
        """Test that execute_list_collections requires a proper server instance."""
        # Test with string instead of server instance
        with pytest.raises(
            AttributeError, match="'str' object has no attribute 'solr_client'"
        ):
            await execute_list_collections("server")

    async def test_execute_list_collections_success(self):
        """Test successful list collections execution."""
        # Create mock server instance with solr_client
        mock_server = MagicMock(spec=SolrMCPServer)
        mock_solr_client = AsyncMock()
        mock_solr_client.list_collections.return_value = ["unified", "collection2"]
        mock_server.solr_client = mock_solr_client

        # Execute tool
        result = await execute_list_collections(mock_server)

        # Verify result
        assert isinstance(result, list)
        assert "unified" in result
        assert len(result) == 2
        mock_solr_client.list_collections.assert_called_once()

    async def test_execute_list_collections_error(self):
        """Test list collections error handling."""
        # Create mock server instance with failing solr_client
        mock_server = MagicMock(spec=SolrMCPServer)
        mock_solr_client = AsyncMock()
        mock_solr_client.list_collections.side_effect = Exception(
            "Failed to list collections"
        )
        mock_server.solr_client = mock_solr_client

        # Execute tool and verify error is propagated
        with pytest.raises(Exception, match="Failed to list collections"):
            await execute_list_collections(mock_server)
