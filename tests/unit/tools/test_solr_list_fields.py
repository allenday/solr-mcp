"""Tests for the list fields tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from solr_mcp.tools.solr_list_fields import execute_list_fields
from solr_mcp.solr.exceptions import SolrError

@pytest.mark.asyncio
async def test_execute_list_fields_success(mock_server):
    """Test successful execution of list_fields tool."""
    # Set up mock response
    mock_fields = {
        "fields": [
            {
                "name": "id",
                "type": "string",
                "indexed": True,
                "stored": True
            },
            {
                "name": "_text_",
                "type": "text_general",
                "indexed": True,
                "stored": False,
                "copies_from": ["title", "content"]
            }
        ]
    }
    mock_server.solr_client.list_fields = AsyncMock(return_value=mock_fields["fields"])

    result = await execute_list_fields(mock_server, "test_collection")

    assert result["collection"] == "test_collection"
    assert len(result["fields"]) == 2
    assert result["fields"][0]["name"] == "id"
    assert result["fields"][1]["name"] == "_text_"
    assert "copies_from" in result["fields"][1]
    mock_server.solr_client.list_fields.assert_called_once_with("test_collection")

@pytest.mark.asyncio
async def test_execute_list_fields_error(mock_error_server):
    """Test error handling in list_fields tool."""
    mock_error_server.solr_client.list_fields = AsyncMock(side_effect=SolrError("Failed to list fields"))

    with pytest.raises(SolrError, match="Failed to list fields"):
        await execute_list_fields(mock_error_server, "test_collection") 