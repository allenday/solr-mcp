"""Tests for the list fields tool."""

import pytest

from solr_mcp.solr.exceptions import SolrError
from solr_mcp.tools.solr_list_fields import execute_list_fields

# Sample field data for testing
FIELD_DATA = {
    "fields": [
        {"name": "id", "type": "string", "indexed": True, "stored": True},
        {
            "name": "_text_",
            "type": "text_general",
            "indexed": True,
            "stored": False,
            "copies_from": ["title", "content"],
        },
    ]
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "collection,custom_fields",
    [
        ("test_collection", None),
        (
            "custom_collection",
            [
                {
                    "name": "custom_id",
                    "type": "string",
                    "indexed": True,
                    "stored": True,
                },
                {
                    "name": "custom_text",
                    "type": "text_general",
                    "indexed": True,
                    "stored": False,
                },
            ],
        ),
    ],
)
async def test_execute_list_fields_success(mock_server, collection, custom_fields):
    """Test successful execution of list_fields tool with different collections and field sets."""
    # Use default fields or custom fields based on parameter
    fields = custom_fields or FIELD_DATA["fields"]
    mock_server.solr_client.list_fields.return_value = fields

    # Execute the tool
    result = await execute_list_fields(mock_server, collection)

    # Verify the result
    assert result["collection"] == collection
    assert len(result["fields"]) == len(fields)
    assert result["fields"][0]["name"] == fields[0]["name"]

    # Check for copies_from in the default test case
    if custom_fields is None and "copies_from" in fields[1]:
        assert "copies_from" in result["fields"][1]

    # Verify the correct collection was used
    mock_server.solr_client.list_fields.assert_called_once_with(collection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_message",
    ["Failed to list fields", "Collection not found", "Connection error"],
)
async def test_execute_list_fields_error(mock_solr_client, mock_config, error_message):
    """Test error handling in list_fields tool with different error messages."""
    # Create a server with a parameterized error client
    error_client = mock_solr_client(param={"error": True})

    from solr_mcp.server import SolrMCPServer

    server = SolrMCPServer(
        solr_base_url=mock_config.solr_base_url,
        zookeeper_hosts=mock_config.zookeeper_hosts,
        connection_timeout=mock_config.connection_timeout,
    )
    server.solr_client = error_client

    # Override the exception message
    error_client.list_fields.side_effect = SolrError(error_message)

    # Verify the exception is raised with the correct message
    with pytest.raises(SolrError, match=error_message):
        await execute_list_fields(server, "test_collection")
