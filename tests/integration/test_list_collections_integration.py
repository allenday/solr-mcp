"""Integration tests for listing Solr collections through MCP."""

import pytest
from mcp import client
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import JSONRPCMessage

@pytest.mark.asyncio
@pytest.mark.integration
async def test_list_collections_through_mcp():
    """Test listing collections through MCP server.
    
    Requirements:
    - Running Solr instance with 'unified' collection
    """
    # Set up server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "solr_mcp.server"]
    )
    
    # Use stdio_client context manager to handle server communication
    async with stdio_client(server_params) as (read_stream, write_stream):
        request = JSONRPCMessage(
            jsonrpc="2.0",
            id=1,
            method="execute_list_collections",
            params={"mcp": "server"}
        )
        await write_stream.send(request)
        
        # Get response
        response = await read_stream.receive()
        assert isinstance(response, JSONRPCMessage)
        assert response.result is not None
        assert isinstance(response.result, list)
        assert "unified" in response.result
        assert len(response.result) > 0, "Expected at least one collection" 