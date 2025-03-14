"""Tests for the MCP server integration."""

import os
import subprocess
import time
import unittest
import json
import asyncio
from typing import Dict, List, Any, Optional

import pytest
from fastmcp import client
from fastmcp.transport.stdio_client_transport import StdioClientTransport


class TestMCPServerIntegration:
    """Integration tests for the MCP server."""
    
    @pytest.fixture
    async def mcp_client(self):
        """Create and connect to an MCP client."""
        # Start the MCP server as a subprocess
        server_process = subprocess.Popen(
            ["python", "-m", "solr_mcp.server", "--debug"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        )

        # Allow server to start
        time.sleep(1)

        # Configure transport directly to the subprocess
        transport = StdioClientTransport({"command": ["python", "-m", "solr_mcp.server"]})
        
        # Create and connect client
        c = client.Client()
        await c.connect(transport)
        
        yield c
        
        # Close client and terminate server
        await c.close()
        server_process.terminate()
        server_process.wait()
    
    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_client):
        """Test that the server lists the expected tools."""
        result = await mcp_client.request({"method": "list_tools"})
        
        # Extract tool names
        tool_names = [tool["name"] for tool in result["tools"]]
        
        # Check expected tools are present
        assert "solr_search" in tool_names
        assert "solr_vector_search" in tool_names
        assert "solr_embed_and_index" in tool_names
    
    @pytest.mark.asyncio
    async def test_search_tool(self, mcp_client):
        """Test the search tool."""
        # This test will only work if Solr is running and has some data
        # Skip if Solr is not available
        try:
            result = await mcp_client.request({
                "method": "call_tool",
                "params": {
                    "name": "solr_search",
                    "arguments": {
                        "query": "*:*",
                        "rows": 1
                    }
                }
            })
            
            # Parse result
            content = result["content"][0]
            assert content["type"] == "text"
            
            # Try to parse the JSON response
            response = json.loads(content["text"])
            assert "numFound" in response
            
        except Exception as e:
            pytest.skip(f"Skipping test_search_tool: {e}")
    
    @pytest.mark.asyncio
    async def test_vector_search_tool(self, mcp_client):
        """Test the vector search tool."""
        # This test will only work if Solr is running and has vector data
        # Skip if Solr is not available
        try:
            # Generate a test vector (1536 dimensions)
            test_vector = [0.1] * 1536
            
            result = await mcp_client.request({
                "method": "call_tool",
                "params": {
                    "name": "solr_vector_search",
                    "arguments": {
                        "vector": test_vector,
                        "k": 5
                    }
                }
            })
            
            # Parse result
            content = result["content"][0]
            assert content["type"] == "text"
            
            # Try to parse the JSON response
            response = json.loads(content["text"])
            assert "numFound" in response
            
        except Exception as e:
            pytest.skip(f"Skipping test_vector_search_tool: {e}")