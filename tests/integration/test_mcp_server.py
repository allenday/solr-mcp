"""Tests for the MCP server integration."""

import os
import sys
import subprocess
import time
import unittest
import json
import asyncio
import importlib
from typing import Dict, List, Any, Optional

import pytest
import requests

# Skip all tests if Solr is not running
def is_solr_running():
    """Check if Solr is running."""
    try:
        response = requests.get("http://localhost:8983/solr/admin/collections")
        return response.status_code == 200
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not is_solr_running(),
    reason="Solr is not running. Start Solr with docker-compose up -d"
)

# Import the server module to test it directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from solr_mcp import server


@pytest.mark.asyncio
class TestMCPServerIntegration:
    """Integration tests for the MCP server."""
    
    @pytest.fixture
    async def tools(self):
        """Create and connect to the MCP server."""
        # Setup server tools - this is what happens in server.py
        tools = server.setup_tools()
        yield tools
    
    async def test_solr_search_tool(self, tools):
        """Test the search tool."""
        search_tool = next((t for t in tools if t.name == "solr_search"), None)
        if not search_tool:
            pytest.skip("solr_search tool not available")
        
        # Execute the tool directly
        result = await search_tool.execute({
            "query": "*:*",
            "rows": 1
        })
        
        # Result should be a JSON string
        assert isinstance(result, str)
        
        # Parse the response
        response = json.loads(result)
        assert "numFound" in response
    
    async def test_solr_vector_search_tool(self, tools):
        """Test the vector search tool."""
        vector_search_tool = next((t for t in tools if t.name == "solr_vector_search"), None)
        if not vector_search_tool:
            pytest.skip("solr_vector_search tool not available")
        
        # Generate a test vector (768 dimensions for nomic-embed-text)
        test_vector = [0.1] * 768
        
        # Execute the tool directly
        result = await vector_search_tool.execute({
            "vector": test_vector,
            "k": 5
        })
        
        # Result should be a JSON string
        assert isinstance(result, str)
        
        # Parse the response
        response = json.loads(result)
        assert "numFound" in response
    
    async def test_hybrid_search_tool(self, tools):
        """Test the hybrid search tool."""
        hybrid_search_tool = next((t for t in tools if t.name == "solr_hybrid_search"), None)
        if not hybrid_search_tool:
            pytest.skip("solr_hybrid_search tool not available")
        
        # Execute the tool directly
        result = await hybrid_search_tool.execute({
            "query": "bitcoin",
            "rows": 5
        })
        
        # Result should be a JSON string
        assert isinstance(result, str)
        
        # Parse the response
        response = json.loads(result)
        assert "numFound" in response