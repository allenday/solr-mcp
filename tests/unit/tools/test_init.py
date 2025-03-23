"""Tests for tools module initialization."""

import pytest

from solr_mcp.tools import (
    TOOLS_DEFINITION,
    execute_list_collections,
    execute_select_query,
    execute_vector_select_query,
    execute_semantic_select_query,
)

def test_tools_definition():
    """Test that TOOLS_DEFINITION contains all expected tools."""
    # All tools should be in TOOLS_DEFINITION
    tools = {
        "solr_list_collections": execute_list_collections,
        "solr_select": execute_select_query,
        "solr_vector_select": execute_vector_select_query,
        "solr_semantic_select": execute_semantic_select_query,
    }
    
    assert len(TOOLS_DEFINITION) == len(tools)
    
    for tool in TOOLS_DEFINITION:
        assert tool._is_tool
        assert tool._tool_name in tools
        assert tool == tools[tool._tool_name]

def test_tools_exports():
    """Test that __all__ exports all tools."""
    from solr_mcp.tools import __all__
    
    expected = {
        "execute_list_collections",
        "execute_select_query",
        "execute_vector_select_query",
        "execute_semantic_select_query",
    }
    
    assert set(__all__) == expected 