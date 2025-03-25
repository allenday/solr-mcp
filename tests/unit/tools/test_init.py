"""Test tools initialization."""

import pytest

from solr_mcp.tools import (
    execute_list_collections,
    execute_list_fields,
    execute_select_query,
    execute_vector_select_query,
    execute_semantic_select_query,
    TOOLS_DEFINITION
)

def test_tools_definition():
    """Test that TOOLS_DEFINITION contains all expected tools."""
    # All tools should be in TOOLS_DEFINITION
    tools = {
        "solr_list_collections": execute_list_collections,
        "solr_list_fields": execute_list_fields,
        "solr_select": execute_select_query,
        "solr_vector_select": execute_vector_select_query,
        "solr_semantic_select": execute_semantic_select_query,
    }
    
    assert len(TOOLS_DEFINITION) == len(tools)
    
    for tool_name, tool_func in tools.items():
        assert tool_func in TOOLS_DEFINITION

def test_tools_exports():
    """Test that __all__ exports all tools."""
    from solr_mcp.tools import __all__
    
    expected = {
        "execute_list_collections",
        "execute_list_fields",
        "execute_select_query",
        "execute_vector_select_query",
        "execute_semantic_select_query",
    }
    
    assert set(__all__) == expected 