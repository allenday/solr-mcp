"""Tests for base tool decorator."""

from typing import Dict, List

import pytest

from solr_mcp.tools.base import tool

def test_tool_decorator_default_values():
    """Test tool decorator with default values."""
    @tool()
    def sample_tool() -> str:
        """Sample tool docstring."""
        return "test"

    assert hasattr(sample_tool, "_is_tool")
    assert sample_tool._is_tool is True
    assert sample_tool._tool_name == "sample_tool"
    assert "Sample tool docstring" in sample_tool._tool_description
    assert sample_tool._tool_parameters == {}

def test_tool_decorator_custom_values():
    """Test tool decorator with custom values."""
    @tool(
        name="custom_name",
        description="Custom description",
        parameters={"param": "description"}
    )
    def sample_tool() -> str:
        return "test"

    assert sample_tool._is_tool is True
    assert sample_tool._tool_name == "custom_name"
    assert sample_tool._tool_description == "Custom description"
    assert sample_tool._tool_parameters == {"param": "description"}

def test_tool_decorator_result_wrapping():
    """Test that tool decorator properly wraps results."""
    @tool()
    def string_tool() -> str:
        return "test"

    @tool()
    def dict_tool() -> Dict[str, str]:
        return {"key": "value"}

    @tool()
    def list_tool() -> List[Dict[str, str]]:
        return [{"type": "text", "text": "test"}]

    # String result should be wrapped
    result = string_tool()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "test"

    # Dict result should be wrapped
    result = dict_tool()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "{'key': 'value'}"

    # List result should be returned as is
    result = list_tool()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "test"

def test_tool_decorator_preserves_function_metadata():
    """Test that tool decorator preserves function metadata."""
    @tool()
    def sample_tool(param1: str, param2: int = 0) -> str:
        """Sample tool docstring."""
        return f"{param1} {param2}"

    assert sample_tool.__name__ == "sample_tool"
    assert "Sample tool docstring" in sample_tool.__doc__
    # Check that the function signature is preserved
    import inspect
    sig = inspect.signature(sample_tool)
    assert list(sig.parameters.keys()) == ["param1", "param2"]
    assert sig.parameters["param1"].annotation == str
    assert sig.parameters["param2"].annotation == int
    assert sig.parameters["param2"].default == 0 