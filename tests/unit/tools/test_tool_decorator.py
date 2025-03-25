"""Tests for tool decorator functionality."""

from typing import List, Literal, Optional, Union, Any
import pytest

from solr_mcp.tools.tool_decorator import tool, get_schema


def test_tool_name_conversion():
    """Test tool name conversion from function name."""
    @tool()
    async def execute_list_collections():
        """List collections."""
        pass

    @tool()
    async def execute_select_query():
        """Execute select query."""
        pass

    @tool()
    async def execute_vector_select_query():
        """Execute vector select query."""
        pass

    assert hasattr(execute_list_collections, "_tool_name")
    assert execute_list_collections._tool_name == "solr_list_collections"
    assert execute_select_query._tool_name == "solr_select"
    assert execute_vector_select_query._tool_name == "solr_vector_select"


@pytest.mark.asyncio
async def test_tool_error_handling():
    """Test error handling in tool wrapper."""
    @tool()
    async def failing_tool():
        """Tool that raises an exception."""
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        await failing_tool()


def test_get_schema_validation():
    """Test schema validation for non-tool functions."""
    def regular_function():
        pass

    with pytest.raises(ValueError, match="is not a tool"):
        get_schema(regular_function)


def test_get_schema_no_params():
    """Test schema generation for function with no parameters."""
    @tool()
    async def no_params_tool():
        """Tool with no parameters."""
        pass

    with pytest.raises(ValueError, match="must have at least one parameter"):
        get_schema(no_params_tool)


def test_get_schema_basic_types():
    """Test schema generation for basic parameter types."""
    @tool()
    async def basic_types_tool(
        str_param: str,
        int_param: int,
        float_param: float,
        bool_param: bool,
        optional_str: Optional[str] = None,
        default_int: int = 42
    ):
        """Test tool with basic types.
        
        Args:
            str_param: String parameter
            int_param: Integer parameter
            float_param: Float parameter
            bool_param: Boolean parameter
            optional_str: Optional string parameter
            default_int: Integer parameter with default
        """
        pass

    schema = get_schema(basic_types_tool)
    properties = schema["inputSchema"]["properties"]
    required = schema["inputSchema"]["required"]

    assert properties["str_param"]["type"] == "string"
    assert properties["int_param"]["type"] == "integer"
    assert properties["float_param"]["type"] == "number"
    assert properties["bool_param"]["type"] == "boolean"
    assert properties["optional_str"]["type"] == "string"
    assert properties["default_int"]["type"] == "integer"

    assert "str_param" in required
    assert "int_param" in required
    assert "float_param" in required
    assert "bool_param" in required
    assert "optional_str" not in required
    assert "default_int" not in required


def test_get_schema_complex_types():
    """Test schema generation for complex parameter types."""
    @tool()
    async def complex_types_tool(
        str_list: List[str],
        mode: Literal["a", "b", "c"],
        optional_mode: Optional[Literal["x", "y", "z"]] = None,
        union_type: Union[str, int] = "default"
    ):
        """Test tool with complex types.
        
        Args:
            str_list: List of strings
            mode: Mode selection
            optional_mode: Optional mode selection
            union_type: Union of string and integer
        """
        pass

    schema = get_schema(complex_types_tool)
    properties = schema["inputSchema"]["properties"]
    required = schema["inputSchema"]["required"]

    assert properties["str_list"]["type"] == "array"
    assert properties["str_list"]["items"]["type"] == "string"
    
    assert properties["mode"]["type"] == "string"
    assert set(properties["mode"]["enum"]) == {"a", "b", "c"}
    
    assert properties["optional_mode"]["type"] == "string"
    assert set(properties["optional_mode"]["enum"]) == {"x", "y", "z"}
    
    assert properties["union_type"]["type"] == "string"

    assert "str_list" in required
    assert "mode" in required
    assert "optional_mode" not in required
    assert "union_type" not in required


def test_get_schema_docstring_parsing():
    """Test docstring parsing in schema generation."""
    @tool()
    async def documented_tool(param1: str, param2: int):
        """Tool with detailed documentation.

        This is a multiline description
        that should be captured.

        Args:
            param1: First parameter with multiline description
            param2: Second parameter with multiple lines

        Returns:
            Some result

        Examples:
            Some examples that should not be in description
        """
        pass

    schema = get_schema(documented_tool)
    
    assert "Tool with detailed documentation" in schema["description"]
    assert "This is a multiline description" in schema["description"]
    assert "Returns:" not in schema["description"]
    assert "Examples:" not in schema["description"]
    
    properties = schema["inputSchema"]["properties"]
    assert "First parameter with multiline description" == properties["param1"]["description"]
    assert "Second parameter with multiple lines" == properties["param2"]["description"]


def test_get_schema_no_docstring():
    """Test schema generation for function without docstring."""
    @tool()
    async def no_doc_tool(param: str):
        pass

    schema = get_schema(no_doc_tool)
    assert schema["description"] == ""
    assert schema["inputSchema"]["properties"]["param"]["description"] == "param parameter"


def test_get_schema_edge_cases():
    """Test schema generation for edge cases in docstring parsing."""
    @tool()
    async def edge_case_tool(param1: Any, param2: int, param3: float):
        """Tool with edge case documentation.

        Args:
            param1: First parameter
            param2: Second parameter
            param3: Third parameter

        Args:
            Duplicate args section should be ignored

        Returns:
            Some value
            More return info

        Examples:
            Example 1
            Example 2
        """
        pass

    schema = get_schema(edge_case_tool)
    properties = schema["inputSchema"]["properties"]

    # Test that parameter descriptions are captured correctly
    assert "First parameter" == properties["param1"]["description"]
    assert "Second parameter" == properties["param2"]["description"]
    assert "Third parameter" == properties["param3"]["description"]
    
    # Test that empty lines and sections after Args are properly handled
    assert "Tool with edge case documentation" in schema["description"]
    assert "Duplicate args section" not in schema["description"]
    assert "Returns:" not in schema["description"]
    assert "Examples:" not in schema["description"]
    
    # Test that Any type is handled correctly
    assert properties["param1"]["type"] == "string" 