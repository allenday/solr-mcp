"""Base tool definitions and decorators."""

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator to mark a function as an MCP tool.

    Args:
        name: Tool name. Defaults to function name if not provided.
        description: Tool description. Defaults to function docstring if not provided.
        parameters: Tool parameters. Defaults to function parameters if not provided.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> List[Dict[str, str]]:
            result = func(*args, **kwargs)
            if not isinstance(result, list):
                result = [{"type": "text", "text": str(result)}]
            return result

        # Mark as tool
        wrapper._is_tool = True

        # Set tool metadata
        wrapper._tool_name = name or func.__name__
        wrapper._tool_description = description or func.__doc__ or ""
        wrapper._tool_parameters = parameters or {}

        return wrapper

    return decorator
