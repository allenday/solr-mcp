"""Tool definitions for Solr MCP server."""

import inspect
import sys

from .tool_decorator import tool, get_schema
from .solr_list_collections import execute_list_collections
from .solr_list_fields import execute_list_fields
from .solr_select import execute_select_query
from .solr_vector_select import execute_vector_select_query
from .solr_semantic_select import execute_semantic_select_query
from .solr_default_vectorizer import get_default_text_vectorizer

__all__ = [
    "execute_list_collections",
    "execute_list_fields",
    "execute_select_query",
    "execute_vector_select_query",
    "execute_semantic_select_query",
    "get_default_text_vectorizer",
]

TOOLS_DEFINITION = [
    obj
    for name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isfunction(obj) and hasattr(obj, "_is_tool") and obj._is_tool
] 