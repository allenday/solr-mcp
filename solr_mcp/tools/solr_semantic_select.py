"""Tool for executing semantic search queries against Solr collections."""

from typing import Dict, List, Optional

from solr_mcp.tools.tool_decorator import tool
from solr_mcp.vector_provider.constants import DEFAULT_OLLAMA_CONFIG


@tool()
async def execute_semantic_select_query(
    mcp, query: str, text: str, field: Optional[str] = None, vector_provider: str = ""
) -> Dict:
    """Execute semantic search queries against Solr collections.

    Extends solr_select tool with semantic search capabilities.

    Additional Parameters:
    - text: Natural language text that is converted to vector, which will be used to match against other vector fields
    - field: Name of the vector field to search against (optional, will auto-detect if not specified)
    - vector_provider: Vector provider specification in format "model@host:port" (e.g., "nomic-embed-text@localhost:11434")
       If not specified, the default vector provider will be used

    The query results will be ranked based on semantic similarity to the provided text. Therefore, ORDER BY is not allowed.

    Collection/Field Rules:
    - Vector field must be a dense_vector or knn_vector field type
    - The specified field must exist in the collection schema
    - The vector provider's dimensionality must match the dimensionality of the vector field

    Supported Features:
    - All standard SELECT query features except ORDER BY
      - Results are ordered by semantic similarity
    - Hybrid search combining keyword (SQL WHERE clauses) and vector distance (text parameter)

    Args:
        mcp: SolrMCPServer instance
        query: SQL query to execute
        text: Search text to convert to vector
        field: Name of the vector field to search against (optional, auto-detected if not specified)
        vector_provider: Optional vector provider specification "model@host:port"

    Returns:
        Query results
    """
    solr_client = mcp.solr_client

    # Configure vector provider from parameter string
    vector_provider_config = {}

    if vector_provider:
        # Parse "model@host:port" format
        model_part = vector_provider
        host_port_part = None

        if "@" in vector_provider:
            parts = vector_provider.split("@", 1)
            model_part = parts[0]
            host_port_part = parts[1]

        # Set model if specified
        if model_part:
            vector_provider_config["model"] = model_part

        # Set host:port if specified
        if host_port_part:
            if ":" in host_port_part:
                host, port_str = host_port_part.split(":", 1)
                try:
                    port = int(port_str)
                    vector_provider_config["base_url"] = f"http://{host}:{port}"
                except ValueError:
                    # If port is not a valid integer, use the host with default port
                    vector_provider_config["base_url"] = f"http://{host_port_part}"
            else:
                # Only host specified, use default port
                vector_provider_config["base_url"] = f"http://{host_port_part}:11434"

    return await solr_client.execute_semantic_select_query(
        query, text, field, vector_provider_config
    )
