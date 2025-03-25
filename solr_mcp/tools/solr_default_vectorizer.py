"""Tool for getting information about the default vector provider."""

import re
from typing import Any, Dict
from urllib.parse import urlparse

from solr_mcp.tools.tool_decorator import tool
from solr_mcp.vector_provider.constants import DEFAULT_OLLAMA_CONFIG, MODEL_DIMENSIONS


@tool()
async def get_default_text_vectorizer(mcp) -> Dict[str, Any]:
    """Get information about the default vector provider used for semantic search.

    Returns information about the default vector provider configuration used for semantic search,
    including the model name, vector dimensionality, host, and port.

    This information is useful for ensuring that your vector fields in Solr have
    the correct dimensionality to match the vector provider model.

    Returns:
        Dictionary containing:
        - vector_provider_model: The name of the default vector provider model
        - vector_provider_dimension: The dimensionality of vectors produced by this model
        - vector_provider_host: The host of the vector provider service
        - vector_provider_port: The port of the vector provider service
        - vector_provider_url: The full URL of the vector provider service
    """
    if hasattr(mcp, "solr_client") and hasattr(mcp.solr_client, "vector_manager"):
        vector_manager = mcp.solr_client.vector_manager
        model_name = vector_manager.client.model
        dimension = MODEL_DIMENSIONS.get(model_name, 768)
        base_url = vector_manager.client.base_url
    else:
        # Fall back to defaults
        model_name = DEFAULT_OLLAMA_CONFIG["model"]
        dimension = MODEL_DIMENSIONS.get(model_name, 768)
        base_url = DEFAULT_OLLAMA_CONFIG["base_url"]

    # Parse URL to extract host and port
    parsed_url = urlparse(base_url)
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 11434  # Default Ollama port

    # Format as "model@host:port" for easy use with vector_provider parameter
    formatted_spec = f"{model_name}@{host}:{port}"

    return {
        "vector_provider_model": model_name,
        "vector_provider_dimension": dimension,
        "vector_provider_host": host,
        "vector_provider_port": port,
        "vector_provider_url": base_url,
        "vector_provider_spec": formatted_spec,
    }
