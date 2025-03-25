"""Vector search fixtures for unit tests."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from solr_mcp.solr.interfaces import VectorSearchProvider
from solr_mcp.solr.vector.manager import VectorManager
from solr_mcp.vector_provider.clients.ollama import OllamaVectorProvider


@pytest.fixture
def mock_ollama(request):
    """Parameterized mock for Ollama client.

    Args:
        request: Pytest request object that can contain parameters:
            - vector_dim: Dimension of returned vectors
            - error: Whether to simulate an error
    """
    # Get parameters or use defaults
    vector_dim = getattr(request, "param", {}).get("vector_dim", 3)
    error = getattr(request, "param", {}).get("error", False)

    provider = Mock(spec=OllamaVectorProvider)

    if error:
        provider.get_vector.side_effect = Exception("Ollama API error")
    else:
        provider.get_vector.return_value = [0.1] * vector_dim

    return provider


@pytest.fixture
def mock_vector_provider(request):
    """Parameterized mock for vector provider.

    Args:
        request: Pytest request object that can contain parameters:
            - vector_dim: Dimension of returned vectors
            - error: Whether to simulate an error
    """
    # Get parameters or use defaults
    vector_dim = getattr(request, "param", {}).get("vector_dim", 768)
    error = getattr(request, "param", {}).get("error", False)

    provider = Mock(spec=VectorSearchProvider)

    if error:
        provider.get_vector.side_effect = Exception("Vector API error")
    else:
        provider.get_vector.return_value = [0.1] * vector_dim

    return provider


@pytest.fixture
def mock_vector_manager(request):
    """Parameterized mock VectorManager.

    Args:
        request: Pytest request object that can contain parameters:
            - vector_dim: Dimension of returned vectors
            - error: Whether to simulate an error
    """
    # Get parameters or use defaults
    vector_dim = getattr(request, "param", {}).get("vector_dim", 3)
    error = getattr(request, "param", {}).get("error", False)

    manager = Mock(spec=VectorManager)

    if error:
        manager.get_vector.side_effect = Exception("Vector generation error")
    else:
        manager.get_vector.return_value = [0.1] * vector_dim

    return manager


@pytest.fixture
def mock_ollama_response(request):
    """Parameterized mock Ollama API response.

    Args:
        request: Pytest request object that can contain parameters:
            - vector_dim: Dimension of returned vectors
            - model: Model name to include in response
    """
    # Get parameters or use defaults
    vector_dim = getattr(request, "param", {}).get("vector_dim", 5)
    model = getattr(request, "param", {}).get("model", "nomic-embed-text")

    return {"embedding": [0.1] * vector_dim, "model": model}
