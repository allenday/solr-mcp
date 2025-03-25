"""Tests for Ollama vector provider."""

import pytest
from unittest.mock import Mock, patch
import requests

from solr_mcp.vector_provider.clients.ollama import OllamaVectorProvider
from solr_mcp.vector_provider.exceptions import VectorGenerationError, VectorConnectionError
from solr_mcp.vector_provider.constants import DEFAULT_OLLAMA_CONFIG, MODEL_DIMENSIONS

@pytest.fixture
def mock_response():
    """Mock successful response from Ollama API."""
    mock = Mock()
    mock.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock.raise_for_status.return_value = None
    return mock

@pytest.fixture
def provider():
    """Create OllamaVectorProvider instance with default config."""
    return OllamaVectorProvider()

def test_init_with_defaults():
    """Test initialization with default values."""
    provider = OllamaVectorProvider()
    assert provider.model == DEFAULT_OLLAMA_CONFIG["model"]
    assert provider.base_url == DEFAULT_OLLAMA_CONFIG["base_url"]
    assert provider.timeout == DEFAULT_OLLAMA_CONFIG["timeout"]
    assert provider.retries == DEFAULT_OLLAMA_CONFIG["retries"]

def test_init_with_custom_config():
    """Test initialization with custom configuration."""
    custom_config = {
        "model": "custom-model",
        "base_url": "http://custom:8080",
        "timeout": 60,
        "retries": 5
    }
    provider = OllamaVectorProvider(**custom_config)
    assert provider.model == custom_config["model"]
    assert provider.base_url == custom_config["base_url"]
    assert provider.timeout == custom_config["timeout"]
    assert provider.retries == custom_config["retries"]

@pytest.mark.asyncio
async def test_get_embedding_success(provider, mock_response):
    """Test successful embedding generation."""
    with patch('requests.post', return_value=mock_response):
        result = await provider.get_embedding("test text")
        assert result == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_get_embedding_retry_success(provider, mock_response):
    """Test successful retry after initial failure."""
    fail_response = Mock()
    fail_response.raise_for_status.side_effect = requests.exceptions.RequestException("Test error")
    
    with patch('requests.post') as mock_post:
        mock_post.side_effect = [fail_response, mock_response]
        result = await provider.get_embedding("test text")
        assert result == [0.1, 0.2, 0.3]
        assert mock_post.call_count == 2

@pytest.mark.asyncio
async def test_get_embedding_all_retries_fail(provider):
    """Test when all retry attempts fail."""
    fail_response = Mock()
    fail_response.raise_for_status.side_effect = requests.exceptions.RequestException("Test error")
    
    with patch('requests.post', return_value=fail_response):
        with pytest.raises(Exception) as exc_info:
            await provider.get_embedding("test text")
        assert "Failed to get embeddings after" in str(exc_info.value)

@pytest.mark.asyncio
async def test_execute_vector_search_success(provider):
    """Test successful vector search execution."""
    mock_client = Mock()
    mock_client.search.return_value = {"response": {"docs": []}}
    vector = [0.1, 0.2, 0.3]
    
    result = await provider.execute_vector_search(mock_client, vector, top_k=5)
    assert result == {"response": {"docs": []}}
    
    # Verify search was called with correct KNN query
    mock_client.search.assert_called_once()
    call_args = mock_client.search.call_args[1]
    assert "knn" in call_args
    assert "topK=5" in call_args["knn"]
    assert "0.1,0.2,0.3" in call_args["knn"]

@pytest.mark.asyncio
async def test_execute_vector_search_error(provider):
    """Test vector search with error."""
    mock_client = Mock()
    mock_client.search.side_effect = Exception("Search failed")
    vector = [0.1, 0.2, 0.3]
    
    with pytest.raises(Exception) as exc_info:
        await provider.execute_vector_search(mock_client, vector)
    assert "Vector search failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_get_embeddings_batch(provider, mock_response):
    """Test getting embeddings for multiple texts."""
    with patch('requests.post', return_value=mock_response):
        texts = ["text1", "text2"]
        result = await provider.get_embeddings(texts)
        assert len(result) == 2
        assert all(v == [0.1, 0.2, 0.3] for v in result)

def test_vector_dimension(provider):
    """Test vector_dimension property returns correct value."""
    assert provider.vector_dimension == MODEL_DIMENSIONS[provider.model]
    
    # Test with custom model
    custom_provider = OllamaVectorProvider(model="custom-model")
    assert custom_provider.vector_dimension == 768  # Default dimension

def test_model_name(provider):
    """Test model_name property returns correct value."""
    assert provider.model_name == provider.model 