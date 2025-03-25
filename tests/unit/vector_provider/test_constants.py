"""Tests for vector provider constants."""

from solr_mcp.vector_provider.constants import (
    DEFAULT_OLLAMA_CONFIG,
    ENV_OLLAMA_BASE_URL,
    ENV_OLLAMA_MODEL,
    MODEL_DIMENSIONS,
    OLLAMA_EMBEDDINGS_PATH,
)


def test_default_ollama_config():
    """Test default Ollama configuration values."""
    assert isinstance(DEFAULT_OLLAMA_CONFIG, dict)
    assert "base_url" in DEFAULT_OLLAMA_CONFIG
    assert "model" in DEFAULT_OLLAMA_CONFIG
    assert "timeout" in DEFAULT_OLLAMA_CONFIG
    assert "retries" in DEFAULT_OLLAMA_CONFIG

    assert DEFAULT_OLLAMA_CONFIG["base_url"] == "http://localhost:11434"
    assert DEFAULT_OLLAMA_CONFIG["model"] == "nomic-embed-text"
    assert DEFAULT_OLLAMA_CONFIG["timeout"] == 30
    assert DEFAULT_OLLAMA_CONFIG["retries"] == 3


def test_environment_variables():
    """Test environment variable names."""
    assert ENV_OLLAMA_BASE_URL == "OLLAMA_BASE_URL"
    assert ENV_OLLAMA_MODEL == "OLLAMA_MODEL"


def test_api_endpoints():
    """Test API endpoint paths."""
    assert OLLAMA_EMBEDDINGS_PATH == "/api/embeddings"


def test_model_dimensions():
    """Test model dimension mappings."""
    assert isinstance(MODEL_DIMENSIONS, dict)
    assert "nomic-embed-text" in MODEL_DIMENSIONS
    assert MODEL_DIMENSIONS["nomic-embed-text"] == 768  # 768-dimensional embeddings
