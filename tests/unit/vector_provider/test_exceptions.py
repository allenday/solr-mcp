"""Tests for vector provider exceptions."""

from solr_mcp.vector_provider.exceptions import (
    VectorError,
    VectorGenerationError,
    VectorConfigError,
    VectorConnectionError
)

def test_vector_error():
    """Test base VectorError exception."""
    error = VectorError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, Exception)

def test_vector_generation_error():
    """Test VectorGenerationError exception."""
    error = VectorGenerationError("Generation failed")
    assert str(error) == "Generation failed"
    assert isinstance(error, VectorError)
    assert isinstance(error, Exception)

def test_vector_config_error():
    """Test VectorConfigError exception."""
    error = VectorConfigError("Invalid config")
    assert str(error) == "Invalid config"
    assert isinstance(error, VectorError)
    assert isinstance(error, Exception)

def test_vector_connection_error():
    """Test VectorConnectionError exception."""
    error = VectorConnectionError("Connection failed")
    assert str(error) == "Connection failed"
    assert isinstance(error, VectorError)
    assert isinstance(error, Exception)

def test_error_inheritance():
    """Test exception inheritance hierarchy."""
    assert issubclass(VectorGenerationError, VectorError)
    assert issubclass(VectorConfigError, VectorError)
    assert issubclass(VectorConnectionError, VectorError)
    assert issubclass(VectorError, Exception) 