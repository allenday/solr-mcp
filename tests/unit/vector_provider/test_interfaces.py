"""Tests for vector provider interfaces."""

import pytest
from typing import List

from solr_mcp.vector_provider.interfaces import VectorProvider
from solr_mcp.vector_provider.exceptions import VectorGenerationError, VectorConnectionError, VectorConfigError

class MockVectorProvider(VectorProvider):
    """Mock implementation of VectorProvider for testing."""
    
    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        self._model = "mock-model"
        
    async def get_vector(self, text: str) -> List[float]:
        if text == "error":
            raise VectorGenerationError("Test error")
        return [0.1] * self._dimension
        
    async def get_vectors(self, texts: List[str]) -> List[List[float]]:
        if any(t == "error" for t in texts):
            raise VectorGenerationError("Test error")
        return [[0.1] * self._dimension for _ in texts]
        
    @property
    def vector_dimension(self) -> int:
        return self._dimension
        
    @property
    def model_name(self) -> str:
        return self._model

def test_vector_provider_is_abstract():
    """Test that VectorProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        VectorProvider()

def test_vector_provider_requires_methods():
    """Test that implementing class must define all abstract methods."""
    class IncompleteProvider(VectorProvider):
        pass
        
    with pytest.raises(TypeError):
        IncompleteProvider()

@pytest.mark.asyncio
async def test_mock_provider_get_vector():
    """Test get_vector implementation."""
    provider = MockVectorProvider()
    result = await provider.get_vector("test")
    assert len(result) == 768
    assert all(x == 0.1 for x in result)

@pytest.mark.asyncio
async def test_mock_provider_get_vector_error():
    """Test get_vector error handling."""
    provider = MockVectorProvider()
    with pytest.raises(VectorGenerationError):
        await provider.get_vector("error")

@pytest.mark.asyncio
async def test_mock_provider_get_vectors():
    """Test get_vectors implementation."""
    provider = MockVectorProvider()
    texts = ["test1", "test2"]
    result = await provider.get_vectors(texts)
    assert len(result) == 2
    assert all(len(v) == 768 for v in result)
    assert all(all(x == 0.1 for x in v) for v in result)

@pytest.mark.asyncio
async def test_mock_provider_get_vectors_error():
    """Test get_vectors error handling."""
    provider = MockVectorProvider()
    with pytest.raises(VectorGenerationError):
        await provider.get_vectors(["test", "error"])

def test_mock_provider_vector_dimension():
    """Test vector_dimension property."""
    provider = MockVectorProvider(dimension=512)
    assert provider.vector_dimension == 512

def test_mock_provider_model_name():
    """Test model_name property."""
    provider = MockVectorProvider()
    assert provider.model_name == "mock-model" 