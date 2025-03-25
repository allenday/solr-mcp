"""Unit tests for Solr client interfaces."""

import pytest
from abc import ABC
from typing import List, Dict, Any, Optional

from solr_mcp.solr.interfaces import CollectionProvider, VectorSearchProvider

def test_collection_provider_is_abstract():
    """Test that CollectionProvider is an abstract base class."""
    assert issubclass(CollectionProvider, ABC)
    assert CollectionProvider.__abstractmethods__ == {"list_collections"}

def test_collection_provider_cannot_instantiate():
    """Test that CollectionProvider cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        CollectionProvider()
    assert "Can't instantiate abstract class CollectionProvider with abstract method list_collections" == str(exc_info.value)

def test_collection_provider_requires_list_collections():
    """Test that implementations must provide list_collections method."""
    class IncompleteProvider(CollectionProvider):
        pass

    with pytest.raises(TypeError) as exc_info:
        IncompleteProvider()
    assert "Can't instantiate abstract class IncompleteProvider with abstract method list_collections" == str(exc_info.value)

def test_collection_provider_implementation():
    """Test that a complete implementation can be instantiated."""
    class ValidProvider(CollectionProvider):
        def list_collections(self) -> List[str]:
            return ["collection1"]

    provider = ValidProvider()
    assert isinstance(provider, CollectionProvider)
    assert provider.list_collections() == ["collection1"]

def test_vector_search_provider_is_abstract():
    """Test that VectorSearchProvider is an abstract base class."""
    assert issubclass(VectorSearchProvider, ABC)
    assert VectorSearchProvider.__abstractmethods__ == {"execute_vector_search", "get_vector"}

def test_vector_search_provider_cannot_instantiate():
    """Test that VectorSearchProvider cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        VectorSearchProvider()
    assert "abstract methods" in str(exc_info.value)
    assert "execute_vector_search" in str(exc_info.value)
    assert "get_vector" in str(exc_info.value)

def test_vector_search_provider_requires_all_methods():
    """Test that implementations must provide all required methods."""
    class IncompleteProvider(VectorSearchProvider):
        def execute_vector_search(
            self,
            client: Any,
            vector: List[float],
            field: str,
            top_k: Optional[int] = None
        ) -> Dict[str, Any]:
            return {"response": {"docs": []}}

    with pytest.raises(TypeError) as exc_info:
        IncompleteProvider()
    assert "Can't instantiate abstract class IncompleteProvider with abstract method get_vector" == str(exc_info.value)

def test_vector_search_provider_implementation():
    """Test that a complete implementation can be instantiated."""
    class ValidProvider(VectorSearchProvider):
        def execute_vector_search(
            self,
            client: Any,
            vector: List[float],
            field: str,
            top_k: Optional[int] = None
        ) -> Dict[str, Any]:
            return {"response": {"docs": []}}

        async def get_vector(self, text: str) -> List[float]:
            return [0.1, 0.2, 0.3]

    provider = ValidProvider()
    assert isinstance(provider, VectorSearchProvider)
    assert provider.execute_vector_search(None, [0.1], "vector_field") == {"response": {"docs": []}}

@pytest.mark.asyncio
async def test_vector_search_provider_async_method():
    """Test that async get_vector method works correctly."""
    class ValidProvider(VectorSearchProvider):
        def execute_vector_search(
            self,
            client: Any,
            vector: List[float],
            field: str,
            top_k: Optional[int] = None
        ) -> Dict[str, Any]:
            return {"response": {"docs": []}}

        async def get_vector(self, text: str) -> List[float]:
            return [0.1, 0.2, 0.3]

    provider = ValidProvider()
    result = await provider.get_vector("test")
    assert result == [0.1, 0.2, 0.3] 