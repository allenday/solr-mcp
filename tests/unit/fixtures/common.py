"""Common fixtures and mock data for unit tests."""

from typing import List, Optional
from unittest.mock import Mock

import pytest

from solr_mcp.solr.interfaces import CollectionProvider, VectorSearchProvider

# Mock response data with various levels of detail
MOCK_RESPONSES = {
    "collections": ["collection1", "collection2"],
    "select": {"result-set": {"docs": [{"id": "1", "field": "value"}], "numFound": 1}},
    "vector": {
        "result-set": {
            "docs": [{"id": "1", "field": "value", "score": 0.95}],
            "numFound": 1,
        }
    },
    "semantic": {
        "result-set": {
            "docs": [{"id": "1", "field": "value", "score": 0.85}],
            "numFound": 1,
        }
    },
    "schema": {
        "schema": {
            "fields": [
                {
                    "name": "id",
                    "type": "string",
                    "multiValued": False,
                    "required": True,
                },
                {"name": "title", "type": "text_general", "multiValued": False},
                {"name": "content", "type": "text_general", "multiValued": False},
                {"name": "vector", "type": "knn_vector", "multiValued": False},
            ],
            "fieldTypes": [
                {"name": "string", "class": "solr.StrField", "sortMissingLast": True},
                {
                    "name": "text_general",
                    "class": "solr.TextField",
                    "positionIncrementGap": "100",
                },
                {
                    "name": "knn_vector",
                    "class": "solr.DenseVectorField",
                    "vectorDimension": 768,
                },
            ],
        }
    },
    "field_list": {
        "fields": [
            {
                "name": "id",
                "type": "string",
                "indexed": True,
                "stored": True,
                "docValues": True,
                "multiValued": False,
            },
            {
                "name": "_text_",
                "type": "text_general",
                "indexed": True,
                "stored": False,
                "docValues": False,
                "multiValued": True,
                "copies_from": ["title", "content"],
            },
        ]
    },
}


class MockCollectionProvider(CollectionProvider):
    """Mock implementation of CollectionProvider."""

    def __init__(self, collections=None):
        """Initialize with optional list of collections."""
        self.collections = (
            collections if collections is not None else MOCK_RESPONSES["collections"]
        )

    async def list_collections(self) -> List[str]:
        """Return mock list of collections."""
        return self.collections

    async def collection_exists(self, collection: str) -> bool:
        """Check if collection exists in mock list."""
        return collection in self.collections


class MockVectorProvider(VectorSearchProvider):
    """Mock vector provider for testing."""

    async def execute_vector_search(self, client, vector, top_k=10):
        """Mock vector search execution."""
        return {
            "response": {
                "docs": [
                    {"_docid_": "1", "score": 0.9, "_vector_distance_": 0.1},
                    {"_docid_": "2", "score": 0.8, "_vector_distance_": 0.2},
                    {"_docid_": "3", "score": 0.7, "_vector_distance_": 0.3},
                ],
                "numFound": 3,
                "start": 0,
            }
        }

    async def get_vector(self, text: str, model: Optional[str] = None) -> List[float]:
        """Mock text to vector conversion."""
        return [0.1, 0.2, 0.3]


@pytest.fixture
def valid_config_dict():
    """Valid configuration dictionary."""
    return {
        "solr_base_url": "http://localhost:8983/solr",
        "zookeeper_hosts": ["localhost:2181"],
        "connection_timeout": 10,
    }
