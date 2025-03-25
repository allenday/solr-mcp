"""Solr client and query fixtures for unit tests."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pysolr
import pytest

from solr_mcp.solr.client import SolrClient
from solr_mcp.solr.exceptions import ConnectionError, QueryError, SolrError
from solr_mcp.solr.interfaces import CollectionProvider
from solr_mcp.solr.query import QueryBuilder
from solr_mcp.solr.schema import FieldManager

from .common import MOCK_RESPONSES


@pytest.fixture
def mock_pysolr(request):
    """Parameterized mock for pysolr.Solr instance.

    Args:
        request: Pytest request object that can contain parameters:
            - query_type: Type of query to mock ("vector", "standard", "error")
    """
    mock = Mock(spec=pysolr.Solr)

    # Get parameters or use defaults
    query_type = getattr(request, "param", {}).get("query_type", "standard")

    if query_type == "error":
        mock.search.side_effect = pysolr.SolrError("Mock Solr error")
    else:

        def mock_search(*args, **kwargs):
            # Check if this is a vector/knn query
            if query_type == "vector" or (args and "{!knn" in args[0]):
                return {
                    "response": {
                        "docs": [
                            {"id": "1", "score": 0.95, "_vector_distance_": 0.05},
                            {"id": "2", "score": 0.85, "_vector_distance_": 0.15},
                        ],
                        "numFound": 2,
                        "maxScore": 0.95,
                    }
                }
            # Default response for regular queries
            return {"response": {"docs": [{"id": "1"}], "numFound": 1, "maxScore": 1.0}}

        mock.search.side_effect = mock_search

    return mock


@pytest.fixture
def mock_solr_instance(mock_pysolr):
    """Mock pysolr.Solr instance with patching."""
    with patch("pysolr.Solr", return_value=mock_pysolr):
        yield mock_pysolr


@pytest.fixture
def mock_collection_provider(request):
    """Parameterized mock for collection provider.

    Args:
        request: Pytest request object that can contain parameters:
            - collections: List of collections to return
            - error: Whether to simulate an error
    """
    # Get parameters or use defaults
    collections = getattr(request, "param", {}).get(
        "collections", MOCK_RESPONSES["collections"]
    )
    error = getattr(request, "param", {}).get("error", False)

    provider = Mock(spec=CollectionProvider)

    if error:
        provider.list_collections.side_effect = ConnectionError("Mock connection error")
    else:
        provider.list_collections.return_value = collections

    return provider


@pytest.fixture
def mock_field_manager(request):
    """Parameterized mock field manager.

    Args:
        request: Pytest request object that can contain parameters:
            - fields: Custom field data to return
            - error: Whether to simulate an error
    """
    # Get parameters or use defaults
    fields = getattr(request, "param", {}).get(
        "fields", MOCK_RESPONSES["field_list"]["fields"]
    )
    error = getattr(request, "param", {}).get("error", False)

    manager = MagicMock()

    if error:
        manager.get_collection_fields = Mock(
            side_effect=SolrError("Failed to retrieve fields")
        )
    else:
        manager.get_collection_fields = Mock(return_value={"fields": fields})

    return manager


@pytest.fixture
def mock_query_builder(request):
    """Parameterized mock QueryBuilder.

    Args:
        request: Pytest request object that can contain parameters:
            - collection: Collection name to return
            - fields: Fields to return
            - args: Query arguments to return
            - error: Whether to simulate an error
    """
    # Get parameters or use defaults
    collection = getattr(request, "param", {}).get("collection", "test_collection")
    fields = getattr(request, "param", {}).get("fields", ["id", "title"])
    args = getattr(request, "param", {}).get("args", {"limit": 10, "offset": 0})
    error = getattr(request, "param", {}).get("error", False)

    builder = Mock(spec=QueryBuilder)

    if error:
        builder.parse_and_validate_select.side_effect = QueryError("Invalid query")
    else:
        builder.parse_and_validate_select.return_value = (
            Mock(args=args),  # AST
            collection,  # Collection name
            fields,  # Fields
        )

        builder.build_vector_query = Mock(
            return_value={"fq": ["1", "2", "3"], "rows": args.get("limit", 10)}
        )

        parser = Mock()
        parser.preprocess_query = Mock(return_value="preprocessed query")
        builder.parser = parser

    return builder


@pytest.fixture
def mock_solr_client(request):
    """Parameterized mock SolrClient.

    Args:
        request: Pytest request object that can contain parameters:
            - error: Whether to simulate error responses
            - select_response: Custom select response
            - vector_response: Custom vector response
            - semantic_response: Custom semantic response
    """
    # Get parameters or use defaults
    error = getattr(request, "param", {}).get("error", False)
    select_response = getattr(request, "param", {}).get(
        "select_response", MOCK_RESPONSES["select"]
    )
    vector_response = getattr(request, "param", {}).get(
        "vector_response", MOCK_RESPONSES["vector"]
    )
    semantic_response = getattr(request, "param", {}).get(
        "semantic_response", MOCK_RESPONSES["semantic"]
    )
    collections = getattr(request, "param", {}).get(
        "collections", MOCK_RESPONSES["collections"]
    )
    fields = getattr(request, "param", {}).get(
        "fields", MOCK_RESPONSES["field_list"]["fields"]
    )

    client = Mock(spec=SolrClient)

    if error:
        client.execute_select_query = AsyncMock(side_effect=QueryError("Test error"))
        client.execute_vector_select_query = AsyncMock(
            side_effect=QueryError("Test error")
        )
        client.execute_semantic_select_query = AsyncMock(
            side_effect=QueryError("Test error")
        )
        client.list_collections = AsyncMock(side_effect=ConnectionError("Test error"))
        client.list_fields = AsyncMock(side_effect=SolrError("Test error"))
    else:
        client.execute_select_query = AsyncMock(return_value=select_response)
        client.execute_vector_select_query = AsyncMock(return_value=vector_response)
        client.execute_semantic_select_query = AsyncMock(return_value=semantic_response)
        client.list_collections = AsyncMock(return_value=collections)
        client.list_fields = AsyncMock(return_value=fields)

    return client


@pytest.fixture
def client(
    mock_config,
    mock_collection_provider,
    mock_field_manager,
    mock_vector_provider,
    mock_query_builder,
):
    """Create a SolrClient instance with mocked dependencies."""
    return SolrClient(
        config=mock_config,
        collection_provider=mock_collection_provider,
        field_manager=mock_field_manager,
        vector_provider=mock_vector_provider,
        query_builder=mock_query_builder,
    )


@pytest.fixture
def patch_module():
    """Factory fixture for patching modules temporarily.

    Returns a function that can be used to create context managers
    for patching different modules or objects.
    """

    def _patcher(target, **kwargs):
        return patch(target, **kwargs)

    return _patcher
