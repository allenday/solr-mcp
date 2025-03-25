"""Unit tests for SolrClient."""

from unittest.mock import Mock, patch

import pytest

from solr_mcp.solr.client import SolrClient
from solr_mcp.solr.interfaces import CollectionProvider, VectorSearchProvider

from .conftest import MOCK_RESPONSES, MockCollectionProvider, MockVectorProvider


class TestSolrClient:
    """Test cases for SolrClient."""

    def test_init_with_defaults(self, mock_config, mock_field_manager, mock_ollama):
        """Test initialization with default dependencies."""
        client = SolrClient(
            config=mock_config,
            field_manager=mock_field_manager,
            vector_provider=mock_ollama,
        )
        assert client.config == mock_config
        assert isinstance(client.collection_provider, CollectionProvider)
        assert client.field_manager == mock_field_manager
        assert client.vector_provider == mock_ollama

    def test_init_with_custom_providers(self, mock_config, mock_field_manager):
        """Test initialization with custom providers."""
        mock_collection_provider = MockCollectionProvider()
        mock_vector_provider = MockVectorProvider()
        mock_solr = Mock()  # Create a simple mock

        client = SolrClient(
            config=mock_config,
            collection_provider=mock_collection_provider,
            solr_client=mock_solr,
            field_manager=mock_field_manager,
            vector_provider=mock_vector_provider,
        )
        assert client.config == mock_config
        assert client.collection_provider == mock_collection_provider
        assert client.field_manager == mock_field_manager
        assert client.vector_provider == mock_vector_provider

    @pytest.mark.asyncio
    @pytest.mark.parametrize("collection", ["collection1", "test_collection"])
    async def test_execute_select_query_success(
        self, mock_config, mock_field_manager, collection
    ):
        """Test successful SQL query execution with different collections."""
        # Create a mock for the query builder
        mock_query_builder = Mock()
        mock_query_builder.parser = Mock()
        mock_query_builder.parser.preprocess_query = Mock(
            return_value=f"SELECT * FROM {collection}"
        )
        mock_query_builder.parse_and_validate_select = Mock(
            return_value=(
                Mock(),  # AST
                collection,  # Collection name
                ["id", "title"],  # Fields
            )
        )

        # Create a mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result-set": {"docs": [{"id": "1", "field": "value"}], "numFound": 1}
        }

        # Create client with dependencies and patch requests.post
        with patch("requests.post", return_value=mock_response):
            client = SolrClient(
                config=mock_config,
                field_manager=mock_field_manager,
                query_builder=mock_query_builder,
            )

            # Execute query
            result = await client.execute_select_query(f"SELECT * FROM {collection}")

            # Verify result structure
            assert "result-set" in result
            assert "docs" in result["result-set"]
            assert result["result-set"]["docs"][0]["id"] == "1"
