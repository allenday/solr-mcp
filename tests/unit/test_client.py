"""Unit tests for SolrClient."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from solr_mcp.solr.client import SolrClient
from solr_mcp.solr.interfaces import CollectionProvider, VectorSearchProvider
from .conftest import (
    MOCK_SELECT_RESPONSE,
    MOCK_VECTOR_RESPONSE,
    MOCK_SEMANTIC_RESPONSE,
    MockCollectionProvider,
    MockVectorProvider
)

class TestSolrClient:
    """Test cases for SolrClient."""

    def test_init_with_defaults(self, mock_config, mock_solr_instance, mock_field_manager, mock_ollama):
        """Test initialization with default dependencies."""
        client = SolrClient(
            config=mock_config,
            field_manager=mock_field_manager,
            vector_provider=mock_ollama
        )
        assert client.config == mock_config
        assert isinstance(client.collection_provider, CollectionProvider)
        assert client.field_manager == mock_field_manager
        assert client.vector_provider == mock_ollama

    def test_init_with_custom_providers(self, mock_config, mock_solr_instance, mock_field_manager):
        """Test initialization with custom providers."""
        mock_collection_provider = MockCollectionProvider()
        mock_vector_provider = MockVectorProvider()
        
        client = SolrClient(
            config=mock_config,
            collection_provider=mock_collection_provider,
            solr_client=mock_solr_instance,
            field_manager=mock_field_manager,
            vector_provider=mock_vector_provider
        )
        assert client.config == mock_config
        assert client.collection_provider == mock_collection_provider
        assert client.field_manager == mock_field_manager
        assert client.vector_provider == mock_vector_provider


    @pytest.mark.asyncio
    async def test_execute_select_query_success(self, mock_config, mock_pysolr, mock_solr_requests, mock_field_manager):
        """Test successful SQL query execution."""
        # Mock the response from requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_SELECT_RESPONSE
        mock_solr_requests.post.return_value = mock_response

        client = SolrClient(
            config=mock_config,
            field_manager=mock_field_manager
        )
        result = await client.execute_select_query("SELECT * FROM collection1")
        
        assert "result-set" in result
        assert "docs" in result["result-set"]
        assert result["result-set"]["docs"][0]["id"] == "1"

        # Verify the request was made correctly
        mock_solr_requests.post.assert_called_once_with(
            f"{mock_config.solr_base_url}/collection1/sql?aggregationMode=facet",
            data={"stmt": "SELECT * FROM collection1"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

