"""Tests for SolrClient."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import requests
import pysolr
import aiohttp
from aiohttp import test_utils
import asyncio

from solr_mcp.solr.client import SolrClient
from solr_mcp.solr.exceptions import (
    SolrError, ConnectionError, QueryError,
    DocValuesError, SQLParseError, SQLExecutionError
)

@pytest.mark.asyncio
async def test_init_with_defaults(mock_config):
    """Test initialization with only config."""
    client = SolrClient(config=mock_config)
    assert client.config == mock_config

@pytest.mark.asyncio
async def test_init_with_custom_providers(client, mock_config, mock_collection_provider,
                                        mock_field_manager, mock_vector_provider, mock_query_builder):
    """Test initialization with custom providers."""
    assert client.config == mock_config
    assert client.collection_provider == mock_collection_provider
    assert client.field_manager == mock_field_manager
    assert client.vector_provider == mock_vector_provider
    assert client.query_builder == mock_query_builder

@pytest.mark.asyncio
async def test_get_or_create_client_with_collection(client):
    """Test getting Solr client with specific collection."""
    solr_client = await client._get_or_create_client("test_collection")
    assert solr_client is not None

@pytest.mark.asyncio
async def test_get_or_create_client_with_different_collection(client):
    """Test getting Solr client with a different collection."""
    solr_client = await client._get_or_create_client("another_collection")
    assert solr_client is not None

@pytest.mark.asyncio
async def test_get_or_create_client_no_collection(mock_config):
    """Test error when no collection specified."""
    client = SolrClient(config=mock_config)
    with pytest.raises(SolrError):
        await client._get_or_create_client(None)

@pytest.mark.asyncio
async def test_list_collections_success(client):
    """Test successful collection listing."""
    # Mock the collection provider's list_collections method
    expected_collections = ["test_collection"]
    client.collection_provider.list_collections = AsyncMock(return_value=expected_collections)
    
    # Test the method
    result = await client.list_collections()
    assert result == expected_collections
    
    # Verify the collection provider was called
    client.collection_provider.list_collections.assert_called_once()

@pytest.mark.asyncio
async def test_list_fields_schema_error(client):
    """Test schema error handling in list_fields."""
    # Mock field_manager.list_fields to raise an error
    client.field_manager.list_fields = AsyncMock(side_effect=SolrError("Schema error"))
    
    # Test that the error is propagated
    with pytest.raises(SolrError):
        await client.list_fields("test_collection")

@pytest.mark.asyncio
async def test_execute_select_query_success(client):
    """Test successful SQL query execution."""
    # Mock parser.preprocess_query
    client.query_builder.parser.preprocess_query = Mock(return_value="SELECT * FROM test_collection")
    
    # Mock the parse_and_validate_select
    client.query_builder.parse_and_validate_select = Mock(return_value=(Mock(), "test_collection", None))
    
    # Mock the query executor
    expected_result = {
        "result-set": {
            "docs": [{"id": "1", "title": "Test"}],
            "numFound": 1
        }
    }
    client.query_executor.execute_select_query = AsyncMock(return_value=expected_result)
    
    # Execute the query
    result = await client.execute_select_query("SELECT * FROM test_collection")
    
    # Verify the result
    assert result == expected_result
    client.query_executor.execute_select_query.assert_called_once_with(
        query="SELECT * FROM test_collection",
        collection="test_collection"
    )

@pytest.mark.asyncio
async def test_execute_select_query_docvalues_error(client):
    """Test SQL query with DocValues error."""
    # Mock parser.preprocess_query
    client.query_builder.parser.preprocess_query = Mock(return_value="SELECT * FROM test_collection")
    
    # Mock the parse_and_validate_select
    client.query_builder.parse_and_validate_select = Mock(return_value=(Mock(), "test_collection", None))
    
    # Mock the query executor to raise a DocValuesError
    client.query_executor.execute_select_query = AsyncMock(side_effect=DocValuesError("must have DocValues to use this feature", 10))
    
    # Execute the query and verify the error
    with pytest.raises(DocValuesError):
        await client.execute_select_query("SELECT * FROM test_collection")

@pytest.mark.asyncio
async def test_execute_select_query_parse_error(client):
    """Test SQL query with parse error."""
    # Mock parser.preprocess_query
    client.query_builder.parser.preprocess_query = Mock(return_value="INVALID SQL")
    
    # Mock the parse_and_validate_select
    client.query_builder.parse_and_validate_select = Mock(return_value=(Mock(), "test_collection", None))
    
    # Mock the query executor to raise a SQLParseError
    client.query_executor.execute_select_query = AsyncMock(side_effect=SQLParseError("parse failed: syntax error", 10))
    
    # Execute the query and verify the error
    with pytest.raises(SQLParseError):
        await client.execute_select_query("INVALID SQL")
