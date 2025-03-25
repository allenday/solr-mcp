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
    # Create a mock response for the request
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "collections": ["test_collection"]
    }
    
    # Use the mock in the test
    with patch('requests.get', return_value=mock_resp):
        result = await client.list_collections()
        assert result == ["test_collection"]

@pytest.mark.asyncio
async def test_list_fields_schema_error(client):
    """Test schema error handling in list_fields."""
    # Create a successful response for the first call
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "collections": ["test_collection"]
    }
    
    # Create an error response for the second call
    mock_error = Mock(spec=requests.Response)
    mock_error.status_code = 500
    mock_error.text = "Schema error"
    
    with patch('requests.get', side_effect=[mock_resp, mock_error]):
        with pytest.raises(SolrError):
            await client.list_fields("test_collection")

@pytest.mark.asyncio
async def test_execute_select_query_success(client):
    """Test successful SQL query execution."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result-set": {
            "docs": [{"id": "1", "title": "Test"}],
            "numFound": 1
        }
    }

    with patch('requests.post', return_value=mock_response):
        result = await client.execute_select_query("SELECT * FROM test_collection")
        assert result["result-set"]["docs"][0]["id"] == "1"

@pytest.mark.asyncio
async def test_execute_select_query_docvalues_error(client):
    """Test SQL query with DocValues error."""
    error_response = Mock()
    error_response.status_code = 200
    error_response.json.return_value = {
        "result-set": {
            "docs": [{
                "EXCEPTION": "must have DocValues to use this feature",
                "RESPONSE_TIME": 10
            }]
        }
    }

    with patch('requests.post', return_value=error_response):
        with pytest.raises(DocValuesError):
            await client.execute_select_query("SELECT * FROM test_collection")

@pytest.mark.asyncio
async def test_execute_select_query_parse_error(client):
    """Test SQL query with parse error."""
    error_response = Mock()
    error_response.status_code = 200
    error_response.json.return_value = {
        "result-set": {
            "docs": [{
                "EXCEPTION": "parse failed: syntax error",
                "RESPONSE_TIME": 10
            }]
        }
    }

    with patch('requests.post', return_value=error_response):
        with pytest.raises(SQLParseError):
            await client.execute_select_query("INVALID SQL")
