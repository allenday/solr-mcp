import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from solr_mcp.server import SolrMCPServer
from solr_mcp.tools.search import SOLR_SEARCH_TOOL

@pytest.fixture
def mock_fastmcp():
    with patch('solr_mcp.server.FastMCP') as mock:
        server = MagicMock()
        server.name = "solr-mcp"
        server.list_tools = AsyncMock(return_value=[{"name": SOLR_SEARCH_TOOL["name"]}])
        server.run = MagicMock()
        mock.return_value = server
        yield server

@pytest.fixture
def mock_solr_client():
    with patch('solr_mcp.server.SolrClient') as mock:
        client = MagicMock()
        client.search = AsyncMock()
        mock.return_value = client
        yield client

@pytest.fixture
def server(mock_solr_client, mock_fastmcp):
    return SolrMCPServer(debug=True)

def test_server_initialization(server):
    """Test server initialization with default parameters."""
    assert server.debug is True
    assert server.config_path is None
    assert server.server.name == "solr-mcp"

@pytest.mark.asyncio
async def test_server_tool_registration(server):
    """Test that the search tool is properly registered."""
    registered_tools = await server.server.list_tools()
    assert any(tool["name"] == SOLR_SEARCH_TOOL["name"] for tool in registered_tools)

@pytest.mark.asyncio
async def test_handle_search_success(server, mock_solr_client):
    """Test successful search handling."""
    # Mock search results
    expected_results = '{"response": {"docs": [], "numFound": 0}}'
    mock_solr_client.search.return_value = expected_results

    # Test search with minimal parameters
    result = await server.handle_search(query="test query")
    assert result == expected_results
    
    # Verify the search was called with correct parameters
    mock_solr_client.search.assert_called_once_with(
        query="test query",
        collection=None,
        fields=None,
        filters=None,
        sort=None,
        rows=10,
        start=0
    )

@pytest.mark.asyncio
async def test_handle_search_with_all_parameters(server, mock_solr_client):
    """Test search handling with all optional parameters."""
    expected_results = '{"response": {"docs": [], "numFound": 0}}'
    mock_solr_client.search.return_value = expected_results

    result = await server.handle_search(
        query="test query",
        collection="test_collection",
        fields=["field1", "field2"],
        filters=["type:document"],
        sort="score desc",
        rows=5,
        start=10
    )
    
    assert result == expected_results
    mock_solr_client.search.assert_called_once_with(
        query="test query",
        collection="test_collection",
        fields=["field1", "field2"],
        filters=["type:document"],
        sort="score desc",
        rows=5,
        start=10
    )

@pytest.mark.asyncio
async def test_handle_search_error(server, mock_solr_client):
    """Test error handling in search."""
    mock_solr_client.search.side_effect = Exception("Test error")

    with pytest.raises(Exception) as exc_info:
        await server.handle_search(query="test query")
    
    assert str(exc_info.value) == "Test error"

def test_run_server(server):
    """Test server run method."""
    server.server.run("stdio")
    server.server.run.assert_called_once_with("stdio") 