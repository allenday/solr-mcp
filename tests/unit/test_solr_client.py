import json
import os
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from solr_mcp.solr.client import SolrClient, SolrConfig, SolrError
import pysolr

@pytest.fixture
def mock_zk_client():
    with patch('solr_mcp.solr.client.KazooClient') as mock:
        client = MagicMock()
        client.exists.return_value = True
        client.get_children.return_value = ["collection1", "collection2"]
        # Mock the start method to avoid connection timeout
        client.start.return_value = None
        mock.return_value = client
        yield client

@pytest.fixture
def mock_pysolr():
    with patch('solr_mcp.solr.client.pysolr') as mock:
        solr = MagicMock()
        mock.Solr.return_value = solr
        yield mock

@pytest.fixture
def solr_client(mock_zk_client, mock_pysolr):
    with patch('solr_mcp.solr.client.OllamaClient', create=True):  # Just ignore OllamaClient
        client = SolrClient()
        return client

def test_default_config(solr_client):
    """Test default configuration initialization."""
    assert solr_client.config.zookeeper_hosts == ["localhost:2181"]
    assert solr_client.config.solr_base_url == "http://localhost:8983/solr"
    assert solr_client.config.default_collection == "unified"
    assert solr_client.config.connection_timeout == 10
    assert solr_client.config.max_retries == 3

def test_config_from_env():
    """Test configuration from environment variables."""
    env_vars = {
        "SOLR_MCP_ZK_HOSTS": "host1:2181,host2:2181",
        "SOLR_MCP_SOLR_URL": "http://solr:8983/solr",
        "SOLR_MCP_DEFAULT_COLLECTION": "test_collection"
    }

    with patch.dict(os.environ, env_vars, clear=True), \
         patch('solr_mcp.solr.client.OllamaClient', create=True), \
         patch('solr_mcp.solr.client.KazooClient') as mock_kazoo:
        # Mock ZooKeeper client
        mock_zk = MagicMock()
        mock_zk.start.return_value = None
        mock_zk.exists.return_value = True
        mock_zk.get_children.return_value = ["collection1", "collection2"]
        mock_kazoo.return_value = mock_zk

        client = SolrClient()
        assert client.config.zookeeper_hosts == ["host1:2181", "host2:2181"]
        assert client.config.solr_base_url == "http://solr:8983/solr"
        assert client.config.default_collection == "test_collection"

def test_config_from_file(tmp_path):
    """Test configuration from file."""
    config = {
        "zookeeper_hosts": ["zk1:2181", "zk2:2181"],
        "solr_base_url": "http://custom-solr:8983/solr",
        "default_collection": "custom_collection",
        "connection_timeout": 20,
        "max_retries": 5
    }

    config_file = tmp_path / "solr_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    with patch('solr_mcp.solr.client.OllamaClient', create=True), \
         patch('solr_mcp.solr.client.KazooClient') as mock_kazoo:
        # Mock ZooKeeper client
        mock_zk = MagicMock()
        mock_zk.start.return_value = None
        mock_zk.exists.return_value = True
        mock_zk.get_children.return_value = ["collection1", "collection2"]
        mock_kazoo.return_value = mock_zk

        client = SolrClient(str(config_file))
        assert client.config.zookeeper_hosts == ["zk1:2181", "zk2:2181"]
        assert client.config.solr_base_url == "http://custom-solr:8983/solr"
        assert client.config.default_collection == "custom_collection"
        assert client.config.connection_timeout == 20
        assert client.config.max_retries == 5

def test_list_collections(solr_client):
    """Test listing collections."""
    # Reset the mock to clear any previous calls
    solr_client.zk_client.exists.reset_mock()
    
    collections = solr_client.list_collections()
    assert collections == ["collection1", "collection2"]
    solr_client.zk_client.exists.assert_called_once_with("/collections")
    solr_client.zk_client.get_children.assert_called_with("/collections")

def test_list_collections_no_zk(mock_pysolr):
    """Test listing collections without ZooKeeper."""
    with patch('solr_mcp.solr.client.KazooClient') as mock_zk:
        # Mock ZooKeeper to raise an error on start
        mock_client = MagicMock()
        mock_client.start.side_effect = Exception("ZK Error")
        mock_zk.return_value = mock_client
        
        with patch('solr_mcp.solr.client.OllamaClient', create=True), \
             pytest.raises(Exception, match="ZK Error"):
            client = SolrClient()

@pytest.mark.asyncio
async def test_search_basic(solr_client, mock_pysolr):
    """Test basic search functionality."""
    # Mock search results
    mock_results = MagicMock()
    mock_results.hits = 10
    mock_results.docs = [{"id": "1", "title": "Test"}]
    mock_results.responseHeader = {"params": {"start": "0"}}
    mock_results.error = None
    
    # Get or create the client for the default collection
    client = solr_client._get_or_create_client(solr_client.config.default_collection)
    client.search = MagicMock(return_value=mock_results)
    
    result = await solr_client.search("test query")
    result_dict = json.loads(result)
    
    assert result_dict["numFound"] == 10
    assert len(result_dict["docs"]) == 1
    assert result_dict["docs"][0]["id"] == "1"

@pytest.mark.asyncio
async def test_search_with_params(solr_client, mock_pysolr):
    """Test search with additional parameters."""
    mock_results = MagicMock()
    mock_results.hits = 5
    mock_results.docs = [{"id": "1"}]
    mock_results.responseHeader = {"params": {"start": "10"}}
    mock_results.error = None
    
    # Create the client for the custom collection before testing
    custom_client = solr_client._get_or_create_client("custom_collection")
    custom_client.search = MagicMock(return_value=mock_results)
    
    result = await solr_client.search(
        query="test",
        collection="custom_collection",
        fields=["id", "title"],
        filters=["type:document"],
        sort="score desc",
        start=10,
        rows=5
    )
    result_dict = json.loads(result)
    
    assert result_dict["numFound"] == 5
    assert result_dict["start"] == 10
    
    # Verify search was called with correct parameters
    custom_client.search.assert_called_once()
    call_args = custom_client.search.call_args[1]
    assert "type:document" in call_args.get("fq", [])
    assert call_args.get("fl") == "id,title"
    assert call_args.get("sort") == "score desc"
    assert call_args.get("start") == 10
    assert call_args.get("rows") == 5

@pytest.mark.asyncio
async def test_search_error_handling(solr_client, mock_pysolr):
    """Test search error handling."""
    # Mock a SolrError with a syntax error
    error_msg = "org.apache.solr.search.SyntaxError: Invalid query"
    solr_client._get_or_create_client(solr_client.config.default_collection).search = MagicMock(side_effect=Exception(error_msg))
    
    with pytest.raises(SolrError) as exc_info:
        await solr_client.search("test query")
    
    assert "org.apache.solr.search.SyntaxError" in str(exc_info.value) 