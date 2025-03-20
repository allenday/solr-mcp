import os
import json
import pytest
import pysolr
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from sqlglot import parse_one, exp
from solr_mcp.solr.client import SolrClient, SolrConfig, SolrError

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
async def test_execute_select_query_basic(solr_client, mock_pysolr):
    """Test basic search functionality."""
    # Mock search results
    mock_results = MagicMock()
    mock_results.hits = 10
    mock_results.docs = [{"id": "1", "title": "Test"}]
    mock_results.responseHeader = {"params": {"start": "0"}}
    mock_results.error = None
    
    # Mock requests.post
    with patch('solr_mcp.solr.client.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "docs": [{"id": "1", "title": "Test"}],
            "numFound": 10,
            "start": 0
        }
        mock_post.return_value = mock_response
        
        result = await solr_client.execute_select_query("SELECT * FROM unified WHERE _text_:test")
        
        assert "rows" in result
        assert "numFound" in result
        assert "offset" in result
        assert len(result["rows"]) > 0

@pytest.mark.asyncio
async def test_execute_select_query_with_params(solr_client, mock_pysolr):
    """Test search with additional parameters."""
    # Mock field cache for validation
    solr_client._field_cache = {
        "custom_collection": {
            "searchable_fields": ["id", "title", "_text_", "type"],
            "sortable_fields": {
                "score": {
                    "type": "numeric",
                    "directions": ["asc", "desc"],
                    "default_direction": "desc",
                    "searchable": True
                },
                "id": {
                    "type": "string",
                    "directions": ["asc", "desc"],
                    "default_direction": "asc",
                    "searchable": True
                }
            },
            "last_updated": time.time()
        }
    }
    
    # Mock requests.post
    with patch('solr_mcp.solr.client.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "docs": [{"id": "1"}],
            "numFound": 5,
            "start": 10
        }
        mock_post.return_value = mock_response
        
        result = await solr_client.execute_select_query(
            "SELECT id, title FROM custom_collection WHERE _text_:test AND type:document ORDER BY score desc LIMIT 5 OFFSET 10"
        )
        
        assert "rows" in result
        assert "numFound" in result
        assert "offset" in result
        assert result["numFound"] == 5
        assert result["offset"] == 10
        
        # Verify the SQL query was sent correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert "json" in call_args
        assert "stmt" in call_args["json"]

@pytest.mark.asyncio
async def test_search_error_handling(solr_client, mock_pysolr):
    """Test search error handling."""
    error_msg = "org.apache.solr.search.SyntaxError: Invalid query"
    
    # Mock requests.post to raise an error
    with patch('solr_mcp.solr.client.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = error_msg
        mock_post.return_value = mock_response
        
        with pytest.raises(SolrError) as exc_info:
            await solr_client.execute_select_query("SELECT * FROM unified WHERE _text_:test")
        
        assert error_msg in str(exc_info.value)

@pytest.mark.asyncio
async def test_field_validation_and_caching(solr_client):
    """Test field validation and caching functionality."""
    collection = "test_collection"
    
    # Mock schema API response
    with patch('solr_mcp.solr.client.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "title", "type": "text_general"},
                {"name": "content", "type": "text_general"},
                {"name": "date", "type": "pdate"},
                {"name": "score", "type": "float"}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # First call should populate cache
        fields = solr_client._get_collection_fields(collection)
        assert "searchable_fields" in fields
        assert "sortable_fields" in fields
        assert "last_updated" in fields
        
        # Verify searchable fields
        assert set(fields["searchable_fields"]) >= {"title", "content", "_text_"}
        
        # Verify sortable fields
        assert set(fields["sortable_fields"].keys()) >= {"date", "score"}
        assert fields["sortable_fields"]["score"]["type"] == "numeric"
        
        # Second call should use cache
        fields_cached = solr_client._get_collection_fields(collection)
        assert fields == fields_cached
        assert mock_get.call_count <= 2  # Allow up to 2 calls (one for searchable, one for sortable)

@pytest.mark.asyncio
async def test_field_validation_errors(solr_client):
    """Test field validation error handling."""
    collection = "test_collection"
    
    # Mock field cache
    solr_client._field_cache = {
        collection: {
            "searchable_fields": ["title", "content"],
            "sortable_fields": {
                "date": {
                    "type": "date",
                    "directions": ["asc", "desc"],
                    "default_direction": "desc"
                }
            },
            "last_updated": time.time()
        }
    }
    
    # Test invalid field validation
    with pytest.raises(SolrError, match="Invalid fields"):
        solr_client._validate_fields(collection, ["nonexistent_field"])
    
    # Test invalid sort field validation
    with pytest.raises(SolrError, match="Fields not sortable"):
        solr_client._validate_sort_fields(collection, ["content"])

@pytest.mark.asyncio
async def test_sort_validation(solr_client):
    """Test sort parameter validation."""
    # Test valid sort
    valid_sort = solr_client._validate_sort("score desc")
    assert valid_sort == "score desc"
    
    # Test sort with default direction
    default_sort = solr_client._validate_sort("score")
    assert default_sort == "score desc"  # score defaults to desc
    
    # Test invalid sort field
    with pytest.raises(ValueError, match="Invalid sort format"):
        solr_client._validate_sort("score desc asc")
    
    # Test invalid sort direction
    with pytest.raises(ValueError, match="Invalid sort direction"):
        solr_client._validate_sort("score invalid")

@pytest.mark.asyncio
async def test_query_preprocessing(solr_client):
    """Test query preprocessing functionality."""
    # Test field:value syntax conversion
    query = "field1:value1 AND field2:value2"
    processed = solr_client._preprocess_solr_query(query)
    assert "field1 = 'value1'" in processed
    assert "field2 = 'value2'" in processed
    
    # Test mixed syntax
    query = "field1:value1 AND normal_field = 'value2'"
    processed = solr_client._preprocess_solr_query(query)
    assert "field1 = 'value1'" in processed
    assert "normal_field = 'value2'" in processed

@pytest.mark.asyncio
async def test_execute_select_query_invalid_query(solr_client):
    """Test handling of invalid SQL queries."""
    # Test non-SELECT query
    with pytest.raises(SolrError, match="Query must be a SELECT statement"):
        await solr_client.execute_select_query("INSERT INTO collection VALUES (1)")
    
    # Test missing FROM clause
    with pytest.raises(SolrError, match="Query must have a FROM clause"):
        await solr_client.execute_select_query("SELECT *")
    
    # Test invalid SQL syntax
    with pytest.raises(SolrError):
        await solr_client.execute_select_query("SELECT * INVALID SQL")

@pytest.mark.asyncio
async def test_execute_select_query_field_validation(solr_client):
    """Test field validation in SELECT queries."""
    # Mock field cache
    solr_client._field_cache = {
        "test_collection": {
            "searchable_fields": ["title", "content"],
            "sortable_fields": {
                "date": {
                    "type": "date",
                    "directions": ["asc", "desc"],
                    "default_direction": "desc"
                }
            },
            "last_updated": time.time()
        }
    }

    # Mock requests.post for both test cases
    with patch('solr_mcp.solr.client.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result-set": {
                "docs": [],
                "numFound": 0,
                "start": 0
            }
        }
        mock_post.return_value = mock_response

        # Test invalid field
        with pytest.raises(SolrError, match="Invalid fields for collection test_collection: nonexistent_field"):
            await solr_client.execute_select_query(
                "SELECT nonexistent_field FROM test_collection"
            )

        # Debug AST structure
        ast = parse_one("SELECT title FROM test_collection ORDER BY content")
        print(f"AST: {ast}")
        print(f"Order clause: {ast.args.get('order')}")
        if ast.args.get('order'):
            for expr in ast.args['order']:
                print(f"Order expr: {expr}")
                print(f"Order expr type: {type(expr)}")
                print(f"Order expr dir: {expr.args.get('direction')}")
                print(f"Order expr this: {expr.this}")
                print(f"Order expr this type: {type(expr.this)}")

        # Test invalid sort field
        with pytest.raises(SolrError, match="Fields not sortable in collection test_collection: content"):
            await solr_client.execute_select_query(
                "SELECT title FROM test_collection ORDER BY content"
            )

@pytest.mark.asyncio
async def test_field_retrieval_fallbacks(solr_client):
    """Test field retrieval fallback mechanisms."""
    collection = "test_collection"
    
    # Mock schema API to fail
    with patch('solr_mcp.solr.client.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Schema API error"
        mock_get.return_value = mock_response
        
        # Mock direct URL call
        with patch('solr_mcp.solr.client.requests.post') as mock_post:
            mock_post_response = MagicMock()
            mock_post_response.status_code = 200
            mock_post_response.json.return_value = {
                "responseHeader": {
                    "params": {
                        "fl": "id,title,content,date"
                    }
                }
            }
            mock_post.return_value = mock_post_response
            
            # Get fields should still work with fallback
            fields = solr_client._get_collection_fields(collection)
            assert "searchable_fields" in fields
            assert "sortable_fields" in fields
            assert set(fields["searchable_fields"]) >= {"content", "title", "_text_"}

@pytest.mark.asyncio
async def test_field_retrieval_error_handling(solr_client):
    """Test error handling in field retrieval."""
    collection = "test_collection"
    
    # Mock both schema API and direct URL to fail
    with patch('solr_mcp.solr.client.requests.get') as mock_get, \
         patch('solr_mcp.solr.client.requests.post') as mock_post:
        mock_get.side_effect = Exception("Schema API error")
        mock_post.side_effect = Exception("Direct URL error")
        
        # Should fall back to default fields
        fields = solr_client._get_collection_fields(collection)
        assert "searchable_fields" in fields
        assert "_text_" in fields["searchable_fields"]
        assert "score" in fields["sortable_fields"]

@pytest.mark.asyncio
async def test_sort_field_validation_edge_cases(solr_client):
    """Test edge cases in sort field validation."""
    collection = "test_collection"
    
    # Mock field cache with special fields
    solr_client._field_cache = {
        collection: {
            "searchable_fields": ["title"],
            "sortable_fields": {
                "score": {
                    "type": "numeric",
                    "directions": ["asc", "desc"],
                    "default_direction": "desc"
                },
                "_docid_": {
                    "type": "numeric",
                    "directions": ["asc", "desc"],
                    "default_direction": "asc",
                    "warning": "Internal Lucene document ID"
                }
            },
            "last_updated": time.time()
        }
    }
    
    # Test special field validation
    solr_client._validate_sort_fields(collection, ["score"])
    solr_client._validate_sort_fields(collection, ["_docid_"])
    
    # Test multiple fields
    solr_client._validate_sort_fields(collection, ["score", "_docid_"])
    
    # Test invalid field
    with pytest.raises(SolrError):
        solr_client._validate_sort_fields(collection, ["invalid_field"])

@pytest.mark.asyncio
async def test_query_preprocessing_edge_cases(solr_client):
    """Test edge cases in query preprocessing."""
    # Test empty query
    assert solr_client._preprocess_solr_query("") == ""
    
    # Test query with no field:value syntax
    query = "SELECT * FROM collection"
    assert solr_client._preprocess_solr_query(query) == query
    
    # Test query with multiple field:value pairs
    query = "field1:value1 AND field2:value2 OR field3:value3"
    processed = solr_client._preprocess_solr_query(query)
    assert "field1 = 'value1'" in processed
    assert "field2 = 'value2'" in processed
    assert "field3 = 'value3'" in processed
    
    # Test query with mixed syntax
    query = "field1:value1 AND (field2 = 'value2' OR field3:value3)"
    processed = solr_client._preprocess_solr_query(query)
    assert "field1 = 'value1'" in processed
    assert "field2 = 'value2'" in processed
    assert "field3 = 'value3'" in processed 