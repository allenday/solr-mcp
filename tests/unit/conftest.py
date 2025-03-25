"""Test configuration and fixtures."""

import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import List, Dict, Any, Optional
import numpy as np
import pysolr
import requests
from kazoo.client import KazooClient
from kazoo.exceptions import NoNodeError, ConnectionLoss

from solr_mcp.solr.client import SolrClient
from solr_mcp.server import SolrMCPServer
from solr_mcp.solr.interfaces import CollectionProvider, VectorSearchProvider
from solr_mcp.solr.config import SolrConfig
from solr_mcp.solr.exceptions import ConnectionError, QueryError, SolrError
from solr_mcp.solr.schema import FieldManager
from solr_mcp.solr.query import QueryBuilder
from solr_mcp.solr.vector.manager import VectorManager
from solr_mcp.vector_provider.clients.ollama import OllamaVectorProvider

# Mock response data
MOCK_COLLECTIONS = ["collection1", "collection2"]
MOCK_SELECT_RESPONSE = {
    "result-set": {
        "docs": [{"id": "1", "field": "value"}],
        "numFound": 1
    }
}
MOCK_VECTOR_RESPONSE = {
    "result-set": {
        "docs": [{"id": "1", "field": "value", "score": 0.95}],
        "numFound": 1
    }
}
MOCK_SEMANTIC_RESPONSE = {
    "result-set": {
        "docs": [{"id": "1", "field": "value", "score": 0.85}],
        "numFound": 1
    }
}

# Mock schema response data
MOCK_SCHEMA_RESPONSE = {
    "schema": {
        "fields": [
            {
                "name": "id",
                "type": "string",
                "multiValued": False,
                "required": True
            },
            {
                "name": "title",
                "type": "text_general",
                "multiValued": False
            },
            {
                "name": "content",
                "type": "text_general",
                "multiValued": False
            },
            {
                "name": "vector",
                "type": "knn_vector",
                "multiValued": False
            }
        ],
        "fieldTypes": [
            {
                "name": "string",
                "class": "solr.StrField",
                "sortMissingLast": True
            },
            {
                "name": "text_general",
                "class": "solr.TextField",
                "positionIncrementGap": "100"
            },
            {
                "name": "knn_vector",
                "class": "solr.DenseVectorField",
                "vectorDimension": 768
            }
        ]
    }
}

@pytest.fixture
def mock_pysolr():
    """Mock pysolr.Solr instance."""
    mock = Mock(spec=pysolr.Solr)
    
    def mock_search(*args, **kwargs):
        # Check if this is a vector/knn query
        if args and '{!knn' in args[0]:
            return {
                "response": {
                    "docs": [
                        {"id": "1", "score": 0.95, "_vector_distance_": 0.05},
                        {"id": "2", "score": 0.85, "_vector_distance_": 0.15}
                    ],
                    "numFound": 2,
                    "maxScore": 0.95
                }
            }
        # Default response for regular queries
        return {
            "response": {
                "docs": [{"id": "1"}],
                "numFound": 1,
                "maxScore": 1.0
            }
        }
    
    mock.search.side_effect = mock_search
    return mock

@pytest.fixture
def mock_config():
    """Mock SolrConfig for testing."""
    config = Mock(spec=SolrConfig)
    config.solr_base_url = "http://localhost:8983/solr"
    config.zookeeper_hosts = ["localhost:2181"]
    config.default_collection = "test"
    config.connection_timeout = 10
    config.vector_field = "vector"
    config.default_top_k = 10
    return config

class MockCollectionProvider(CollectionProvider):
    """Mock implementation of CollectionProvider."""
    
    def __init__(self, collections=None):
        """Initialize with optional list of collections."""
        self.collections = collections if collections is not None else ["collection1", "collection2"]
        
    def list_collections(self) -> List[str]:
        """Return mock list of collections."""
        return self.collections

class MockVectorProvider(VectorSearchProvider):
    """Mock vector provider for testing."""

    async def execute_vector_search(self, client, vector, top_k=10):
        """Mock vector search execution."""
        return {
            'response': {
                'docs': [
                    {'_docid_': '1', 'score': 0.9, '_vector_distance_': 0.1},
                    {'_docid_': '2', 'score': 0.8, '_vector_distance_': 0.2},
                    {'_docid_': '3', 'score': 0.7, '_vector_distance_': 0.3}
                ],
                'numFound': 3,
                'start': 0
            }
        }

    async def get_vector(self, text: str) -> List[float]:
        """Mock text to vector conversion."""
        return [0.1, 0.2, 0.3]

@pytest.fixture
def mock_solr_client():
    """Mock Solr client for testing."""
    client = Mock(spec=SolrClient)
    client.execute_select_query = AsyncMock()
    client.execute_vector_select_query = AsyncMock()
    client.execute_semantic_select_query = AsyncMock()
    client.list_collections = AsyncMock()
    client.list_fields = AsyncMock()
    return client

@pytest.fixture
def mock_server(mock_solr_client, mock_config):
    """Mock SolrMCPServer for testing."""
    server = SolrMCPServer(
        solr_base_url=mock_config.solr_base_url,
        zookeeper_hosts=mock_config.zookeeper_hosts,
        default_collection=mock_config.default_collection,
        connection_timeout=mock_config.connection_timeout,
        vector_field=mock_config.vector_field,
        default_top_k=mock_config.default_top_k
    )
    server.solr_client = mock_solr_client
    return server

@pytest.fixture
def mock_server_instance():
    """Create a mock FastMCP server instance for testing."""
    mock_server = MagicMock()
    
    # Mock list collections response
    async def mock_list_collections(*args, **kwargs):
        return [{"type": "text", "text": json.dumps(["collection1", "collection2"])}]
    mock_server.list_collections = AsyncMock(side_effect=mock_list_collections)
    
    # Mock select query response
    async def mock_select(*args, **kwargs):
        return [{"type": "text", "text": json.dumps({"rows": [{"id": "1", "title": "Test Doc"}]})}]
    mock_server.select = AsyncMock(side_effect=mock_select)
    
    # Mock vector select response
    async def mock_vector_select(*args, **kwargs):
        return [{"type": "text", "text": json.dumps({"rows": [{"id": "1", "title": "Test Doc"}]})}]
    mock_server.vector_select = AsyncMock(side_effect=mock_vector_select)
    
    # Mock semantic select response
    async def mock_semantic_select(*args, **kwargs):
        return [{"type": "text", "text": json.dumps({"rows": [{"id": "1", "title": "Test Doc"}]})}]
    mock_server.semantic_select = AsyncMock(side_effect=mock_semantic_select)
    
    return mock_server

@pytest.fixture
def mock_error_client():
    """Mock SolrClient that raises exceptions for testing error cases."""
    client = MagicMock(spec=SolrClient)
    client.list_collections = AsyncMock(side_effect=ConnectionError("Test error"))
    client.execute_select_query = AsyncMock(side_effect=QueryError("Test error"))
    client.execute_vector_select_query = AsyncMock(side_effect=QueryError("Test error"))
    client.execute_semantic_select_query = AsyncMock(side_effect=QueryError("Test error"))
    return client

@pytest.fixture
def mock_error_server(mock_error_client, mock_config):
    """Mock SolrMCPServer with error client for testing error cases."""
    server = SolrMCPServer(
        solr_base_url=mock_config.solr_base_url,
        zookeeper_hosts=mock_config.zookeeper_hosts,
        default_collection=mock_config.default_collection,
        connection_timeout=mock_config.connection_timeout,
        vector_field=mock_config.vector_field,
        default_top_k=mock_config.default_top_k
    )
    server.solr_client = mock_error_client
    return server

@pytest.fixture
def mock_singleton_server():
    """Mock SolrMCPServer for singleton pattern testing."""
    # Create a mock class to avoid affecting real singleton
    MockServer = Mock(spec=SolrMCPServer)
    MockServer._instance = None
    
    # Create a proper classmethod mock
    def get_instance():
        return MockServer._instance
    MockServer.get_instance = classmethod(get_instance)
    
    # Create two different instances
    server1 = Mock(spec=SolrMCPServer)
    server2 = Mock(spec=SolrMCPServer)
    
    with patch("solr_mcp.server.SolrMCPServer", MockServer):
        yield {
            "MockServer": MockServer,
            "server1": server1,
            "server2": server2
        }

@pytest.fixture
def mock_field_manager():
    """Mock field manager for testing list_fields tool."""
    manager = MagicMock()
    manager.get_collection_fields.return_value = {
        "fields": [
            {
                "name": "id",
                "type": "string",
                "indexed": True,
                "stored": True,
                "docValues": True,
                "multiValued": False
            },
            {
                "name": "_text_",
                "type": "text_general",
                "indexed": True,
                "stored": False,
                "docValues": False,
                "multiValued": True,
                "copies_from": ["title", "content"]
            }
        ]
    }
    return manager

@pytest.fixture
def mock_schema_response():
    """Mock response for schema API calls."""
    return MOCK_SCHEMA_RESPONSE

@pytest.fixture
def mock_ollama():
    """Mock Ollama client for testing."""
    provider = Mock(spec=OllamaVectorProvider)
    provider.get_vector.return_value = [0.1, 0.2, 0.3]
    return provider

@pytest.fixture
def mock_http_client(mock_schema_response):
    """Mock HTTP client for Solr requests."""
    mock = Mock(spec=requests)
    
    # Mock response object
    mock_response = Mock()
    mock_response.status_code = 200
    
    # Configure responses for different endpoints
    def mock_request(method, url, **kwargs):
        mock_response.json.return_value = MOCK_SELECT_RESPONSE
        
        if "/sql" in url:
            if method.lower() == "post":
                mock_response.json.return_value = MOCK_SELECT_RESPONSE
        elif "/schema" in url:
            mock_response.json.return_value = mock_schema_response
        elif "/fields" in url:
            mock_response.json.return_value = {
                "fields": mock_schema_response["schema"]["fields"]
            }
        
        return mock_response
    
    # Setup the mock methods
    mock.get = Mock(side_effect=lambda url, **kwargs: mock_request("get", url, **kwargs))
    mock.post = Mock(side_effect=lambda url, **kwargs: mock_request("post", url, **kwargs))
    
    return mock

@pytest.fixture
def mock_time_expired():
    """Mock time.time to return a timestamp 5 minutes in the future."""
    with patch('time.time', return_value=time.time() + 301) as mock:
        yield mock

@pytest.fixture
def mock_time_custom(minutes: int = 2):
    """Mock time.time to return a timestamp N minutes in the future.
    
    Args:
        minutes: Number of minutes to advance time by (default: 2)
    """
    with patch('time.time', return_value=time.time() + (minutes * 60)) as mock:
        yield mock 

@pytest.fixture
def mock_kazoo_client(request):
    """Create mock KazooClient.
    
    Args:
        request: Pytest request object with param indicating the mock type:
            - "success": Returns list of collections
            - "no_collections": No collections path exists
            - "empty": Empty collections list
            - "error": Raises ConnectionLoss on get_children
            - "connection_error": Raises ConnectionLoss on start
    """
    mock = MagicMock(spec=KazooClient)
    
    # Default to success behavior if no param is provided
    mock_type = getattr(request, 'param', 'success')
    
    if mock_type == "success":
        mock.get_children.return_value = ["collection1", "collection2"]
        mock.exists.return_value = True
        mock.start.return_value = None
        mock.stop.return_value = None
    elif mock_type == "no_collections":
        mock.exists.return_value = False
        mock.get_children.side_effect = NoNodeError
        mock.start.return_value = None
        mock.stop.return_value = None
    elif mock_type == "empty":
        mock.exists.return_value = True
        mock.get_children.return_value = []
        mock.start.return_value = None
        mock.stop.return_value = None
    elif mock_type == "error":
        mock.exists.return_value = True
        mock.get_children.side_effect = ConnectionLoss("ZooKeeper error")
        mock.start.return_value = None
        mock.stop.return_value = None
    elif mock_type == "connection_error":
        mock.start.side_effect = ConnectionLoss("ZooKeeper connection error")
        mock.stop.return_value = None
        
    return mock

@pytest.fixture(autouse=True)
def mock_kazoo_client_factory(mock_kazoo_client):
    """Create a factory for KazooClient mocks."""
    with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
        mock_factory.return_value = mock_kazoo_client
        yield mock_factory

@pytest.fixture
def provider(mock_kazoo_client):
    """Create ZooKeeperCollectionProvider instance."""
    from solr_mcp.solr.zookeeper import ZooKeeperCollectionProvider
    provider = ZooKeeperCollectionProvider(hosts=["localhost:2181"])
    provider.zk = mock_kazoo_client  # Ensure the mock is used
    return provider

@pytest.fixture
def mock_schema_requests(mock_http_client):
    """Mock requests module for schema operations."""
    with patch("solr_mcp.solr.schema.fields.requests", mock_http_client):
        yield mock_http_client

@pytest.fixture
def mock_solr_requests():
    """Mock requests module for Solr operations.
    
    This fixture patches requests.post for SQL queries and other Solr operations.
    """
    mock = Mock(spec=requests.Session)
    
    # Mock response object
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result-set": {
            "docs": [{"id": "1"}],
            "numFound": 1
        }
    }
    
    # Setup the mock methods
    mock.post = Mock(return_value=mock_response)
    
    with patch("requests.post", mock.post):
        yield mock 

@pytest.fixture
def mock_valid_config_file():
    """Mock a valid configuration file."""
    config_content = """
    {
        "solr_base_url": "http://solr:8983/solr",
        "zookeeper_hosts": ["zk1:2181", "zk2:2181"],
        "default_collection": "custom",
        "connection_timeout": 30,
        "vector_field": "vector",
        "default_top_k": 20
    }
    """
    with patch("builtins.open", mock_open(read_data=config_content)):
        yield

@pytest.fixture
def mock_invalid_json_config():
    """Mock an invalid JSON configuration file."""
    config_content = "invalid json content"
    with patch("builtins.open", mock_open(read_data=config_content)):
        yield

@pytest.fixture
def mock_minimal_config_file():
    """Mock a minimal configuration file with only required fields."""
    config_content = """
    {
        "zookeeper_hosts": ["zk1:2181"],
        "default_collection": "custom"
    }
    """
    with patch("builtins.open", mock_open(read_data=config_content)):
        yield

@pytest.fixture
def mock_missing_file():
    """Mock a missing configuration file."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        yield 

@pytest.fixture
def mock_field_manager_methods():
    """Mock FieldManager methods for testing."""
    mock_fields = {
        "searchable_fields": ["title", "content"],
        "sortable_fields": {
            "id": {
                "directions": ["asc", "desc"],
                "default_direction": "asc"
            },
            "score": {
                "directions": ["asc", "desc"],
                "default_direction": "desc",
                "type": "numeric",
                "searchable": True
            }
        }
    }

    def patch_get_collection_fields(field_manager):
        """Create a context manager for patching _get_collection_fields."""
        return patch.object(field_manager, '_get_collection_fields', return_value=mock_fields)

    def patch_get_searchable_fields(field_manager):
        """Create a context manager for patching _get_searchable_fields."""
        return patch.object(field_manager, '_get_searchable_fields', side_effect=Exception("API error"))

    return {
        "mock_fields": mock_fields,
        "patch_get_collection_fields": patch_get_collection_fields,
        "patch_get_searchable_fields": patch_get_searchable_fields
    }

@pytest.fixture
def mock_solr_instance(mock_pysolr):
    """Mock pysolr.Solr instance with patching.
    
    This fixture patches pysolr.Solr to return a mock instance.
    """
    with patch("pysolr.Solr", return_value=mock_pysolr):
        yield mock_pysolr 

@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    return {
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        "model": "nomic-embed-text"
    }

@pytest.fixture
def mock_ollama_error_response():
    """Mock failed Ollama API response."""
    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error": "Internal server error"}'
    return response

@pytest.fixture
def mock_requests_post(mock_ollama_response):
    """Mock requests.post for Ollama API calls."""
    with patch('requests.post') as mock_post:
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(mock_ollama_response).encode('utf-8')
        mock_post.return_value = response
        yield mock_post 

@pytest.fixture
def mock_vector_manager():
    """Mock VectorManager for testing."""
    manager = Mock(spec=VectorManager)
    manager.get_vector.return_value = [0.1, 0.2, 0.3]
    return manager

@pytest.fixture
def mock_query_builder():
    """Create a mock QueryBuilder."""
    builder = Mock(spec=QueryBuilder)
    builder.parse_and_validate_select.return_value = (
        Mock(args={"limit": 10, "offset": 0}),  # AST
        "test_collection",  # Collection name
        ["id", "title"]  # Fields
    )
    return builder

@pytest.fixture
def mock_response():
    """Create a mock HTTP response."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "collections": ["test_collection", "other_collection"],
        "schema": {
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "title", "type": "text_general"}
            ],
            "copyFields": [
                {"source": "title", "dest": "_text_"}
            ]
        }
    }
    return response 

@pytest.fixture
def mock_error_response():
    response = Mock()
    response.status_code = 500
    response.text = "Schema Error"
    return response

@pytest.fixture
def mock_query_builder():
    builder = Mock()
    parser = Mock()
    parser.preprocess_query = Mock(return_value="preprocessed query")
    builder.parser = parser
    builder.parse_and_validate_select = Mock(return_value=(
        Mock(args={}),
        "test_collection",
        ["id", "title"]
    ))
    builder.build_vector_query = Mock(return_value={
        'fq': ['1', '2', '3'],
        'rows': 10
    })
    return builder

@pytest.fixture
def client(mock_config, mock_collection_provider, mock_field_manager, mock_vector_provider, mock_query_builder):
    return SolrClient(
        config=mock_config,
        collection_provider=mock_collection_provider,
        field_manager=mock_field_manager,
        vector_provider=mock_vector_provider,
        query_builder=mock_query_builder
    )

@pytest.fixture
def mock_solr_client():
    client = Mock(spec=pysolr.Solr)
    client.search.return_value = {
        'response': {
            'docs': [{'_docid_': '1', 'score': 0.9, '_vector_distance_': 0.1}],
            'numFound': 1
        }
    }
    return client

@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session with proper async context management."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {'Content-Type': 'application/json'}
    mock_response.text = AsyncMock(return_value='{"result-set": {"docs": [{"id": "1"}], "numFound": 1}}')
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()

    mock_session = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    # Mock the vector search response
    mock_vector_response = {
        'response': {
            'docs': [{'_docid_': '1', 'score': 0.9, '_vector_distance_': 0.1}],
            'numFound': 1
        }
    }
    mock_solr_response = AsyncMock()
    mock_solr_response.search = AsyncMock(return_value=mock_vector_response)

    return mock_session

@pytest.fixture
def valid_config_dict():
    """Valid configuration dictionary."""
    return {
        "solr_base_url": "http://localhost:8983/solr",
        "zookeeper_hosts": ["localhost:2181"],
        "default_collection": "test",
        "connection_timeout": 10,
        "vector_field": "vector",
        "default_top_k": 10
    }

@pytest.fixture
def mock_collection_provider():
    """Mock collection provider for testing."""
    provider = Mock()
    provider.list_collections.return_value = ["collection1", "collection2"]
    return provider

@pytest.fixture
def mock_vector_provider(mocker):
    """Mock vector provider."""
    provider = mocker.Mock()
    provider.get_vector.return_value = [0.1] * 768
    return provider
