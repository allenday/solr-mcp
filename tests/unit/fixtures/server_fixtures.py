"""Server fixtures for unit tests."""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from solr_mcp.server import SolrMCPServer

from .common import MOCK_RESPONSES


@pytest.fixture
def mock_server(mock_solr_client, mock_config):
    """Create a mock SolrMCPServer for testing."""
    server = SolrMCPServer(
        solr_base_url=mock_config.solr_base_url,
        zookeeper_hosts=mock_config.zookeeper_hosts,
        connection_timeout=mock_config.connection_timeout,
    )
    server.solr_client = mock_solr_client
    return server


@pytest.fixture
def mock_server_instance():
    """Create a mock FastMCP server instance for testing."""
    mock_server = MagicMock()

    # Mock list collections response
    async def mock_list_collections(*args, **kwargs):
        return [{"type": "text", "text": json.dumps(MOCK_RESPONSES["collections"])}]

    mock_server.list_collections = AsyncMock(side_effect=mock_list_collections)

    # Mock select query response
    async def mock_select(*args, **kwargs):
        return [
            {
                "type": "text",
                "text": json.dumps({"rows": [{"id": "1", "title": "Test Doc"}]}),
            }
        ]

    mock_server.select = AsyncMock(side_effect=mock_select)

    # Mock vector select response
    async def mock_vector_select(*args, **kwargs):
        return [
            {
                "type": "text",
                "text": json.dumps({"rows": [{"id": "1", "title": "Test Doc"}]}),
            }
        ]

    mock_server.vector_select = AsyncMock(side_effect=mock_vector_select)

    # Mock semantic select response
    async def mock_semantic_select(*args, **kwargs):
        return [
            {
                "type": "text",
                "text": json.dumps({"rows": [{"id": "1", "title": "Test Doc"}]}),
            }
        ]

    mock_server.semantic_select = AsyncMock(side_effect=mock_semantic_select)

    return mock_server


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
        yield {"MockServer": MockServer, "server1": server1, "server2": server2}
