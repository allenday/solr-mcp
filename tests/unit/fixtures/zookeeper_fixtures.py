"""ZooKeeper fixtures for unit tests."""

from unittest.mock import MagicMock, patch

import pytest
from kazoo.client import KazooClient
from kazoo.exceptions import ConnectionLoss, NoNodeError


@pytest.fixture(
    params=["success", "no_collections", "empty", "error", "connection_error"]
)
def mock_kazoo_client(request):
    """Parameterized KazooClient mock with different behavior scenarios."""
    mock = MagicMock(spec=KazooClient)

    scenario = request.param

    if scenario == "success":
        mock.get_children.return_value = ["collection1", "collection2"]
        mock.exists.return_value = True
        mock.start.return_value = None
        mock.stop.return_value = None
    elif scenario == "no_collections":
        mock.exists.return_value = False
        mock.get_children.side_effect = NoNodeError
        mock.start.return_value = None
        mock.stop.return_value = None
    elif scenario == "empty":
        mock.exists.return_value = True
        mock.get_children.return_value = []
        mock.start.return_value = None
        mock.stop.return_value = None
    elif scenario == "error":
        mock.exists.return_value = True
        mock.get_children.side_effect = ConnectionLoss("ZooKeeper error")
        mock.start.return_value = None
        mock.stop.return_value = None
    elif scenario == "connection_error":
        mock.start.side_effect = ConnectionLoss("ZooKeeper connection error")
        mock.stop.return_value = None

    yield mock, scenario


@pytest.fixture
def mock_kazoo_client_factory(request):
    """Factory for creating KazooClient mocks with specific behavior."""

    def _create_client(scenario="success"):
        mock = MagicMock(spec=KazooClient)

        if scenario == "success":
            mock.get_children.return_value = ["collection1", "collection2"]
            mock.exists.return_value = True
            mock.start.return_value = None
            mock.stop.return_value = None
        elif scenario == "no_collections":
            mock.exists.return_value = False
            mock.get_children.side_effect = NoNodeError
            mock.start.return_value = None
            mock.stop.return_value = None
        elif scenario == "empty":
            mock.exists.return_value = True
            mock.get_children.return_value = []
            mock.start.return_value = None
            mock.stop.return_value = None
        elif scenario == "error":
            mock.exists.return_value = True
            mock.get_children.side_effect = ConnectionLoss("ZooKeeper error")
            mock.start.return_value = None
            mock.stop.return_value = None
        elif scenario == "connection_error":
            mock.start.side_effect = ConnectionLoss("ZooKeeper connection error")
            mock.stop.return_value = None

        return mock

    scenario = getattr(request, "param", "success")
    mock_client = _create_client(scenario)

    with patch("solr_mcp.solr.zookeeper.KazooClient", return_value=mock_client):
        yield _create_client


@pytest.fixture
def provider(mock_kazoo_client_factory):
    """Create ZooKeeperCollectionProvider instance with mocked dependencies."""
    from solr_mcp.solr.zookeeper import ZooKeeperCollectionProvider

    provider = ZooKeeperCollectionProvider(hosts=["localhost:2181"])
    # The KazooClient is already mocked via the factory fixture
    return provider
