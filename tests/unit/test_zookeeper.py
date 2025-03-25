"""Unit tests for ZooKeeperCollectionProvider."""

from unittest.mock import MagicMock, patch

import pytest
from kazoo.exceptions import ConnectionLoss, NoNodeError

from solr_mcp.solr.exceptions import ConnectionError
from solr_mcp.solr.zookeeper import ZooKeeperCollectionProvider


class TestZooKeeperCollectionProvider:
    """Test ZooKeeperCollectionProvider."""

    def test_init(self):
        """Test initialization."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = True
            mock_factory.return_value = mock_client

            hosts = ["host1:2181", "host2:2181"]
            provider = ZooKeeperCollectionProvider(hosts)

            assert provider.hosts == hosts
            assert provider.zk is not None
            mock_factory.assert_called_once_with(hosts="host1:2181,host2:2181")
            mock_client.start.assert_called_once()
            mock_client.exists.assert_called_once_with("/collections")

    def test_connect_success(self):
        """Test successful connection."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = True
            mock_factory.return_value = mock_client

            # Create provider and test initial connection
            provider = ZooKeeperCollectionProvider(["host1:2181"])
            mock_factory.assert_called_once_with(hosts="host1:2181")
            mock_client.start.assert_called_once()
            mock_client.exists.assert_called_once_with("/collections")

            # Reset mocks and test reconnection after cleanup
            mock_factory.reset_mock()
            mock_client.reset_mock()

            # Create a new mock for reconnection
            mock_reconnect_client = MagicMock()
            mock_reconnect_client.exists.return_value = True
            mock_factory.return_value = mock_reconnect_client

            provider.cleanup()
            provider.connect()

            mock_factory.assert_called_once_with(hosts="host1:2181")
            mock_reconnect_client.start.assert_called_once()
            mock_reconnect_client.exists.assert_called_once_with("/collections")

    def test_connect_no_collections(self):
        """Test connection when /collections path doesn't exist."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = False
            mock_factory.return_value = mock_client

            with pytest.raises(
                ConnectionError, match="ZooKeeper /collections path does not exist"
            ):
                provider = ZooKeeperCollectionProvider(["host1:2181"])

    def test_connect_error(self):
        """Test connection error."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.start.side_effect = ConnectionLoss("ZooKeeper connection error")
            mock_factory.return_value = mock_client

            with pytest.raises(ConnectionError, match="Failed to connect to ZooKeeper"):
                provider = ZooKeeperCollectionProvider(["host1:2181"])

    @pytest.mark.asyncio
    async def test_list_collections_success(self):
        """Test successful collection listing."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = True
            mock_client.get_children.return_value = ["collection1", "collection2"]
            mock_factory.return_value = mock_client

            provider = ZooKeeperCollectionProvider(["localhost:2181"])
            collections = await provider.list_collections()

            assert collections == ["collection1", "collection2"]
            mock_client.get_children.assert_called_once_with("/collections")

    @pytest.mark.asyncio
    async def test_list_collections_empty(self):
        """Test empty collection listing."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = True
            mock_client.get_children.return_value = []
            mock_factory.return_value = mock_client

            provider = ZooKeeperCollectionProvider(["localhost:2181"])
            collections = await provider.list_collections()

            assert collections == []
            mock_client.get_children.assert_called_once_with("/collections")

    @pytest.mark.asyncio
    async def test_list_collections_not_connected(self):
        """Test listing collections when not connected."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = True
            mock_factory.return_value = mock_client

            provider = ZooKeeperCollectionProvider(["localhost:2181"])
            provider.cleanup()  # Force disconnect

            with pytest.raises(ConnectionError, match="Not connected to ZooKeeper"):
                await provider.list_collections()

    @pytest.mark.asyncio
    async def test_list_collections_connection_loss(self):
        """Test connection loss during collection listing."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = True
            mock_client.get_children.side_effect = ConnectionLoss("ZooKeeper error")
            mock_factory.return_value = mock_client

            provider = ZooKeeperCollectionProvider(["localhost:2181"])

            with pytest.raises(ConnectionError, match="Lost connection to ZooKeeper"):
                await provider.list_collections()

            mock_client.get_children.assert_called_once_with("/collections")

    def test_cleanup(self):
        """Test cleanup."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = True
            mock_factory.return_value = mock_client

            provider = ZooKeeperCollectionProvider(["localhost:2181"])
            provider.cleanup()

            mock_client.stop.assert_called_once()
            mock_client.close.assert_called_once()
            assert provider.zk is None

    def test_cleanup_error(self):
        """Test cleanup with error."""
        with patch("solr_mcp.solr.zookeeper.KazooClient") as mock_factory:
            mock_client = MagicMock()
            mock_client.exists.return_value = True
            mock_client.stop.side_effect = Exception("Cleanup error")
            mock_factory.return_value = mock_client

            provider = ZooKeeperCollectionProvider(["localhost:2181"])
            provider.cleanup()  # Should not raise exception

            assert provider.zk is None
