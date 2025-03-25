"""Unit tests for ZooKeeperCollectionProvider."""

import pytest
from kazoo.exceptions import NoNodeError, ConnectionLoss

from solr_mcp.solr.exceptions import ConnectionError
from solr_mcp.solr.zookeeper import ZooKeeperCollectionProvider

class TestZooKeeperCollectionProvider:
    """Test ZooKeeperCollectionProvider."""

    def test_init(self, mock_kazoo_client_factory, mock_kazoo_client):
        """Test initialization."""
        hosts = ["host1:2181", "host2:2181"]
        provider = ZooKeeperCollectionProvider(hosts)
        
        assert provider.hosts == hosts
        assert provider.zk is not None
        mock_kazoo_client_factory.assert_called_once_with(hosts="host1:2181,host2:2181")
        mock_kazoo_client.start.assert_called_once()
        mock_kazoo_client.exists.assert_called_once_with("/collections")

    def test_connect_success(self, mock_kazoo_client_factory, mock_kazoo_client):
        """Test successful connection."""
        provider = ZooKeeperCollectionProvider(["host1:2181"])
        # First connection happens in __init__
        mock_kazoo_client_factory.assert_called_once_with(hosts="host1:2181")
        mock_kazoo_client.start.assert_called_once()
        mock_kazoo_client.exists.assert_called_once_with("/collections")

        # Reset mock and test reconnecting after cleanup
        mock_kazoo_client_factory.reset_mock()
        mock_kazoo_client.exists.return_value = True
        provider.cleanup()
        provider.connect()

        mock_kazoo_client_factory.assert_called_once_with(hosts="host1:2181")
        mock_kazoo_client.start.assert_called_once()
        mock_kazoo_client.exists.assert_called_once_with("/collections")

    @pytest.mark.parametrize('mock_kazoo_client', ['no_collections'], indirect=True)
    def test_connect_no_collections(self, mock_kazoo_client_factory, mock_kazoo_client):
        """Test connection when /collections path doesn't exist."""
        with pytest.raises(ConnectionError, match="ZooKeeper /collections path does not exist"):
            provider = ZooKeeperCollectionProvider(["host1:2181"])

        mock_kazoo_client_factory.assert_called_once_with(hosts="host1:2181")

    @pytest.mark.parametrize('mock_kazoo_client', ['connection_error'], indirect=True)
    def test_connect_error(self, mock_kazoo_client_factory, mock_kazoo_client):
        """Test connection error."""
        with pytest.raises(ConnectionError, match="Failed to connect to ZooKeeper"):
            provider = ZooKeeperCollectionProvider(["host1:2181"])

        mock_kazoo_client_factory.assert_called_once_with(hosts="host1:2181")

    @pytest.mark.asyncio
    async def test_list_collections_success(self, provider):
        """Test successful collection listing."""
        collections = await provider.list_collections()
        assert collections == ["collection1", "collection2"]
        provider.zk.get_children.assert_called_once_with("/collections")

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mock_kazoo_client', ['empty'], indirect=True)
    async def test_list_collections_empty(self, provider):
        """Test empty collection listing."""
        collections = await provider.list_collections()
        assert collections == []

    @pytest.mark.asyncio
    async def test_list_collections_not_connected(self, provider):
        """Test listing collections when not connected."""
        provider.cleanup()  # Force disconnect
        with pytest.raises(ConnectionError, match="Not connected to ZooKeeper"):
            await provider.list_collections()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mock_kazoo_client', ['error'], indirect=True)
    async def test_list_collections_connection_loss(self, provider):
        """Test connection loss during collection listing."""
        with pytest.raises(ConnectionError, match="Lost connection to ZooKeeper"):
            await provider.list_collections()

    def test_cleanup(self, provider):
        """Test cleanup."""
        # Store reference to zk client before cleanup
        zk = provider.zk
        provider.cleanup()
        zk.stop.assert_called_once()
        zk.close.assert_called_once()
        assert provider.zk is None

    def test_cleanup_error(self, mock_kazoo_client, provider):
        """Test cleanup with error."""
        # Store reference to zk client before cleanup
        zk = provider.zk
        mock_kazoo_client.stop.side_effect = Exception("Cleanup error")
        provider.cleanup()  # Should not raise exception
        assert provider.zk is None 