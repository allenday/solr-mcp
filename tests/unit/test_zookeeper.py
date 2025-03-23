"""Unit tests for ZooKeeper collection provider."""

import pytest
from solr_mcp.solr.exceptions import ConnectionError
from solr_mcp.solr.zookeeper import ZooKeeperCollectionProvider

class TestZooKeeperCollectionProvider:
    """Test ZooKeeperCollectionProvider."""

    def test_init(self):
        """Test initialization."""
        provider = ZooKeeperCollectionProvider(hosts=["localhost:2181"])
        assert provider.hosts == ["localhost:2181"]
        assert provider.zk is None

    @pytest.mark.parametrize("mock_kazoo_client", ["success"], indirect=True)
    def test_connect_success(self, provider):
        """Test successful connection to ZooKeeper."""
        provider.connect()
        assert provider.zk is not None
        provider.zk.start.assert_called_once()

    @pytest.mark.parametrize("mock_kazoo_client", ["no_collections"], indirect=True)
    def test_connect_no_collections_path(self, provider):
        """Test connection when /collections path doesn't exist."""
        with pytest.raises(ConnectionError):
            provider.connect()

    @pytest.mark.parametrize("mock_kazoo_client", ["connection_error"], indirect=True)
    def test_connect_error(self, provider):
        """Test connection error handling."""
        with pytest.raises(ConnectionError):
            provider.connect()

    @pytest.mark.parametrize("mock_kazoo_client", ["success"], indirect=True)
    def test_list_collections_success(self, provider):
        """Test successful collection listing."""
        provider.connect()
        collections = provider.list_collections()
        assert collections == ["collection1", "collection2"]

    @pytest.mark.parametrize("mock_kazoo_client", ["empty"], indirect=True)
    def test_list_collections_no_collections(self, provider):
        """Test listing collections when none exist."""
        provider.connect()
        collections = provider.list_collections()
        assert collections == []

    @pytest.mark.parametrize("mock_kazoo_client", ["error"], indirect=True)
    def test_list_collections_error(self, provider):
        """Test error handling in list_collections."""
        provider.connect()
        with pytest.raises(ConnectionError):
            provider.list_collections()

    @pytest.mark.parametrize("mock_kazoo_client", ["success"], indirect=True)
    def test_cleanup(self, provider):
        """Test cleanup."""
        provider.connect()
        mock_zk = provider.zk  # Store reference to mock
        provider.close()
        mock_zk.stop.assert_called_once()
        mock_zk.close.assert_called_once()
        assert provider.zk is None

    def test_cleanup_no_zk(self):
        """Test cleanup when no ZooKeeper connection exists."""
        provider = ZooKeeperCollectionProvider(hosts=["localhost:2181"])
        provider.close()  # Should not raise any exceptions 