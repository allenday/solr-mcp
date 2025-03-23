"""ZooKeeper collection provider for Solr."""

import logging
from typing import List, Optional, Union

import anyio
from kazoo.client import KazooClient
from kazoo.exceptions import NoNodeError, ConnectionLoss, KazooException
from loguru import logger

from solr_mcp.solr.exceptions import ConnectionError
from solr_mcp.solr.interfaces import CollectionProvider

logger = logging.getLogger(__name__)

class ZooKeeperCollectionProvider(CollectionProvider):
    """Collection provider that uses ZooKeeper to discover collections."""

    def __init__(self, hosts: Union[str, List[str]], default_collection: str | None = None):
        """Initialize provider.

        Args:
            hosts: Comma-separated list of ZooKeeper hosts or list of hosts
            default_collection: Default collection name
        """
        super().__init__()
        if isinstance(hosts, str):
            self.hosts = [h.strip() for h in hosts.split(',')]
        else:
            self.hosts = hosts
        self.default_collection = default_collection
        self.zk = None
        self._connected = False

    def connect(self):
        """Connect to ZooKeeper and verify collections path exists."""
        try:
            if not self.zk:
                self.zk = KazooClient(hosts=",".join(self.hosts))
            
            self.zk.start()
            if not self.zk.exists("/collections"):
                raise ConnectionError("Collections path does not exist")
            self._connected = True
        except ConnectionLoss as e:
            raise ConnectionError(f"Failed to connect to ZooKeeper: {e}")

    async def list_collections(self) -> List[str]:
        """List available collections.

        Returns:
            List of collection names

        Raises:
            ConnectionError: If unable to list collections
        """
        try:
            if not self._connected:
                await anyio.to_thread.run_sync(self.connect)

            if not self.zk:
                raise ConnectionError("Not connected to ZooKeeper")

            return await anyio.to_thread.run_sync(self.zk.get_children, "/collections")
        except NoNodeError:
            return []
        except ConnectionLoss as e:
            raise ConnectionError(f"Failed to list collections: {e}")

    def close(self):
        """Close connection."""
        self.cleanup()

    def cleanup(self):
        """Clean up ZooKeeper connection."""
        if self.zk:
            zk = self.zk  # Keep reference for assertions
            self.zk = None
            self._connected = False
            zk.stop()
            zk.close() 