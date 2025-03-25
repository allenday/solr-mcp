"""ZooKeeper-based collection provider."""

from typing import List

import anyio
from kazoo.client import KazooClient
from kazoo.exceptions import ConnectionLoss, NoNodeError

from solr_mcp.solr.exceptions import ConnectionError
from solr_mcp.solr.interfaces import CollectionProvider


class ZooKeeperCollectionProvider(CollectionProvider):
    """Collection provider that uses ZooKeeper to discover collections."""

    def __init__(self, hosts: List[str]):
        """Initialize with ZooKeeper hosts.

        Args:
            hosts: List of ZooKeeper hosts in format host:port
        """
        self.hosts = hosts
        self.zk = None
        self.connect()

    def connect(self):
        """Connect to ZooKeeper and verify /collections path exists."""
        try:
            self.zk = KazooClient(hosts=",".join(self.hosts))
            self.zk.start()

            # Check if /collections path exists
            if not self.zk.exists("/collections"):
                raise ConnectionError("ZooKeeper /collections path does not exist")

        except ConnectionLoss as e:
            raise ConnectionError(f"Failed to connect to ZooKeeper: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Error connecting to ZooKeeper: {str(e)}")

    def cleanup(self):
        """Clean up ZooKeeper connection."""
        if self.zk:
            try:
                self.zk.stop()
                self.zk.close()
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self.zk = None

    async def list_collections(self) -> List[str]:
        """List available collections from ZooKeeper.

        Returns:
            List of collection names

        Raises:
            ConnectionError: If there is an error communicating with ZooKeeper
        """
        try:
            if not self.zk:
                raise ConnectionError("Not connected to ZooKeeper")

            collections = await anyio.to_thread.run_sync(
                self.zk.get_children, "/collections"
            )
            return collections

        except NoNodeError:
            return []  # No collections exist yet
        except ConnectionLoss as e:
            raise ConnectionError(f"Lost connection to ZooKeeper: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Error listing collections: {str(e)}")

    async def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists in ZooKeeper.

        Args:
            collection: Name of the collection to check

        Returns:
            True if the collection exists, False otherwise

        Raises:
            ConnectionError: If there is an error communicating with ZooKeeper
        """
        try:
            if not self.zk:
                raise ConnectionError("Not connected to ZooKeeper")

            # Check for collection in ZooKeeper
            collection_path = f"/collections/{collection}"
            exists = await anyio.to_thread.run_sync(self.zk.exists, collection_path)
            return exists is not None

        except ConnectionLoss as e:
            raise ConnectionError(f"Lost connection to ZooKeeper: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Error checking collection existence: {str(e)}")
