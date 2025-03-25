"""Collection providers for SolrCloud."""

import logging
from typing import List, Optional

import anyio
import requests
from kazoo.client import KazooClient
from kazoo.exceptions import ConnectionLoss, NoNodeError

from solr_mcp.solr.exceptions import ConnectionError, SolrError
from solr_mcp.solr.interfaces import CollectionProvider

logger = logging.getLogger(__name__)


class HttpCollectionProvider(CollectionProvider):
    """Collection provider that uses Solr HTTP API to discover collections."""

    def __init__(self, base_url: str):
        """Initialize with Solr base URL.

        Args:
            base_url: Base URL for Solr instance (e.g., http://localhost:8983/solr)
        """
        self.base_url = base_url.rstrip("/")

    async def list_collections(self) -> List[str]:
        """List all available collections using Solr HTTP API.

        Returns:
            List of collection names

        Raises:
            SolrError: If unable to retrieve collections
        """
        try:
            response = requests.get(f"{self.base_url}/admin/collections?action=LIST")
            if response.status_code != 200:
                raise SolrError(f"Failed to list collections: {response.text}")

            collections = response.json().get("collections", [])
            return collections

        except Exception as e:
            raise SolrError(f"Failed to list collections: {str(e)}")

    async def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists.

        Args:
            collection: Name of the collection to check

        Returns:
            True if the collection exists, False otherwise

        Raises:
            SolrError: If unable to check collection existence
        """
        try:
            collections = await self.list_collections()
            return collection in collections
        except Exception as e:
            raise SolrError(f"Failed to check if collection exists: {str(e)}")


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
