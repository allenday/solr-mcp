"""Field caching for SolrCloud client."""

import logging
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from solr_mcp.solr.constants import SYNTHETIC_SORT_FIELDS

logger = logging.getLogger(__name__)


class FieldCache:
    """Caches field information for Solr collections."""

    def __init__(self):
        """Initialize the FieldCache."""
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, collection: str) -> Optional[Dict[str, Any]]:
        """Get cached field information for a collection.

        Args:
            collection: Collection name

        Returns:
            Dict containing field information or None if not cached
        """
        if collection in self._cache:
            return self._cache[collection]
        return None

    def set(self, collection: str, field_info: Dict[str, Any]) -> None:
        """Cache field information for a collection.

        Args:
            collection: Collection name
            field_info: Field information to cache
        """
        self._cache[collection] = {**field_info, "last_updated": time.time()}

    def is_stale(self, collection: str, max_age: float = 300.0) -> bool:
        """Check if cached field information is stale.

        Args:
            collection: Collection name
            max_age: Maximum age in seconds before cache is considered stale

        Returns:
            True if cache is stale or missing, False otherwise
        """
        if collection not in self._cache:
            return True

        last_updated = self._cache[collection].get("last_updated", 0)
        return (time.time() - last_updated) > max_age

    def get_or_default(self, collection: str) -> Dict[str, Any]:
        """Get cached field information or return defaults.

        Args:
            collection: Collection name

        Returns:
            Dict containing field information (cached or default)
        """
        if collection in self._cache:
            return self._cache[collection]

        # Return safe defaults
        return {
            "searchable_fields": ["_text_"],
            "sortable_fields": {"score": SYNTHETIC_SORT_FIELDS["score"]},
            "last_updated": time.time(),
        }

    def clear(self, collection: Optional[str] = None) -> None:
        """Clear cached field information.

        Args:
            collection: Collection name to clear, or None to clear all
        """
        if collection:
            self._cache.pop(collection, None)
        else:
            self._cache.clear()

    def update(self, collection: str, field_info: Dict[str, Any]) -> None:
        """Update cached field information.

        Args:
            collection: Collection name
            field_info: Field information to update
        """
        if collection in self._cache:
            self._cache[collection].update(field_info)
            self._cache[collection]["last_updated"] = time.time()
        else:
            self.set(collection, field_info)
