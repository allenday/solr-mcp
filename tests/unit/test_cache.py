"""Unit tests for FieldCache."""

import time
from unittest.mock import patch

import pytest

from solr_mcp.solr.constants import SYNTHETIC_SORT_FIELDS
from solr_mcp.solr.schema.cache import FieldCache

# Sample data for testing
SAMPLE_FIELD_INFO = {
    "searchable_fields": ["title", "content"],
    "sortable_fields": {
        "id": {"directions": ["asc", "desc"], "default_direction": "asc"},
        "score": SYNTHETIC_SORT_FIELDS["score"],
    },
}


@pytest.fixture
def field_cache():
    """Create FieldCache instance for testing."""
    return FieldCache()


@pytest.mark.parametrize(
    "collection,info",
    [
        ("collection1", SAMPLE_FIELD_INFO),
        (
            "test_collection",
            {
                "searchable_fields": ["title"],
                "sortable_fields": {
                    "id": {"directions": ["asc"], "default_direction": "asc"}
                },
            },
        ),
    ],
)
class TestFieldCacheBasic:
    """Test cases for basic FieldCache operations."""

    def test_get_existing(self, field_cache, collection, info):
        """Test getting existing cache entry."""
        field_cache.set(collection, info)
        cached = field_cache.get(collection)
        assert cached is not None
        assert cached["searchable_fields"] == info["searchable_fields"]
        assert cached["sortable_fields"] == info["sortable_fields"]
        assert "last_updated" in cached

    def test_set(self, field_cache, collection, info):
        """Test setting cache entry."""
        field_cache.set(collection, info)
        assert collection in field_cache._cache
        assert (
            field_cache._cache[collection]["searchable_fields"]
            == info["searchable_fields"]
        )
        assert (
            field_cache._cache[collection]["sortable_fields"] == info["sortable_fields"]
        )
        assert "last_updated" in field_cache._cache[collection]

    def test_get_or_default_existing(self, field_cache, collection, info):
        """Test getting existing cache entry instead of defaults."""
        field_cache.set(collection, info)
        result = field_cache.get_or_default(collection)
        assert result["searchable_fields"] == info["searchable_fields"]
        assert result["sortable_fields"] == info["sortable_fields"]

    def test_is_stale_fresh(self, field_cache, collection, info):
        """Test stale check for fresh cache entry."""
        field_cache.set(collection, info)
        assert field_cache.is_stale(collection) is False


class TestFieldCacheOperations:
    """Test cases for FieldCache operations."""

    def test_init(self, field_cache):
        """Test FieldCache initialization."""
        assert field_cache._cache == {}

    def test_get_missing(self, field_cache):
        """Test getting non-existent cache entry."""
        assert field_cache.get("collection1") is None

    def test_is_stale_missing(self, field_cache):
        """Test stale check for non-existent cache entry."""
        assert field_cache.is_stale("collection1") is True

    @pytest.mark.parametrize(
        "time_values,max_age,expected_stale",
        [
            # Format: (initial_time, check_time), max_age, expected_stale
            ((100, 1100), 300, True),  # Default max_age=300, elapsed=1000
            ((100, 200), 60, True),  # Custom max_age=60, elapsed=100
            ((100, 150), 60, False),  # Custom max_age=60, elapsed=50
        ],
    )
    def test_is_stale_with_time(
        self, field_cache, patch_module, time_values, max_age, expected_stale
    ):
        """Test stale check with various time scenarios."""
        # Use the factory fixture to create a patch
        with patch_module("time.time", side_effect=time_values):
            field_cache.set("collection1", SAMPLE_FIELD_INFO)
            if max_age == 300:  # Default max_age
                assert field_cache.is_stale("collection1") is expected_stale
            else:
                assert (
                    field_cache.is_stale("collection1", max_age=max_age)
                    is expected_stale
                )

    def test_get_or_default_missing(self, field_cache):
        """Test getting defaults for non-existent cache entry."""
        result = field_cache.get_or_default("collection1")
        assert result["searchable_fields"] == ["_text_"]
        assert result["sortable_fields"] == {"score": SYNTHETIC_SORT_FIELDS["score"]}
        assert "last_updated" in result

    @pytest.mark.parametrize(
        "collections",
        [["collection1"], ["collection1", "collection2"], ["test1", "test2", "test3"]],
    )
    def test_clear_operations(self, field_cache, collections):
        """Test clearing operations with different collection sets."""
        # Setup - add all collections to cache
        for collection in collections:
            field_cache.set(collection, SAMPLE_FIELD_INFO)

        # Verify all collections are in cache
        for collection in collections:
            assert collection in field_cache._cache

        if len(collections) > 1:
            # Test clearing specific collection
            field_cache.clear(collections[0])
            assert collections[0] not in field_cache._cache
            for collection in collections[1:]:
                assert collection in field_cache._cache

            # Test clearing all collections
            field_cache.clear()
            assert len(field_cache._cache) == 0
        else:
            # Just test clear all for single collection
            field_cache.clear()
            assert len(field_cache._cache) == 0

    @pytest.mark.parametrize(
        "update_info",
        [
            {"searchable_fields": ["new_field"]},
            {
                "sortable_fields": {
                    "new_id": {"directions": ["asc"], "default_direction": "asc"}
                }
            },
            {"searchable_fields": ["field1", "field2"], "sortable_fields": {}},
        ],
    )
    def test_update_operations(self, field_cache, update_info):
        """Test update operations with different update payloads."""
        # Test updating non-existent entry
        field_cache.update("collection1", update_info)
        for key, value in update_info.items():
            assert field_cache._cache["collection1"][key] == value
        assert "last_updated" in field_cache._cache["collection1"]

        # Test updating existing entry
        field_cache.set("collection2", SAMPLE_FIELD_INFO)
        old_time = field_cache._cache["collection2"]["last_updated"]

        # Wait to ensure timestamp changes
        time.sleep(0.001)
        field_cache.update("collection2", update_info)

        # Verify updated fields
        for key, value in update_info.items():
            assert field_cache._cache["collection2"][key] == value

        # Verify non-updated fields retained original values
        for key in SAMPLE_FIELD_INFO:
            if key not in update_info:
                assert field_cache._cache["collection2"][key] == SAMPLE_FIELD_INFO[key]

        # Verify timestamp updated
        assert field_cache._cache["collection2"]["last_updated"] > old_time
