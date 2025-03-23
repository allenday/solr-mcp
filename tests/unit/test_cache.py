"""Unit tests for FieldCache."""

import time
import pytest
from solr_mcp.solr.schema.cache import FieldCache
from solr_mcp.solr.constants import SYNTHETIC_SORT_FIELDS

@pytest.fixture
def field_cache():
    """Create FieldCache instance for testing."""
    return FieldCache()

@pytest.fixture
def sample_field_info():
    """Sample field information for testing."""
    return {
        "searchable_fields": ["title", "content"],
        "sortable_fields": {
            "id": {"directions": ["asc", "desc"], "default_direction": "asc"},
            "score": SYNTHETIC_SORT_FIELDS["score"]
        }
    }

@pytest.fixture
def mock_time_expired(mocker):
    """Mock time for expired cache test."""
    mock = mocker.patch("time.time")
    mock.side_effect = [100, 1100]  # First call returns 100, second call returns 1100 (1000 seconds later)
    return mock

@pytest.fixture
def mock_time_custom(mocker):
    """Mock time for custom max age test."""
    mock = mocker.patch("time.time")
    mock.side_effect = [100, 200]  # First call returns 100, second call returns 200 (100 seconds later)
    return mock

class TestFieldCache:
    """Test cases for FieldCache."""

    def test_init(self, field_cache):
        """Test FieldCache initialization."""
        assert field_cache._cache == {}

    def test_get_missing(self, field_cache):
        """Test getting non-existent cache entry."""
        assert field_cache.get("collection1") is None

    def test_get_existing(self, field_cache, sample_field_info):
        """Test getting existing cache entry."""
        field_cache.set("collection1", sample_field_info)
        cached = field_cache.get("collection1")
        assert cached is not None
        assert cached["searchable_fields"] == sample_field_info["searchable_fields"]
        assert cached["sortable_fields"] == sample_field_info["sortable_fields"]
        assert "last_updated" in cached

    def test_set(self, field_cache, sample_field_info):
        """Test setting cache entry."""
        field_cache.set("collection1", sample_field_info)
        assert "collection1" in field_cache._cache
        assert field_cache._cache["collection1"]["searchable_fields"] == sample_field_info["searchable_fields"]
        assert field_cache._cache["collection1"]["sortable_fields"] == sample_field_info["sortable_fields"]
        assert "last_updated" in field_cache._cache["collection1"]

    def test_is_stale_missing(self, field_cache):
        """Test stale check for non-existent cache entry."""
        assert field_cache.is_stale("collection1") is True

    def test_is_stale_fresh(self, field_cache, sample_field_info):
        """Test stale check for fresh cache entry."""
        field_cache.set("collection1", sample_field_info)
        assert field_cache.is_stale("collection1") is False

    def test_is_stale_expired(self, field_cache, sample_field_info, mock_time_expired):
        """Test stale check for expired cache entry."""
        field_cache.set("collection1", sample_field_info)
        # First call to time.time() returns 100, second call returns 1100
        # Default max_age is 300 seconds, so cache should be stale
        assert field_cache.is_stale("collection1") is True
        assert mock_time_expired.call_count == 2

    def test_is_stale_custom_max_age(self, field_cache, sample_field_info, mock_time_custom):
        """Test stale check with custom max age."""
        field_cache.set("collection1", sample_field_info)
        # First call to time.time() returns 100, second call returns 200
        # With max_age=60, cache should be stale after 100 seconds
        assert field_cache.is_stale("collection1", max_age=60) is True
        assert mock_time_custom.call_count == 2

    def test_get_or_default_missing(self, field_cache):
        """Test getting defaults for non-existent cache entry."""
        result = field_cache.get_or_default("collection1")
        assert result["searchable_fields"] == ["_text_"]
        assert result["sortable_fields"] == {"score": SYNTHETIC_SORT_FIELDS["score"]}
        assert "last_updated" in result

    def test_get_or_default_existing(self, field_cache, sample_field_info):
        """Test getting existing cache entry instead of defaults."""
        field_cache.set("collection1", sample_field_info)
        result = field_cache.get_or_default("collection1")
        assert result["searchable_fields"] == sample_field_info["searchable_fields"]
        assert result["sortable_fields"] == sample_field_info["sortable_fields"]

    def test_clear_specific(self, field_cache, sample_field_info):
        """Test clearing specific collection from cache."""
        field_cache.set("collection1", sample_field_info)
        field_cache.set("collection2", sample_field_info)
        field_cache.clear("collection1")
        assert "collection1" not in field_cache._cache
        assert "collection2" in field_cache._cache

    def test_clear_all(self, field_cache, sample_field_info):
        """Test clearing entire cache."""
        field_cache.set("collection1", sample_field_info)
        field_cache.set("collection2", sample_field_info)
        field_cache.clear()
        assert len(field_cache._cache) == 0

    def test_update_existing(self, field_cache, sample_field_info):
        """Test updating existing cache entry."""
        field_cache.set("collection1", sample_field_info)
        update_info = {"searchable_fields": ["new_field"]}
        old_time = field_cache._cache["collection1"]["last_updated"]
        
        # Wait a moment to ensure timestamp changes
        time.sleep(0.001)
        field_cache.update("collection1", update_info)
        
        assert field_cache._cache["collection1"]["searchable_fields"] == ["new_field"]
        assert field_cache._cache["collection1"]["sortable_fields"] == sample_field_info["sortable_fields"]
        assert field_cache._cache["collection1"]["last_updated"] > old_time

    def test_update_missing(self, field_cache, sample_field_info):
        """Test updating non-existent cache entry."""
        update_info = {"searchable_fields": ["new_field"]}
        field_cache.update("collection1", update_info)
        assert field_cache._cache["collection1"]["searchable_fields"] == ["new_field"]
        assert "last_updated" in field_cache._cache["collection1"] 