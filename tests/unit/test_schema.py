"""Unit tests for FieldManager."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from solr_mcp.solr.schema import FieldManager, FieldCache
from solr_mcp.solr.exceptions import SchemaError
from solr_mcp.solr.constants import FIELD_TYPE_MAPPING, SYNTHETIC_SORT_FIELDS

@pytest.fixture
def mock_schema_requests():
    """Mock requests module for schema tests."""
    with patch("solr_mcp.solr.schema.fields.requests") as mock_requests:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "schema": {
                "fieldTypes": [
                    {
                        "name": "string",
                        "class": "solr.StrField",
                        "sortMissingLast": True
                    },
                    {
                        "name": "text_general",
                        "class": "solr.TextField",
                        "positionIncrementGap": "100"
                    },
                    {
                        "name": "knn_vector",
                        "class": "solr.DenseVectorField",
                        "vectorDimension": 768
                    }
                ],
                "fields": [
                    {
                        "name": "id",
                        "type": "string",
                        "required": True,
                        "multiValued": False
                    },
                    {
                        "name": "title",
                        "type": "text_general",
                        "multiValued": False
                    },
                    {
                        "name": "content",
                        "type": "text_general",
                        "multiValued": False
                    },
                    {
                        "name": "embedding",
                        "type": "knn_vector",
                        "multiValued": False
                    }
                ]
            }
        }
        mock_requests.get.return_value = mock_response
        yield mock_requests

@pytest.fixture
def field_manager():
    """Create FieldManager instance."""
    return FieldManager("http://localhost:8983/solr")

class TestFieldManager:
    """Test cases for FieldManager."""

    def test_init(self, field_manager):
        """Test FieldManager initialization."""
        assert field_manager.solr_base_url == "http://localhost:8983/solr"

    def test_get_schema_success(self, field_manager, mock_schema_requests):
        """Test successful schema retrieval."""
        schema = field_manager.get_schema("test_collection")
        assert "fieldTypes" in schema
        assert "fields" in schema

    def test_get_schema_error(self, field_manager, mock_schema_requests):
        """Test schema retrieval error handling."""
        mock_schema_requests.get.return_value.status_code = 500
        mock_schema_requests.get.return_value.text = "Internal Server Error"
        mock_schema_requests.get.return_value.raise_for_status.side_effect = Exception("Server error")
        
        with pytest.raises(SchemaError):
            field_manager.get_schema("test_collection")

    def test_get_field_types_success(self, field_manager, mock_schema_requests):
        """Test successful field types retrieval."""
        field_types = field_manager.get_field_types("test_collection")
        assert isinstance(field_types, dict)
        assert "id" in field_types
        assert field_types["id"] == "string"

    def test_get_field_types_cache(self, field_manager, mock_schema_requests):
        """Test field types caching."""
        # First call should make HTTP request
        field_manager.get_field_types("test_collection")
        initial_call_count = mock_schema_requests.get.call_count
        
        # Second call should use cache
        field_manager.get_field_types("test_collection")
        assert mock_schema_requests.get.call_count == initial_call_count

    def test_get_field_type_success(self, field_manager, mock_schema_requests):
        """Test getting single field type."""
        field_type = field_manager.get_field_type("test_collection", "id")
        assert field_type == "string"

    def test_get_field_type_nonexistent(self, field_manager, mock_schema_requests):
        """Test getting nonexistent field type."""
        with pytest.raises(SchemaError) as exc_info:
            field_manager.get_field_type("test_collection", "nonexistent")
        assert "not found" in str(exc_info.value)

    def test_validate_field_exists_success(self, field_manager, mock_schema_requests):
        """Test field existence validation."""
        # Should not raise exception
        field_manager.validate_field_exists("id", "test_collection")

    def test_validate_field_exists_error(self, field_manager, mock_schema_requests):
        """Test field existence validation error."""
        with pytest.raises(SchemaError):
            field_manager.validate_field_exists("nonexistent", "test_collection")

    def test_get_searchable_fields_success(self, field_manager, mock_schema_requests):
        """Test getting searchable fields from schema API."""
        searchable_fields = field_manager._get_searchable_fields("test_collection")
        assert isinstance(searchable_fields, list)
        assert "title" in searchable_fields
        assert "content" in searchable_fields
        assert "_text_" in searchable_fields

    def test_get_searchable_fields_fallback(self, field_manager, mock_schema_requests):
        """Test getting searchable fields with fallback."""
        # Configure mock to fail first call (schema API) but succeed second call (direct URL)
        mock_schema_requests.get.side_effect = [
            Exception("Schema API error"),
            Mock(
                json=lambda: {
                    "responseHeader": {
                        "params": {
                            "fl": "title,content"
                        }
                    }
                }
            )
        ]
        
        searchable_fields = field_manager._get_searchable_fields("test_collection")
        assert isinstance(searchable_fields, list)
        assert set(searchable_fields) == set(["title", "content", "_text_"])

    def test_get_sortable_fields_success(self, field_manager, mock_schema_requests):
        """Test getting sortable fields."""
        sortable_fields = field_manager._get_sortable_fields("test_collection")
        assert isinstance(sortable_fields, dict)
        assert "_docid_" in sortable_fields
        assert "score" in sortable_fields
        assert sortable_fields["_docid_"]["type"] == "numeric"
        assert sortable_fields["score"] == SYNTHETIC_SORT_FIELDS["score"]

    def test_get_sortable_fields_error(self, field_manager, mock_schema_requests):
        """Test getting sortable fields with error."""
        mock_schema_requests.get.side_effect = Exception("API error")
        
        sortable_fields = field_manager._get_sortable_fields("test_collection")
        assert isinstance(sortable_fields, dict)
        assert len(sortable_fields) == 1
        assert "score" in sortable_fields
        assert sortable_fields["score"] == SYNTHETIC_SORT_FIELDS["score"]

    def test_validate_fields_success(self, field_manager, mock_field_manager_methods):
        """Test validating fields."""
        with mock_field_manager_methods["patch_get_collection_fields"](field_manager):
            # Should not raise exception
            field_manager.validate_fields("test_collection", ["title", "id"])

    def test_validate_fields_error(self, field_manager, mock_field_manager_methods):
        """Test validating fields with invalid fields."""
        with mock_field_manager_methods["patch_get_collection_fields"](field_manager):
            with pytest.raises(SchemaError) as exc_info:
                field_manager.validate_fields("test_collection", ["nonexistent"])
            assert "Invalid fields" in str(exc_info.value)

    def test_validate_sort_fields_success(self, field_manager, mock_field_manager_methods):
        """Test validating sort fields."""
        with mock_field_manager_methods["patch_get_collection_fields"](field_manager):
            # Should not raise exception
            field_manager.validate_sort_fields("test_collection", ["id", "score"])

    def test_validate_sort_fields_error(self, field_manager, mock_field_manager_methods):
        """Test validating sort fields with invalid fields."""
        with mock_field_manager_methods["patch_get_collection_fields"](field_manager):
            with pytest.raises(SchemaError) as exc_info:
                field_manager.validate_sort_fields("test_collection", ["title"])
            assert "Fields not sortable" in str(exc_info.value)

    def test_get_field_info_success(self, field_manager, mock_schema_requests):
        """Test getting field information."""
        field_info = field_manager.get_field_info("test_collection")
        assert "searchable_fields" in field_info
        assert "sortable_fields" in field_info
        assert "id" in field_info["searchable_fields"]
        assert "title" in field_info["searchable_fields"]
        assert "content" in field_info["searchable_fields"]
        assert "_docid_" in field_info["sortable_fields"]
        assert "score" in field_info["sortable_fields"]

    def test_clear_cache_specific(self, field_manager, mock_schema_requests):
        """Test clearing cache for specific collection."""
        # Cache some data
        field_manager.get_schema("test_collection")
        field_manager.get_field_types("test_collection")
        initial_call_count = mock_schema_requests.get.call_count
        
        # Clear cache for test_collection
        field_manager.clear_cache("test_collection")
        
        # Should make new request
        field_manager.get_schema("test_collection")
        assert mock_schema_requests.get.call_count > initial_call_count

    def test_clear_cache_all(self, field_manager, mock_schema_requests):
        """Test clearing all cache."""
        # Cache some data
        field_manager.get_schema("test_collection")
        field_manager.get_schema("test_collection2")
        initial_call_count = mock_schema_requests.get.call_count
        
        # Clear all cache
        field_manager.clear_cache()
        
        # Should make new requests
        field_manager.get_schema("test_collection")
        field_manager.get_schema("test_collection2")
        assert mock_schema_requests.get.call_count > initial_call_count + 1

    def test_get_collection_fields_cached(self, field_manager):
        """Test getting collection fields from cache."""
        field_manager.cache = FieldCache()
        expected_info = {
            "searchable_fields": ["title", "content"],
            "sortable_fields": {"id": {}, "score": {}}
        }
        field_manager.cache.set("test_collection", expected_info)
        
        field_info = field_manager._get_collection_fields("test_collection")
        
        # Remove last_updated for comparison since it's dynamic
        field_info.pop("last_updated", None)
        assert field_info == expected_info

    def test_get_collection_fields_error_with_cache(self, field_manager, mock_field_manager_methods):
        """Test getting collection fields with error and cache fallback."""
        field_manager.cache = FieldCache()
        with mock_field_manager_methods["patch_get_searchable_fields"](field_manager):
            field_info = field_manager._get_collection_fields("test_collection")
            assert "searchable_fields" in field_info
            assert "_text_" in field_info["searchable_fields"]
            assert "score" in field_info["sortable_fields"]

    def test_get_searchable_fields_direct_url_error(self, field_manager, mock_schema_requests):
        """Test getting searchable fields with both API and direct URL failing."""
        mock_schema_requests.get.side_effect = [
            Exception("Schema API error"),
            Exception("Direct URL error")
        ]
        
        searchable_fields = field_manager._get_searchable_fields("test_collection")
        assert set(searchable_fields) == set(["content", "title", "_text_"])

    def test_get_sortable_fields_empty_response(self, field_manager, mock_schema_requests):
        """Test getting sortable fields with empty response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"fields": []}
        mock_schema_requests.get.return_value = mock_response
        
        sortable_fields = field_manager._get_sortable_fields("test_collection")
        assert "score" in sortable_fields
        assert sortable_fields["score"] == SYNTHETIC_SORT_FIELDS["score"]

    def test_get_collection_fields_error_no_cache(self, field_manager, mock_field_manager_methods):
        """Test getting collection fields with error and no cache."""
        field_manager.cache = FieldCache()
        with mock_field_manager_methods["patch_get_searchable_fields"](field_manager):
            field_info = field_manager._get_collection_fields("test_collection")
            assert "searchable_fields" in field_info
            assert "_text_" in field_info["searchable_fields"]
            assert "score" in field_info["sortable_fields"]

    def test_get_searchable_fields_schema_error(self, field_manager, mock_schema_requests):
        """Test getting searchable fields with schema error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Schema not found"
        mock_schema_requests.get.return_value = mock_response
        
        searchable_fields = field_manager._get_searchable_fields("test_collection")
        assert set(searchable_fields) == set(["content", "title", "_text_"])

    def test_get_searchable_fields_empty_response(self, field_manager, mock_schema_requests):
        """Test getting searchable fields with empty response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"fields": []}
        mock_schema_requests.get.return_value = mock_response
        
        searchable_fields = field_manager._get_searchable_fields("test_collection")
        assert set(searchable_fields) == set(["content", "title", "_text_"])

    def test_get_collection_fields_error_with_cache_fallback(self, field_manager, mock_schema_requests):
        """Test getting collection fields with error and cache fallback."""
        field_manager.cache = FieldCache()
        expected_info = {
            "searchable_fields": ["title", "content"],
            "sortable_fields": {"id": {}, "score": {}}
        }
        field_manager.cache.set("test_collection", expected_info)
        
        mock_schema_requests.get.side_effect = Exception("API error")
        
        field_info = field_manager._get_collection_fields("test_collection")
        field_info.pop("last_updated", None)
        assert field_info == expected_info