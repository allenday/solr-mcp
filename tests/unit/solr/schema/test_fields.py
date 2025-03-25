"""Tests for solr_mcp.solr.schema.fields module."""

import json
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
import requests

from solr_mcp.solr.exceptions import SchemaError
from solr_mcp.solr.schema.fields import FieldManager


@pytest.fixture
def field_manager():
    """Create a FieldManager instance."""
    return FieldManager("http://localhost:8983")


@pytest.fixture
def mock_schema_response() -> Dict[str, Any]:
    """Create a mock schema response."""
    return {
        "schema": {
            "name": "test",
            "version": 1.6,
            "uniqueKey": "id",
            "fieldTypes": [
                {
                    "name": "text_general",
                    "class": "solr.TextField",
                    "positionIncrementGap": "100",
                },
                {"name": "string", "class": "solr.StrField", "sortMissingLast": True},
            ],
            "fields": [
                {
                    "name": "id",
                    "type": "string",
                    "indexed": True,
                    "stored": True,
                    "required": True,
                    "docValues": True,
                },
                {
                    "name": "title",
                    "type": "text_general",
                    "indexed": True,
                    "stored": True,
                    "docValues": False,
                },
                {
                    "name": "sort_field",
                    "type": "string",
                    "indexed": True,
                    "stored": True,
                    "docValues": True,
                },
            ],
            "copyFields": [{"source": "title", "dest": "_text_"}],
        }
    }


@pytest.fixture
def mock_direct_response():
    """Create a mock direct response."""
    return {"responseHeader": {"params": {"fl": "title,content,_text_"}}}


def test_get_field_info_success(field_manager, mock_schema_response):
    """Test getting field info."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ):
        field_info = field_manager.get_field_info("test_collection")
        assert "searchable_fields" in field_info
        assert "sortable_fields" in field_info
        assert set(field_info["searchable_fields"]) == {"id", "title", "sort_field"}
        assert set(field_info["sortable_fields"].keys()) >= {
            "id",
            "sort_field",
            "_docid_",
            "score",
        }


def test_get_searchable_fields_schema_api(field_manager, mock_schema_response):
    """Test getting searchable fields using schema API."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_schema_response

        fields = field_manager._get_searchable_fields("test_collection")

        assert "title" in fields
        assert "content" in fields
        assert "_text_" in fields


def test_get_searchable_fields_direct_url(field_manager, mock_direct_response):
    """Test getting searchable fields using direct URL."""
    with patch("requests.get") as mock_get:
        # First call fails to trigger fallback
        mock_get.side_effect = [
            requests.exceptions.RequestException("Schema API error"),
            Mock(status_code=200, json=lambda: mock_direct_response),
        ]

        fields = field_manager._get_searchable_fields("test_collection")

        assert "title" in fields
        assert "content" in fields
        assert "_text_" in fields


def test_get_searchable_fields_fallback(field_manager):
    """Test getting searchable fields with both methods failing."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = [
            requests.exceptions.RequestException("Schema API error"),
            requests.exceptions.RequestException("Direct URL error"),
        ]

        fields = field_manager._get_searchable_fields("test_collection")

        # Should return default fields
        assert fields == ["content", "title", "_text_"]


def test_get_searchable_fields_skip_special(field_manager):
    """Test skipping special fields except _text_."""
    schema_response = {
        "fields": [
            {"name": "_version_", "type": "long"},
            {"name": "_text_", "type": "text_general"},
            {"name": "_root_", "type": "string"},
            {"name": "title", "type": "text_general"},
        ]
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = schema_response

        fields = field_manager._get_searchable_fields("test_collection")

        assert "_text_" in fields
        assert "title" in fields
        assert "_version_" not in fields
        assert "_root_" not in fields


def test_get_searchable_fields_text_types(field_manager):
    """Test identifying text type fields."""
    schema_response = {
        "fields": [
            {"name": "text_field", "type": "text_general"},
            {"name": "string_field", "type": "string"},
            {"name": "custom_text", "type": "custom_text_type"},
            {"name": "numeric_field", "type": "long"},
        ]
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = schema_response

        fields = field_manager._get_searchable_fields("test_collection")

        assert "text_field" in fields
        assert "string_field" in fields
        assert "custom_text" in fields
        assert "numeric_field" not in fields


def test_get_field_info_specific_field(field_manager, mock_schema_response):
    """Test getting field info for specific field."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ):
        field_info = field_manager.get_field_info("test_collection", "title")
        assert field_info["type"] == "text_general"
        assert field_info["searchable"] is True


def test_get_field_info_nonexistent_field(field_manager, mock_schema_response):
    """Test getting field info for non-existent field."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ):
        with pytest.raises(
            SchemaError,
            match="Field nonexistent not found in collection test_collection",
        ):
            field_manager.get_field_info("test_collection", "nonexistent")


def test_get_schema_cached(field_manager, mock_schema_response):
    """Test schema caching."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_schema_response

        # First call should make the request
        schema1 = field_manager.get_schema("test_collection")
        assert mock_get.call_count == 1

        # Second call should use cache
        schema2 = field_manager.get_schema("test_collection")
        assert mock_get.call_count == 1
        assert schema1 == schema2


def test_get_schema_invalid_response(field_manager):
    """Test handling of invalid schema response."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"invalid": "response"}

        with pytest.raises(SchemaError, match="Invalid schema response"):
            field_manager.get_schema("test_collection")


def test_get_field_info_no_fields(field_manager):
    """Test getting field info with no fields in schema."""
    with patch.object(field_manager, "get_schema", return_value={"schema": {}}):
        field_info = field_manager.get_field_info("test_collection")
        assert field_info["searchable_fields"] == []
        assert field_info["sortable_fields"] == {
            "_docid_": {
                "type": "numeric",
                "searchable": False,
                "directions": ["asc", "desc"],
                "default_direction": "asc",
            },
            "score": {
                "type": "numeric",
                "searchable": True,
                "directions": ["asc", "desc"],
                "default_direction": "desc",
            },
        }


def test_get_field_info_invalid_field_def(field_manager):
    """Test getting field info with invalid field definition."""
    schema = {"schema": {"fields": [{"invalid": "field"}]}}
    with patch.object(field_manager, "get_schema", return_value=schema):
        field_info = field_manager.get_field_info("test_collection")
        assert field_info["searchable_fields"] == []
        assert set(field_info["sortable_fields"].keys()) == {"_docid_", "score"}


def test_get_field_info_with_copy_fields(field_manager, mock_schema_response):
    """Test getting field info with copy fields."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ):
        field_info = field_manager.get_field_info("test_collection")
        assert "title" in field_info["searchable_fields"]
        assert "_text_" not in field_info["searchable_fields"]


def test_get_field_types(field_manager, mock_schema_response):
    """Test getting field types."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ):
        field_types = field_manager.get_field_types("test_collection")
        assert field_types["text_general"] == "text_general"
        assert field_types["string"] == "string"
        assert field_types["title"] == "text_general"
        assert field_types["id"] == "string"


def test_get_field_type(field_manager, mock_schema_response):
    """Test getting field type for a specific field."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ):
        field_type = field_manager.get_field_type("test_collection", "title")
        assert field_type == "text_general"


def test_get_field_type_not_found(field_manager, mock_schema_response):
    """Test getting field type for a non-existent field."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ):
        with pytest.raises(SchemaError, match="Field not found: nonexistent"):
            field_manager.get_field_type("test_collection", "nonexistent")


def test_validate_field_exists_success(field_manager):
    """Test validating existing field."""
    with patch.object(field_manager, "get_field_info") as mock_get_info:
        mock_get_info.return_value = {"searchable_fields": ["title", "id"]}
        assert field_manager.validate_field_exists("title", "test_collection") is True


def test_validate_field_exists_wildcard(field_manager):
    """Test validating wildcard field."""
    with patch.object(field_manager, "get_field_info") as mock_get_info:
        mock_get_info.return_value = {"searchable_fields": ["title", "id"]}
        assert field_manager.validate_field_exists("*", "test_collection") is True


def test_validate_field_exists_not_found(field_manager):
    """Test validating non-existent field."""
    with patch.object(field_manager, "get_field_info") as mock_get_info:
        mock_get_info.return_value = {"searchable_fields": ["title", "id"]}
        with pytest.raises(
            SchemaError,
            match="Field nonexistent not found in collection test_collection",
        ):
            field_manager.validate_field_exists("nonexistent", "test_collection")


def test_validate_field_exists_error(field_manager):
    """Test field validation with error."""
    with patch.object(
        field_manager, "get_field_info", side_effect=Exception("Test error")
    ):
        with pytest.raises(
            SchemaError, match="Error validating field test: Test error"
        ):
            field_manager.validate_field_exists("test", "test_collection")


def test_validate_sort_field_success(field_manager):
    """Test validating sortable field."""
    with patch.object(field_manager, "get_field_info") as mock_get_info:
        mock_get_info.return_value = {"sortable_fields": {"sort_field": {}, "id": {}}}
        assert (
            field_manager.validate_sort_field("sort_field", "test_collection") is True
        )


def test_validate_sort_field_not_found(field_manager):
    """Test validating non-sortable field."""
    with patch.object(field_manager, "get_field_info") as mock_get_info:
        mock_get_info.return_value = {"sortable_fields": {"sort_field": {}, "id": {}}}
        with pytest.raises(
            SchemaError,
            match="Field title is not sortable in collection test_collection",
        ):
            field_manager.validate_sort_field("title", "test_collection")


def test_validate_sort_field_error(field_manager):
    """Test sort field validation with error."""
    with patch.object(
        field_manager, "get_field_info", side_effect=Exception("Test error")
    ):
        with pytest.raises(
            SchemaError, match="Error validating sort field test: Test error"
        ):
            field_manager.validate_sort_field("test", "test_collection")


def test_get_field_types_cached(field_manager, mock_schema_response):
    """Test field types caching."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ) as mock_get_schema:
        # First call should hit the API
        field_types1 = field_manager.get_field_types("test_collection")
        # Second call should use cache
        field_types2 = field_manager.get_field_types("test_collection")

        assert field_types1 == field_types2
        mock_get_schema.assert_called_once()


def test_clear_cache_specific_collection(field_manager, mock_schema_response):
    """Test clearing cache for specific collection."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ) as mock_get_schema:
        # Populate cache
        field_manager.get_field_types("test_collection")
        field_manager.get_field_types("other_collection")

        # Clear specific collection
        field_manager.clear_cache("test_collection")

        # Should hit API again for cleared collection
        field_manager.get_field_types("test_collection")
        # Should use cache for other collection
        field_manager.get_field_types("other_collection")

        assert mock_get_schema.call_count == 3


def test_clear_cache_all(field_manager, mock_schema_response):
    """Test clearing entire cache."""
    with patch.object(
        field_manager, "get_schema", return_value=mock_schema_response["schema"]
    ) as mock_get_schema:
        # Populate cache
        field_manager.get_field_types("test_collection")
        field_manager.get_field_types("other_collection")

        # Clear all cache
        field_manager.clear_cache()

        # Should hit API again for both collections
        field_manager.get_field_types("test_collection")
        field_manager.get_field_types("other_collection")

        assert mock_get_schema.call_count == 4


def test_validate_collection_exists_success(field_manager, mock_schema_response):
    """Test validating existing collection."""
    with patch.object(field_manager, "get_schema") as mock_get_schema:
        mock_get_schema.return_value = mock_schema_response["schema"]
        assert field_manager.validate_collection_exists("test_collection") is True


def test_validate_collection_exists_error(field_manager):
    """Test collection validation with error."""
    with patch.object(field_manager, "get_schema", side_effect=Exception("Test error")):
        with pytest.raises(
            SchemaError, match="Error validating collection: Test error"
        ):
            field_manager.validate_collection_exists("test_collection")


def test_validate_fields_success(field_manager):
    """Test validating multiple fields."""
    with patch.object(field_manager, "_get_collection_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "searchable_fields": ["title", "id"],
            "sortable_fields": {"id": {}, "sort_field": {}},
        }
        field_manager.validate_fields("test_collection", ["title", "id"])
        mock_get_fields.assert_called_once_with("test_collection")


def test_validate_fields_error(field_manager):
    """Test validating multiple fields with error."""
    with patch.object(field_manager, "_get_collection_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "searchable_fields": ["title"],
            "sortable_fields": {"sort_field": {}},
        }
        with pytest.raises(
            SchemaError,
            match="Invalid fields for collection test_collection: nonexistent",
        ):
            field_manager.validate_fields("test_collection", ["title", "nonexistent"])


def test_validate_sort_fields_success(field_manager):
    """Test validating multiple sort fields."""
    with patch.object(field_manager, "_get_collection_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "searchable_fields": ["title", "id"],
            "sortable_fields": {"id": {}, "sort_field": {}},
        }
        field_manager.validate_sort_fields("test_collection", ["sort_field", "id"])
        mock_get_fields.assert_called_once_with("test_collection")


def test_validate_sort_fields_error(field_manager):
    """Test validating multiple sort fields with error."""
    with patch.object(field_manager, "_get_collection_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "searchable_fields": ["title"],
            "sortable_fields": {"sort_field": {}},
        }
        with pytest.raises(
            SchemaError,
            match="Fields not sortable in collection test_collection: title",
        ):
            field_manager.validate_sort_fields(
                "test_collection", ["sort_field", "title"]
            )
