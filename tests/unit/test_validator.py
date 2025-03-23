"""Unit tests for QueryValidator."""

import pytest
from unittest.mock import Mock

from solr_mcp.solr.query.validator import QueryValidator
from solr_mcp.solr.exceptions import QueryError

@pytest.fixture
def mock_field_manager():
    """Mock FieldManager for testing."""
    mock = Mock()
    mock.get_field_types.return_value = {
        "id": "string",
        "title": "text_general",
        "content": "text_general",
        "embedding": "knn_vector"
    }
    mock.get_field_info.return_value = {
        "sortable_fields": {
            "id": {"directions": ["asc", "desc"], "default_direction": "asc"},
            "title": {"directions": ["asc", "desc"], "default_direction": "asc"}
        }
    }
    return mock

@pytest.fixture
def query_validator(mock_field_manager):
    """Create QueryValidator instance with mocked dependencies."""
    return QueryValidator(field_manager=mock_field_manager)

class TestQueryValidator:
    """Test cases for QueryValidator."""

    def test_init(self, query_validator, mock_field_manager):
        """Test QueryValidator initialization."""
        assert query_validator.field_manager == mock_field_manager

    def test_validate_fields_valid(self, query_validator):
        """Test validating valid fields."""
        fields = ["id", "title", "content"]
        # Should not raise any exceptions
        query_validator.validate_fields("collection1", fields)

    def test_validate_fields_invalid(self, query_validator):
        """Test validating invalid fields."""
        fields = ["id", "nonexistent_field"]
        with pytest.raises(QueryError) as exc_info:
            query_validator.validate_fields("collection1", fields)
        assert "Invalid field 'nonexistent_field'" in str(exc_info.value)

    def test_validate_fields_error_handling(self, query_validator, mock_field_manager):
        """Test error handling in validate_fields."""
        mock_field_manager.get_field_types.side_effect = Exception("Test error")
        with pytest.raises(QueryError) as exc_info:
            query_validator.validate_fields("collection1", ["id"])
        assert "Field validation error" in str(exc_info.value)

    def test_validate_sort_fields_valid(self, query_validator, mock_field_manager):
        """Test validating valid sort fields."""
        fields = ["id", "title"]
        # Should not raise any exceptions
        query_validator.validate_sort_fields("collection1", fields)

    def test_validate_sort_fields_invalid(self, query_validator, mock_field_manager):
        """Test validating invalid sort fields."""
        mock_field_manager.validate_sort_fields.side_effect = Exception("Invalid sort field")
        with pytest.raises(QueryError) as exc_info:
            query_validator.validate_sort_fields("collection1", ["nonexistent_field"])
        assert "Sort field validation error" in str(exc_info.value)

    def test_validate_sort_none(self, query_validator):
        """Test validating None sort parameter."""
        result = query_validator.validate_sort(None, "collection1")
        assert result is None

    def test_validate_sort_field_only(self, query_validator):
        """Test validating sort with field only."""
        result = query_validator.validate_sort("id", "collection1")
        assert result == "id asc"  # Uses default direction

    def test_validate_sort_field_and_direction(self, query_validator):
        """Test validating sort with field and direction."""
        result = query_validator.validate_sort("id desc", "collection1")
        assert result == "id desc"

    def test_validate_sort_invalid_format(self, query_validator):
        """Test validating sort with invalid format."""
        with pytest.raises(QueryError) as exc_info:
            query_validator.validate_sort("id desc asc", "collection1")
        assert "Invalid sort format" in str(exc_info.value)

    def test_validate_sort_non_sortable_field(self, query_validator):
        """Test validating sort with non-sortable field."""
        with pytest.raises(QueryError) as exc_info:
            query_validator.validate_sort("content desc", "collection1")
        assert "Field 'content' is not sortable" in str(exc_info.value)

    def test_validate_sort_invalid_direction(self, query_validator):
        """Test validating sort with invalid direction."""
        with pytest.raises(QueryError) as exc_info:
            query_validator.validate_sort("id invalid", "collection1")
        assert "Invalid sort direction 'invalid'" in str(exc_info.value)

    def test_validate_sort_error_handling(self, query_validator, mock_field_manager):
        """Test error handling in validate_sort."""
        mock_field_manager.get_field_info.side_effect = Exception("Test error")
        with pytest.raises(QueryError) as exc_info:
            query_validator.validate_sort("id desc", "collection1")
        assert "Sort field validation error" in str(exc_info.value) 