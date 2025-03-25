"""Unit tests for utility functions."""

import pytest

from solr_mcp.utils import SolrUtils


def test_ensure_json_object():
    """Test JSON object conversion."""
    # Test valid JSON string
    assert SolrUtils.ensure_json_object('{"key": "value"}') == {"key": "value"}
    assert SolrUtils.ensure_json_object('["a", "b"]') == ["a", "b"]

    # Test non-JSON string
    assert SolrUtils.ensure_json_object("plain text") == "plain text"

    # Test dict/list input
    test_dict = {"test": 123}
    test_list = [1, 2, 3]
    assert SolrUtils.ensure_json_object(test_dict) == test_dict
    assert SolrUtils.ensure_json_object(test_list) == test_list

    # Test other types
    assert SolrUtils.ensure_json_object(123) == 123
    assert SolrUtils.ensure_json_object(None) is None


def test_sanitize_filters():
    """Test filter sanitization."""
    # Test None input
    assert SolrUtils.sanitize_filters(None) is None

    # Test string input
    assert SolrUtils.sanitize_filters("field:value") == ["field:value"]
    assert SolrUtils.sanitize_filters('{"field": "value"}') == ["field:value"]

    # Test list input
    assert SolrUtils.sanitize_filters(["field1:value1", "field2:value2"]) == [
        "field1:value1",
        "field2:value2",
    ]

    # Test dict input
    assert SolrUtils.sanitize_filters({"field": "value"}) == ["field:value"]

    # Test mixed input with JSON strings
    mixed_input = ['{"field1": "value1"}', "field2:value2"]
    result = SolrUtils.sanitize_filters(mixed_input)
    assert len(result) == 2
    assert "field2:value2" in result
    assert any("field1" in item and "value1" in item for item in result)

    # Test empty inputs
    assert SolrUtils.sanitize_filters("") is None
    assert SolrUtils.sanitize_filters([]) is None
    assert SolrUtils.sanitize_filters({}) is None

    # Test sanitization
    assert SolrUtils.sanitize_filters("field;value") == [
        "fieldvalue"
    ]  # Removes semicolons

    # Test non-string/list/dict input
    assert SolrUtils.sanitize_filters(123) == ["123"]


def test_sanitize_sort():
    """Test sort parameter sanitization."""
    sortable_fields = {
        "score": {
            "type": "numeric",
            "directions": ["asc", "desc"],
            "default_direction": "desc",
        },
        "date": {
            "type": "date",
            "directions": ["asc", "desc"],
            "default_direction": "desc",
        },
    }

    # Test None input
    assert SolrUtils.sanitize_sort(None, sortable_fields) is None

    # Test valid inputs
    assert SolrUtils.sanitize_sort("score desc", sortable_fields) == "score desc"
    assert SolrUtils.sanitize_sort("date asc", sortable_fields) == "date asc"

    # Test default direction
    assert SolrUtils.sanitize_sort("score", sortable_fields) == "score desc"

    # Test whitespace normalization
    assert SolrUtils.sanitize_sort("  score    desc  ", sortable_fields) == "score desc"

    # Test invalid field
    with pytest.raises(ValueError, match="Field 'invalid' is not sortable"):
        SolrUtils.sanitize_sort("invalid desc", sortable_fields)

    # Test invalid direction
    with pytest.raises(
        ValueError, match="Invalid sort direction 'invalid' for field 'score'"
    ):
        SolrUtils.sanitize_sort("score invalid", sortable_fields)

    # Test empty input
    assert SolrUtils.sanitize_sort("", sortable_fields) is None


def test_sanitize_fields():
    """Test field list sanitization."""
    # Test None input
    assert SolrUtils.sanitize_fields(None) is None

    # Test string input
    assert SolrUtils.sanitize_fields("field1,field2") == ["field1", "field2"]

    # Test list input
    assert SolrUtils.sanitize_fields(["field1", "field2"]) == ["field1", "field2"]

    # Test dict input
    assert SolrUtils.sanitize_fields({"field1": 1, "field2": 2}) == ["field1", "field2"]

    # Test JSON string input
    assert SolrUtils.sanitize_fields('["field1", "field2"]') == ["field1", "field2"]

    # Test empty inputs
    assert SolrUtils.sanitize_fields("") is None
    assert SolrUtils.sanitize_fields([]) is None
    assert SolrUtils.sanitize_fields({}) is None

    # Test sanitization
    assert SolrUtils.sanitize_fields("field;name") == [
        "fieldname"
    ]  # Removes semicolons

    # Test complex objects
    assert SolrUtils.sanitize_fields([{"complex": "object"}]) is None

    # Test non-string/list/dict input
    assert SolrUtils.sanitize_fields(123) == ["123"]


def test_sanitize_facets():
    """Test facet sanitization."""
    # Test None/invalid input
    assert SolrUtils.sanitize_facets(None) == {}
    assert SolrUtils.sanitize_facets("not a dict") == {}

    # Test simple dict
    input_dict = {"field1": "value1", "field2": 123}
    assert SolrUtils.sanitize_facets(input_dict) == input_dict

    # Test nested dict
    nested_dict = {"field1": {"subfield1": "value1"}, "field2": ["value2", "value3"]}
    assert SolrUtils.sanitize_facets(nested_dict) == nested_dict

    # Test JSON string input
    json_input = '{"field1": "value1", "field2": ["value2", "value3"]}'
    expected = {"field1": "value1", "field2": ["value2", "value3"]}
    assert SolrUtils.sanitize_facets(json_input) == expected

    # Test nested JSON strings
    nested_json = {
        "field1": '{"subfield1": "value1"}',
        "field2": '["value2", "value3"]',
    }
    expected = {"field1": {"subfield1": "value1"}, "field2": ["value2", "value3"]}
    assert SolrUtils.sanitize_facets(nested_json) == expected


def test_sanitize_highlighting():
    """Test highlighting sanitization."""
    # Test None/invalid input
    assert SolrUtils.sanitize_highlighting(None) == {}
    assert SolrUtils.sanitize_highlighting("not a dict") == {}

    # Test simple highlighting dict
    input_dict = {
        "doc1": {"field1": ["snippet1", "snippet2"]},
        "doc2": {"field2": ["snippet3"]},
    }
    assert SolrUtils.sanitize_highlighting(input_dict) == input_dict

    # Test JSON string input
    json_input = '{"doc1": {"field1": ["snippet1", "snippet2"]}}'
    expected = {"doc1": {"field1": ["snippet1", "snippet2"]}}
    assert SolrUtils.sanitize_highlighting(json_input) == expected

    # Test nested JSON strings
    nested_json = {
        "doc1": '{"field1": ["snippet1", "snippet2"]}',
        "doc2": '{"field2": ["snippet3"]}',
    }
    expected = {
        "doc1": {"field1": ["snippet1", "snippet2"]},
        "doc2": {"field2": ["snippet3"]},
    }
    assert SolrUtils.sanitize_highlighting(nested_json) == expected

    # Test invalid field values
    invalid_input = {"doc1": {"field1": "not a list"}, "doc2": {"field2": 123}}
    assert SolrUtils.sanitize_highlighting(invalid_input) == {"doc1": {}, "doc2": {}}
