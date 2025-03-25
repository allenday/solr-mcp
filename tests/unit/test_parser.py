"""Unit tests for QueryParser."""

import pytest

from solr_mcp.solr.exceptions import QueryError
from solr_mcp.solr.query.parser import QueryParser


@pytest.fixture
def query_parser():
    """Create QueryParser instance for testing."""
    return QueryParser()


class TestQueryParser:
    """Test cases for QueryParser."""

    def test_init(self, query_parser):
        """Test QueryParser initialization."""
        assert isinstance(query_parser, QueryParser)

    def test_preprocess_query_basic(self, query_parser):
        """Test preprocessing basic field:value syntax."""
        query = "SELECT * FROM collection1 WHERE field:value"
        result = query_parser.preprocess_query(query)
        assert "field = 'value'" in result

    def test_preprocess_query_multiple(self, query_parser):
        """Test preprocessing multiple field:value pairs."""
        query = "SELECT * FROM collection1 WHERE field1:value1 AND field2:value2"
        result = query_parser.preprocess_query(query)
        assert "field1 = 'value1'" in result
        assert "field2 = 'value2'" in result

    def test_parse_select_valid(self, query_parser):
        """Test parsing valid SELECT query."""
        query = "SELECT id, title FROM collection1"
        ast, collection, fields = query_parser.parse_select(query)

        assert ast is not None
        assert collection == "collection1"
        assert fields == ["id", "title"]

    def test_parse_select_no_select(self, query_parser):
        """Test parsing non-SELECT query."""
        query = "INSERT INTO collection1 (id) VALUES (1)"

        with pytest.raises(QueryError) as exc_info:
            query_parser.parse_select(query)
        assert exc_info.type == QueryError

    def test_parse_select_no_from(self, query_parser):
        """Test parsing query without FROM clause."""
        query = "SELECT id, title"

        with pytest.raises(QueryError) as exc_info:
            query_parser.parse_select(query)
        assert exc_info.type == QueryError

    def test_parse_select_with_alias(self, query_parser):
        """Test parsing query with aliased fields."""
        query = "SELECT id as doc_id, title as doc_title FROM collection1"
        ast, collection, fields = query_parser.parse_select(query)

        assert ast is not None
        assert collection == "collection1"
        assert "doc_id" in fields
        assert "doc_title" in fields

    def test_parse_select_with_star(self, query_parser):
        """Test parsing query with * selector."""
        query = "SELECT * FROM collection1"
        ast, collection, fields = query_parser.parse_select(query)

        assert ast is not None
        assert collection == "collection1"
        assert "*" in fields

    def test_parse_select_invalid_syntax(self, query_parser):
        """Test parsing query with invalid syntax."""
        query = "INVALID SQL"

        with pytest.raises(QueryError) as exc_info:
            query_parser.parse_select(query)
        assert exc_info.type == QueryError

    def test_extract_sort_fields_single(self, query_parser):
        """Test extracting fields from single sort specification."""
        sort_spec = "title desc"
        fields = query_parser.extract_sort_fields(sort_spec)
        assert fields == ["title"]

    def test_extract_sort_fields_multiple(self, query_parser):
        """Test extracting fields from multiple sort specifications."""
        sort_spec = "title desc, id asc"
        fields = query_parser.extract_sort_fields(sort_spec)
        assert fields == ["title", "id"]
