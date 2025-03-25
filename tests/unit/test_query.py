"""Unit tests for query module."""

import pytest
from sqlglot import parse_one

from solr_mcp.solr.exceptions import QueryError
from solr_mcp.solr.query.builder import QueryBuilder
from solr_mcp.solr.query.parser import QueryParser
from solr_mcp.solr.schema.fields import FieldManager


@pytest.fixture
def query_parser():
    """Create a QueryParser instance."""
    return QueryParser()


@pytest.fixture
def field_manager(mocker):
    """Create a mocked FieldManager."""
    manager = mocker.Mock(spec=FieldManager)
    manager.validate_collection_exists.return_value = True
    manager.validate_field_exists.return_value = True
    manager.validate_sort_field.return_value = True
    return manager


@pytest.fixture
def query_builder(field_manager):
    """Create a QueryBuilder instance."""
    return QueryBuilder(field_manager)


class TestQueryParser:
    """Test QueryParser class."""

    def test_parse_select_with_star(self, query_parser):
        """Test parsing SELECT * query."""
        query = "SELECT * FROM test_collection"
        ast, collection, fields = query_parser.parse_select(query)
        assert collection == "test_collection"
        assert fields == ["*"]

    def test_parse_select_with_fields(self, query_parser):
        """Test parsing SELECT with specific fields."""
        query = "SELECT id, title FROM test_collection"
        ast, collection, fields = query_parser.parse_select(query)
        assert collection == "test_collection"
        assert fields == ["id", "title"]

    def test_parse_select_with_where(self, query_parser):
        """Test parsing SELECT with WHERE clause."""
        query = "SELECT * FROM test_collection WHERE title = 'test'"
        ast, collection, fields = query_parser.parse_select(query)
        assert collection == "test_collection"
        assert fields == ["*"]

    def test_parse_select_with_order(self, query_parser):
        """Test parsing SELECT with ORDER BY."""
        query = "SELECT * FROM test_collection ORDER BY score DESC"
        ast, collection, fields = query_parser.parse_select(query)
        assert collection == "test_collection"
        assert fields == ["*"]
        sort_fields = query_parser.get_sort_fields(ast)
        assert sort_fields == [("score", "DESC")]

    def test_parse_select_invalid_syntax(self, query_parser):
        """Test parsing invalid SQL syntax."""
        query = "SELECT FROM test_collection"
        with pytest.raises(QueryError) as exc_info:
            query_parser.parse_select(query)

    def test_parse_select_no_from(self, query_parser):
        """Test parsing SELECT without FROM."""
        query = "SELECT *"
        with pytest.raises(QueryError) as exc_info:
            query_parser.parse_select(query)

    def test_parse_select_empty_from(self, query_parser):
        """Test parsing SELECT with empty FROM."""
        query = "SELECT * FROM"
        with pytest.raises(QueryError) as exc_info:
            query_parser.parse_select(query)

    def test_parse_select_invalid_collection(self, query_parser):
        """Test parsing SELECT with invalid collection."""
        query = "SELECT * FROM ''"
        with pytest.raises(QueryError) as exc_info:
            query_parser.parse_select(query)

    def test_parse_select_with_field_value_syntax(self, query_parser):
        """Test parsing field:value syntax."""
        query = "SELECT * FROM test_collection WHERE field:value"
        ast, collection, fields = query_parser.parse_select(query)
        assert collection == "test_collection"
        assert fields == ["*"]


class TestQueryBuilder:
    """Test QueryBuilder class."""

    def test_init(self, query_builder):
        """Test initialization."""
        assert query_builder.field_manager is not None
        assert query_builder.parser is not None

    def test_parse_and_validate_select_success(self, query_builder):
        """Test successful query parsing and validation."""
        query = "SELECT title, content FROM test_collection WHERE title = 'test'"
        ast, collection, fields, sort_fields = query_builder.parse_and_validate(query)
        assert collection == "test_collection"
        assert fields == ["title", "content"]
        assert sort_fields == []

    def test_parse_and_validate_invalid_collection(self, query_builder, field_manager):
        """Test parsing with invalid collection."""
        field_manager.validate_collection_exists.return_value = False
        query = "SELECT * FROM invalid_collection"
        with pytest.raises(QueryError) as exc_info:
            query_builder.parse_and_validate(query)

    def test_parse_and_validate_invalid_fields(self, query_builder, field_manager):
        """Test parsing with invalid fields."""
        field_manager.validate_field_exists.return_value = False
        query = "SELECT invalid_field FROM test_collection"
        with pytest.raises(QueryError) as exc_info:
            query_builder.parse_and_validate(query)

    def test_parse_and_validate_invalid_sort(self, query_builder, field_manager):
        """Test parsing with invalid sort field."""
        field_manager.validate_sort_field.return_value = False
        query = "SELECT * FROM test_collection ORDER BY invalid_field"
        with pytest.raises(QueryError) as exc_info:
            query_builder.parse_and_validate(query)

    def test_build_solr_query_success(self, query_builder):
        """Test building Solr query."""
        ast = parse_one(
            "SELECT * FROM test_collection WHERE title = 'test' ORDER BY score DESC"
        )
        solr_query = query_builder.build_solr_query(ast)
        assert "fq" in solr_query
        assert solr_query["fq"] == 'title:"test"'
        assert "sort" in solr_query
        assert solr_query["sort"] == "score DESC"

    def test_build_solr_query_with_fields(self, query_builder):
        """Test building Solr query with specific fields."""
        ast = parse_one("SELECT id, title FROM test_collection")
        solr_query = query_builder.build_solr_query(ast)
        assert solr_query["fl"] == "id,title"

    def test_parse_and_validate_select_invalid_query(self, query_builder):
        """Test parsing invalid query."""
        query = "INVALID SQL"
        with pytest.raises(QueryError) as exc_info:
            query_builder.parse_and_validate_select(query)

    def test_parse_and_validate_select_invalid_collection(
        self, query_builder, field_manager
    ):
        """Test parsing with invalid collection."""
        field_manager.validate_collection_exists.return_value = False
        query = "SELECT * FROM invalid_collection"
        with pytest.raises(QueryError) as exc_info:
            query_builder.parse_and_validate_select(query)

    def test_parse_and_validate_select_invalid_fields(
        self, query_builder, field_manager
    ):
        """Test parsing with invalid fields."""
        field_manager.validate_field_exists.return_value = False
        query = "SELECT invalid_field FROM test_collection"
        with pytest.raises(QueryError) as exc_info:
            query_builder.parse_and_validate_select(query)

    def test_parse_and_validate_select_with_sort(self, query_builder):
        """Test parsing query with ORDER BY clause."""
        query = "SELECT id, title FROM collection1 ORDER BY id ASC, title DESC"
        ast, collection, fields, sort_fields = query_builder.parse_and_validate(query)
        assert collection == "collection1"
        assert fields == ["id", "title"]
        assert sort_fields == [("id", "ASC"), ("title", "DESC")]

    def test_parse_and_validate_select_invalid_sort_field(
        self, query_builder, field_manager
    ):
        """Test parsing with invalid sort field."""
        field_manager.validate_sort_field.return_value = False
        query = "SELECT * FROM test_collection ORDER BY invalid_field"
        with pytest.raises(QueryError) as exc_info:
            query_builder.parse_and_validate_select(query)

    def test_validate_sort_valid(self, query_builder):
        """Test validating valid sort specification."""
        result = query_builder.validate_sort("id DESC", "collection1")
        assert result == "id DESC"

    def test_validate_sort_default_direction(self, query_builder):
        """Test validating sort with default direction."""
        result = query_builder.validate_sort("id", "collection1")
        assert result == "id ASC"

    def test_validate_sort_invalid_format(self, query_builder):
        """Test validating invalid sort format."""
        with pytest.raises(QueryError) as exc_info:
            query_builder.validate_sort("id desc asc", "collection1")

    def test_validate_sort_invalid_field(self, query_builder, field_manager):
        """Test validating invalid sort field."""
        field_manager.validate_field_exists.return_value = False
        with pytest.raises(QueryError) as exc_info:
            query_builder.validate_sort("invalid_field ASC", "collection1")

    def test_validate_sort_invalid_direction(self, query_builder):
        """Test validating invalid sort direction."""
        with pytest.raises(QueryError) as exc_info:
            query_builder.validate_sort("id INVALID", "collection1")

    def test_validate_sort_none(self, query_builder):
        """Test validating None sort specification."""
        result = query_builder.validate_sort(None, "collection1")
        assert result is None

    def test_extract_sort_fields_single(self, query_builder):
        """Test extracting single sort field."""
        fields = query_builder.extract_sort_fields("id DESC")
        assert fields == ["id"]

    def test_extract_sort_fields_multiple(self, query_builder):
        """Test extracting multiple sort fields."""
        fields = query_builder.extract_sort_fields("id DESC, title ASC")
        assert fields == ["id", "title"]

    def test_build_vector_query(self, query_builder):
        """Test building query with vector search results."""
        query = "SELECT * FROM collection1"
        doc_ids = ["1", "2", "3"]
        result = query_builder.build_vector_query(query, doc_ids)
        assert "fq" in result
        assert result["fq"] == "_docid_:(1 OR 2 OR 3)"

    def test_build_vector_query_with_existing_where(self, query_builder):
        """Test building vector query with existing WHERE clause."""
        base_query = "SELECT id, title FROM collection1 WHERE status = 'active'"
        doc_ids = ["1", "2"]
        result = query_builder.build_vector_query(base_query, doc_ids)
        assert "fq" in result
        assert 'status:"active"' in result["fq"]
        assert "_docid_:(1 OR 2)" in result["fq"]

    def test_build_vector_query_empty_ids(self, query_builder):
        """Test building vector query with empty document IDs."""
        base_query = "SELECT id, title FROM collection1"
        doc_ids = []
        result = query_builder.build_vector_query(base_query, doc_ids)
        assert "fl" in result
        assert result["fl"] == "id,title"

    def test_build_vector_query_with_order_by(self, query_builder):
        """Test building vector query preserving ORDER BY clause."""
        base_query = "SELECT id, title FROM collection1 ORDER BY title DESC"
        doc_ids = ["1", "2"]
        result = query_builder.build_vector_query(base_query, doc_ids)
        assert "sort" in result
        assert result["sort"] == "title DESC"

    def test_build_vector_query_with_limit(self, query_builder):
        """Test building vector query preserving LIMIT clause."""
        base_query = "SELECT id, title FROM collection1 LIMIT 10"
        doc_ids = ["1", "2"]
        result = query_builder.build_vector_query(base_query, doc_ids)
        assert "rows" in result
        assert result["rows"] == "10"

    def test_build_vector_query_no_from(self, query_builder):
        """Test building vector query with invalid query."""
        query = "SELECT *"
        doc_ids = ["1", "2"]
        with pytest.raises(QueryError) as exc_info:
            query_builder.build_vector_query(query, doc_ids)

    def test_build_vector_query_error(self, query_builder):
        """Test building vector query with error."""
        query = "INVALID SQL"
        doc_ids = ["1", "2"]
        with pytest.raises(QueryError) as exc_info:
            query_builder.build_vector_query(query, doc_ids)
