"""Query builder for Solr."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlglot import exp, parse_one
from sqlglot.expressions import (
    EQ,
    Binary,
    Column,
    From,
    Identifier,
    Literal,
    Ordered,
    Select,
    Star,
    Where,
)

from solr_mcp.solr.exceptions import QueryError
from solr_mcp.solr.query.parser import QueryParser
from solr_mcp.solr.schema.fields import FieldManager

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Builds Solr queries from SQL."""

    def __init__(self, field_manager: FieldManager):
        """Initialize QueryBuilder.

        Args:
            field_manager: Field manager for validating fields
        """
        self.field_manager = field_manager
        self.parser = QueryParser()

    def parse_and_validate(
        self, query: str
    ) -> tuple[Select, str, list[str], list[tuple[str, str]]]:
        """Parse and validate a SELECT query.

        Args:
            query: SQL query to parse

        Returns:
            Tuple of (AST, collection name, selected fields, sort fields)

        Raises:
            QueryError: If query is invalid
        """
        # Parse query
        ast, collection, fields = self.parser.parse_select(query)

        # Validate collection exists
        if not collection:
            raise QueryError("FROM clause must specify a collection")

        if not self.field_manager.validate_collection_exists(collection):
            raise QueryError(f"Collection '{collection}' does not exist")

        # Validate fields exist in collection
        if "*" not in fields:
            for field in fields:
                if not self.field_manager.validate_field_exists(field, collection):
                    raise QueryError(
                        f"Field '{field}' does not exist in collection '{collection}'"
                    )

        # Extract and validate sort fields
        sort_fields = self.parser.get_sort_fields(ast)
        if sort_fields:
            for field, direction in sort_fields:
                if not self.field_manager.validate_field_exists(field, collection):
                    raise QueryError(
                        f"Sort field '{field}' does not exist in collection '{collection}'"
                    )
                if not self.field_manager.validate_sort_field(field, collection):
                    raise QueryError(
                        f"Field '{field}' is not sortable in collection '{collection}'"
                    )

        return ast, collection, fields, sort_fields

    def parse_and_validate_select(self, query: str) -> Tuple[Select, str, List[str]]:
        """Parse and validate a SELECT query.

        Args:
            query: SQL query to parse and validate

        Returns:
            Tuple of (AST, collection name, selected fields)

        Raises:
            QueryError: If query is invalid
        """
        ast, collection, fields, _ = self.parse_and_validate(query)
        return ast, collection, fields

    def validate_sort(self, sort_spec: str | None, collection: str) -> str | None:
        """Validate sort specification.

        Args:
            sort_spec: Sort specification (field direction)
            collection: Collection name

        Returns:
            Validated sort specification

        Raises:
            QueryError: If sort specification is invalid
        """
        if not sort_spec:
            return None

        try:
            parts = sort_spec.strip().split()
            if len(parts) > 2:
                raise QueryError("Invalid sort format. Must be 'field [ASC|DESC]'")

            field = parts[0]
            direction = parts[1].upper() if len(parts) > 1 else "ASC"

            if direction not in ["ASC", "DESC"]:
                raise QueryError(
                    f"Invalid sort direction '{direction}'. Must be ASC or DESC"
                )

            if not self.field_manager.validate_field_exists(field, collection):
                raise QueryError(
                    f"Sort field '{field}' does not exist in collection '{collection}'"
                )

            if not self.field_manager.validate_sort_field(field, collection):
                raise QueryError(
                    f"Field '{field}' is not sortable in collection '{collection}'"
                )

            return f"{field} {direction}"

        except QueryError as e:
            raise e
        except Exception as e:
            raise QueryError(f"Invalid sort specification: {str(e)}")

    def extract_sort_fields(self, sort_spec: str) -> List[str]:
        """Extract sort fields from specification.

        Args:
            sort_spec: Sort specification (field direction, field direction, ...)

        Returns:
            List of field names
        """
        fields = []
        for spec in sort_spec.split(","):
            field = spec.strip().split()[0]
            fields.append(field)
        return fields

    def _convert_where_to_solr(self, where_expr: exp.Expression) -> str:
        """Convert WHERE expression to Solr filter query.

        Args:
            where_expr: WHERE expression

        Returns:
            Solr filter query

        Raises:
            QueryError: If expression type is unsupported
        """
        if isinstance(where_expr, Where):
            return self._convert_where_to_solr(where_expr.this)
        elif isinstance(where_expr, EQ):
            left = self._convert_where_to_solr(where_expr.this)
            right = self._convert_where_to_solr(where_expr.expression)
            return f"{left}:{right}"
        elif isinstance(where_expr, Binary):
            left = self._convert_where_to_solr(where_expr.this)
            right = self._convert_where_to_solr(where_expr.expression)
            op = where_expr.args.get("op", "=").upper()
            if op == "AND":
                return f"({left} AND {right})"
            elif op == "OR":
                return f"({left} OR {right})"
            elif op == "=":
                return f"{left}:{right}"
            else:
                raise QueryError(f"Unsupported operator '{op}' in WHERE clause")
        elif isinstance(where_expr, Identifier):
            return where_expr.this if hasattr(where_expr, "this") else where_expr.name
        elif isinstance(where_expr, Column):
            return (
                where_expr.args["this"].name
                if "this" in where_expr.args
                else where_expr.name
            )
        elif isinstance(where_expr, Literal):
            if where_expr.is_string:
                return f'"{where_expr.this}"'
            return str(where_expr.this)
        else:
            raise QueryError(
                f"Unsupported expression type '{type(where_expr).__name__}' in WHERE clause"
            )

    def build_solr_query(self, ast: Select) -> Dict[str, Any]:
        """Build Solr query from AST.

        Args:
            ast: Query AST

        Returns:
            Solr query parameters
        """
        params = {}

        # Add fields
        if ast.expressions and not isinstance(ast.expressions[0], Star):
            params["fl"] = ",".join(
                expr.args["this"].name if isinstance(expr, Column) else str(expr)
                for expr in ast.expressions
            )

        # Add filters
        if ast.args.get("where"):
            params["fq"] = self._convert_where_to_solr(ast.args["where"])

        # Add sort
        sort_fields = self.parser.get_sort_fields(ast)
        if sort_fields:
            params["sort"] = ",".join(
                f"{field} {direction}" for field, direction in sort_fields
            )

        # Add limit
        if ast.args.get("limit"):
            try:
                limit = ast.args["limit"]
                if isinstance(limit, exp.Limit):
                    params["rows"] = str(limit.expression)
                else:
                    params["rows"] = str(limit)
            except Exception:
                params["rows"] = "10"  # Default limit

        # Add default query if none specified
        if "fq" not in params:
            params["q"] = "*:*"

        return params

    def build_vector_query(self, base_query: str, doc_ids: List[str]) -> Dict[str, Any]:
        """Build vector query from base query and document IDs.

        Args:
            base_query: Base SQL query
            doc_ids: List of document IDs to filter by

        Returns:
            Solr query parameters

        Raises:
            QueryError: If query is invalid
        """
        try:
            # Parse and validate base query
            ast, collection, fields, sort_fields = self.parse_and_validate(base_query)

            # Add document ID filter
            if doc_ids:
                id_filter = f"_docid_:({' OR '.join(doc_ids)})"
                if ast.args.get("where"):
                    ast.args["where"] = exp.Binary(
                        this=ast.args["where"],
                        expression=exp.Identifier(this=id_filter),
                        op="AND",
                    )
                else:
                    ast.args["where"] = exp.Identifier(this=id_filter)

            # Build Solr query
            return self.build_solr_query(ast)

        except QueryError as e:
            raise e
        except Exception as e:
            raise QueryError(f"Error building vector query: {str(e)}")
