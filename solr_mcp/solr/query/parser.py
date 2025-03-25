"""Query parser for Solr."""

import logging
from typing import List, Optional, Tuple

from loguru import logger
from sqlglot import ParseError, exp, parse_one
from sqlglot.expressions import (
    Alias,
    Binary,
    Column,
    From,
    Identifier,
    Ordered,
    Select,
    Star,
    Table,
    Where,
)

from solr_mcp.solr.exceptions import QueryError

logger = logging.getLogger(__name__)


class QueryParser:
    """Parses SQL queries for Solr."""

    def preprocess_query(self, query: str) -> str:
        """Preprocess query to handle field:value syntax.

        Args:
            query: SQL query to preprocess

        Returns:
            Preprocessed query
        """
        # Convert field:value to field = 'value'
        parts = query.split()
        for i, part in enumerate(parts):
            if ":" in part and not part.startswith('"') and not part.endswith('"'):
                field, value = part.split(":")
                parts[i] = f"{field} = '{value}'"
        return " ".join(parts)

    def parse_select(self, query: str) -> Tuple[Select, str, List[str]]:
        """Parse a SELECT query.

        Args:
            query: SQL query to parse

        Returns:
            Tuple of (AST, collection name, selected fields)

        Raises:
            QueryError: If query is invalid
        """
        try:
            # Validate and parse query
            preprocessed = self.preprocess_query(query)
            try:
                ast = parse_one(preprocessed)
            except ParseError as e:
                raise QueryError(f"Invalid SQL syntax: {str(e)}")

            if not isinstance(ast, Select):
                raise QueryError("Query must be a SELECT statement")

            # Validate selected fields
            if not ast.expressions:
                raise QueryError("SELECT clause must specify at least one field")

            # Get collection from FROM clause
            from_expr = ast.args.get("from")
            if not from_expr:
                raise QueryError("FROM clause is required")

            # Extract collection name
            collection = None
            if isinstance(from_expr, Table):
                collection = from_expr.name
            elif isinstance(from_expr, From):
                if isinstance(from_expr.this, Table):
                    collection = from_expr.this.name
                elif isinstance(from_expr.this, Identifier):
                    collection = from_expr.this.name
                elif hasattr(from_expr.this, "this") and isinstance(
                    from_expr.this.this, (Table, Identifier)
                ):
                    collection = from_expr.this.this.name

            if not collection:
                raise QueryError("FROM clause must specify a collection")

            # Get selected fields
            fields = []
            logger.debug(f"AST: {repr(ast)}")
            for expr in ast.expressions:
                logger.debug(f"Expression: {repr(expr)}")
                logger.debug(f"Expression type: {type(expr)}")
                logger.debug(f"Expression args: {expr.args}")
                if isinstance(expr, Star):
                    fields.append("*")
                elif isinstance(expr, Column):
                    fields.append(expr.args["this"].name)
                elif isinstance(expr, Alias):
                    fields.append(expr.args["alias"].this)
                elif isinstance(expr, Identifier):
                    fields.append(expr.name)

            return ast, collection, fields

        except QueryError as e:
            raise e
        except Exception as e:
            raise QueryError(f"Error parsing query: {str(e)}")

    def get_sort_fields(self, ast: Select) -> List[Tuple[str, str]]:
        """Get sort fields from AST.

        Args:
            ast: Query AST

        Returns:
            List of (field, direction) tuples
        """
        sort_fields = []
        if ast.args.get("order"):
            for expr in ast.args["order"]:
                if isinstance(expr, Ordered):
                    field = (
                        expr.this.name
                        if isinstance(expr.this, Identifier)
                        else expr.this.args["this"].name
                    )
                    direction = expr.args["desc"] and "DESC" or "ASC"
                    sort_fields.append((field, direction))

        return sort_fields

    def extract_sort_fields(self, sort_spec: str) -> List[str]:
        """Extract field names from a sort specification.

        Args:
            sort_spec: Sort specification string

        Returns:
            List of field names
        """
        fields = []
        parts = sort_spec.split(",")
        for part in parts:
            field = part.strip().split()[0]
            fields.append(field)
        return fields
