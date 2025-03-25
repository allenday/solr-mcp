"""Query validation for SolrCloud client."""

import logging
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlglot.expressions import Select

from solr_mcp.solr.exceptions import QueryError

logger = logging.getLogger(__name__)


class QueryValidator:
    """Validates SQL queries for Solr."""

    def __init__(self, field_manager):
        """Initialize the QueryValidator.

        Args:
            field_manager: FieldManager instance for field validation
        """
        self.field_manager = field_manager

    def validate_fields(self, collection: str, fields: List[str]) -> None:
        """Validate that fields exist in the collection.

        Args:
            collection: Collection name
            fields: List of field names to validate

        Raises:
            QueryError: If fields are invalid
        """
        try:
            # Get available fields for collection
            available_fields = self.field_manager.get_field_types(collection)

            # Check each field exists
            for field in fields:
                if field not in available_fields:
                    raise QueryError(
                        f"Invalid field '{field}' - field does not exist in collection '{collection}'"
                    )

        except QueryError:
            raise
        except Exception as e:
            raise QueryError(f"Field validation error: {str(e)}")

    def validate_sort_fields(self, collection: str, fields: List[str]) -> None:
        """Validate that fields are sortable in the collection.

        Args:
            collection: Collection name
            fields: List of field names to validate

        Raises:
            QueryError: If fields are not sortable
        """
        try:
            self.field_manager.validate_sort_fields(collection, fields)
        except Exception as e:
            raise QueryError(f"Sort field validation error: {str(e)}")

    def validate_sort(self, sort: Optional[str], collection: str) -> Optional[str]:
        """Validate and normalize sort parameter.

        Args:
            sort: Sort string in format "field direction" or just "field"
            collection: Collection name

        Returns:
            Validated sort string or None if sort is None

        Raises:
            QueryError: If sort specification is invalid
        """
        if not sort:
            return None

        parts = sort.strip().split()
        if len(parts) == 1:
            field = parts[0]
            direction = None
        elif len(parts) == 2:
            field, direction = parts
        else:
            raise QueryError(f"Invalid sort format: {sort}")

        try:
            # Get sortable fields for the collection
            field_info = self.field_manager.get_field_info(collection)
            sortable_fields = field_info["sortable_fields"]

            # Check if field is sortable
            if field not in sortable_fields:
                raise QueryError(f"Field '{field}' is not sortable")

            # Validate direction if provided
            if direction:
                valid_directions = sortable_fields[field]["directions"]
                if direction.lower() not in [d.lower() for d in valid_directions]:
                    raise QueryError(
                        f"Invalid sort direction '{direction}' for field '{field}'"
                    )
            else:
                # Use default direction for field
                direction = sortable_fields[field]["default_direction"]

            return f"{field} {direction}"
        except QueryError:
            raise
        except Exception as e:
            raise QueryError(f"Sort field validation error: {str(e)}")
