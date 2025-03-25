"""Utility functions for Solr MCP."""

import json
from typing import Any, Dict, List, Optional, Union

# Map Solr field types to our simplified type system
FIELD_TYPE_MAPPING = {
    "pint": "numeric",
    "plong": "numeric",
    "pfloat": "numeric",
    "pdouble": "numeric",
    "pdate": "date",
    "string": "string",
    "text_general": "text",
    "boolean": "boolean",
}

# Define synthetic sort fields available in Solr
SYNTHETIC_SORT_FIELDS = {
    "score": {
        "type": "numeric",
        "directions": ["asc", "desc"],
        "default_direction": "desc",
        "searchable": True,
    },
    "_docid_": {
        "type": "numeric",
        "directions": ["asc", "desc"],
        "default_direction": "asc",
        "searchable": False,
        "warning": "Internal Lucene document ID. Not stable across restarts or reindexing.",
    },
}


class SolrUtils:
    """Utility functions for Solr operations."""

    @staticmethod
    def ensure_json_object(value: Union[str, Dict, List, Any]) -> Any:
        """Ensure value is a JSON object if it's a JSON string.

        Args:
            value: Value that might be a JSON string

        Returns:
            Parsed JSON object if input was JSON string, original value otherwise
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def sanitize_filters(
        filters: Optional[Union[str, List[str], Dict[str, Any]]]
    ) -> Optional[List[str]]:
        """Sanitize and normalize filter queries.

        Args:
            filters: Raw filter input (string, list, dict, or None)

        Returns:
            List of sanitized filter strings or None
        """
        if filters is None:
            return None

        # Handle potential JSON string
        filters = SolrUtils.ensure_json_object(filters)

        # Convert to list if string or dict
        if isinstance(filters, str):
            filters = [filters]
        elif isinstance(filters, dict):
            # Convert dict to list of "key:value" strings
            filters = [f"{k}:{v}" for k, v in filters.items()]
        elif not isinstance(filters, list):
            # Try to convert to string if not list
            filters = [str(filters)]

        # Sanitize each filter
        sanitized = []
        for f in filters:
            if f:  # Skip empty filters
                # Handle nested JSON objects/strings
                f = SolrUtils.ensure_json_object(f)
                if isinstance(f, (dict, list)):
                    f = json.dumps(f)

                # Remove any dangerous characters or patterns
                f = str(f).strip()
                f = f.replace(";", "")  # Remove potential command injection
                sanitized.append(f)

        return sanitized if sanitized else None

    @staticmethod
    def sanitize_sort(
        sort: Optional[str], sortable_fields: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """Sanitize and normalize sort parameter.

        Args:
            sort: Raw sort string
            sortable_fields: Dict of available sortable fields and their properties

        Returns:
            Normalized sort string or None

        Raises:
            ValueError: If sort field or direction is invalid
        """
        if not sort:
            return None

        # Remove extra whitespace and normalize
        sort = " ".join(sort.strip().split())

        # Split into parts
        parts = sort.split(" ")
        if not parts:
            return None

        field = parts[0]
        direction = parts[1].lower() if len(parts) > 1 else None

        # Validate field
        if field not in sortable_fields:
            raise ValueError(
                f"Field '{field}' is not sortable. Available sort fields: {list(sortable_fields.keys())}"
            )

        field_info = sortable_fields[field]

        # Validate and normalize direction
        if direction:
            if direction not in field_info["directions"]:
                raise ValueError(
                    f"Invalid sort direction '{direction}' for field '{field}'. Allowed directions: {field_info['directions']}"
                )
        else:
            direction = field_info["default_direction"]

        return f"{field} {direction}"

    @staticmethod
    def sanitize_fields(
        fields: Optional[Union[str, List[str], Dict[str, Any]]]
    ) -> Optional[List[str]]:
        """Sanitize and normalize field list.

        Args:
            fields: Raw field list (string, list, dict, or None)

        Returns:
            List of sanitized field names or None
        """
        if fields is None:
            return None

        # Handle potential JSON string
        fields = SolrUtils.ensure_json_object(fields)

        # Convert to list if string or dict
        if isinstance(fields, str):
            fields = fields.split(",")
        elif isinstance(fields, dict):
            fields = list(fields.keys())
        elif not isinstance(fields, list):
            try:
                fields = [str(fields)]
            except:
                return None

        sanitized = []
        for field in fields:
            if field:  # Skip empty fields
                # Handle nested JSON
                field = SolrUtils.ensure_json_object(field)
                if isinstance(field, (dict, list)):
                    continue  # Skip complex objects

                field = str(field).strip()
                field = field.replace(";", "")  # Remove potential command injection
                sanitized.append(field)

        return sanitized if sanitized else None

    @staticmethod
    def sanitize_facets(facets: Union[str, Dict, Any]) -> Dict:
        """Sanitize facet results.

        Args:
            facets: Raw facet data (string, dict, or other)

        Returns:
            Sanitized facet dictionary
        """
        # Handle potential JSON string
        facets = SolrUtils.ensure_json_object(facets)

        if not isinstance(facets, dict):
            return {}

        sanitized = {}
        for key, value in facets.items():
            # Handle nested JSON strings
            value = SolrUtils.ensure_json_object(value)

            if isinstance(value, dict):
                sanitized[key] = SolrUtils.sanitize_facets(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    SolrUtils.ensure_json_object(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    @staticmethod
    def sanitize_highlighting(highlighting: Union[str, Dict, Any]) -> Dict:
        """Sanitize highlighting results.

        Args:
            highlighting: Raw highlighting data (string, dict, or other)

        Returns:
            Sanitized highlighting dictionary
        """
        # Handle potential JSON string
        highlighting = SolrUtils.ensure_json_object(highlighting)

        if not isinstance(highlighting, dict):
            return {}

        sanitized = {}
        for doc_id, fields in highlighting.items():
            # Handle potential JSON string in fields
            fields = SolrUtils.ensure_json_object(fields)
            if not isinstance(fields, dict):
                continue

            sanitized[str(doc_id)] = {
                str(field): [
                    str(snippet) for snippet in SolrUtils.ensure_json_object(snippets)
                ]
                for field, snippets in fields.items()
                if isinstance(SolrUtils.ensure_json_object(snippets), (list, tuple))
            }

        return sanitized
