"""Solr client exceptions."""

from typing import Any, Dict, Optional


class SolrError(Exception):
    """Base exception for Solr-related errors."""

    pass


class ConfigurationError(SolrError):
    """Configuration-related errors."""

    pass


class ConnectionError(SolrError):
    """Exception raised for connection-related errors."""

    pass


class QueryError(SolrError):
    """Base exception for query-related errors."""

    def __init__(
        self,
        message: str,
        error_type: Optional[str] = None,
        response_time: Optional[int] = None,
    ):
        self.message = message
        self.error_type = error_type
        self.response_time = response_time
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary format."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "response_time": self.response_time,
        }


class DocValuesError(QueryError):
    """Exception raised when a query requires DocValues but fields don't have them enabled."""

    def __init__(self, message: str, response_time: Optional[int] = None):
        super().__init__(
            message, error_type="MISSING_DOCVALUES", response_time=response_time
        )


class SQLParseError(QueryError):
    """Exception raised when SQL query parsing fails."""

    def __init__(self, message: str, response_time: Optional[int] = None):
        super().__init__(message, error_type="PARSE_ERROR", response_time=response_time)


class SQLExecutionError(QueryError):
    """Exception raised for other SQL execution errors."""

    def __init__(self, message: str, response_time: Optional[int] = None):
        super().__init__(
            message, error_type="SOLR_SQL_ERROR", response_time=response_time
        )


class SchemaError(SolrError):
    """Base exception for schema-related errors."""

    def __init__(
        self,
        message: str,
        error_type: str = "schema_error",
        collection: str = "unknown",
    ):
        """Initialize SchemaError.

        Args:
            message: Error message
            error_type: Type of schema error
            collection: Collection name
        """
        self.error_type = error_type
        self.collection = collection
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary format."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "collection": self.collection,
        }


class CollectionNotFoundError(SchemaError):
    """Exception raised when a collection does not exist."""

    def __init__(self, collection: str):
        super().__init__(
            message=f"Collection '{collection}' not found",
            error_type="COLLECTION_NOT_FOUND",
            collection=collection,
        )


class SchemaNotFoundError(SchemaError):
    """Exception raised when a collection's schema cannot be retrieved."""

    def __init__(self, collection: str, details: str = None):
        message = f"Schema for collection '{collection}' could not be retrieved"
        if details:
            message += f": {details}"
        super().__init__(
            message=message, error_type="SCHEMA_NOT_FOUND", collection=collection
        )
