"""Exceptions for SolrCloud client."""

class SolrError(Exception):
    """Base exception for Solr-related errors."""
    pass

class ConfigurationError(SolrError):
    """Configuration-related errors."""
    pass

class ConnectionError(SolrError):
    """Connection-related errors."""
    pass

class QueryError(SolrError):
    """Query-related errors."""
    pass

class SchemaError(SolrError):
    """Schema-related errors."""
    pass 