"""SolrCloud client package."""

from solr_mcp.solr.client import SolrClient
from solr_mcp.solr.config import SolrConfig
from solr_mcp.solr.constants import FIELD_TYPE_MAPPING, SYNTHETIC_SORT_FIELDS
from solr_mcp.solr.exceptions import (
    ConfigurationError,
    ConnectionError,
    QueryError,
    SchemaError,
    SolrError,
)

__all__ = [
    "SolrConfig",
    "SolrClient",
    "SolrError",
    "ConfigurationError",
    "ConnectionError",
    "QueryError",
    "SchemaError",
    "FIELD_TYPE_MAPPING",
    "SYNTHETIC_SORT_FIELDS",
]
