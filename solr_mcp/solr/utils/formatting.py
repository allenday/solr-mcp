"""Utilities for formatting Solr search results."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import pysolr

from solr_mcp.solr.exceptions import QueryError, SolrError

logger = logging.getLogger(__name__)


def format_search_results(
    results: pysolr.Results,
    start: int = 0,
    include_score: bool = True,
    include_facets: bool = True,
    include_highlighting: bool = True,
) -> str:
    """Format Solr search results for consumption.

    Args:
        results: pysolr Results object
        start: Start offset used in the search
        include_score: Whether to include score information
        include_facets: Whether to include facet information
        include_highlighting: Whether to include highlighting information

    Returns:
        Formatted results as JSON string
    """
    try:
        formatted = {
            "result-set": {
                "numFound": results.hits,
                "start": start,
                "docs": list(results.docs) if hasattr(results, "docs") else [],
            }
        }

        # Include score information if requested and available
        if include_score and hasattr(results, "max_score"):
            formatted["result-set"]["maxScore"] = results.max_score

        # Include facets if requested and available
        if include_facets and hasattr(results, "facets") and results.facets:
            formatted["result-set"]["facets"] = results.facets

        # Include highlighting if requested and available
        if (
            include_highlighting
            and hasattr(results, "highlighting")
            and results.highlighting
        ):
            formatted["result-set"]["highlighting"] = results.highlighting

        try:
            return json.dumps(formatted, default=str)
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # Fall back to basic result format
            return json.dumps(
                {
                    "result-set": {
                        "numFound": results.hits,
                        "start": start,
                        "docs": (
                            [str(doc) for doc in results.docs]
                            if hasattr(results, "docs")
                            else []
                        ),
                    }
                }
            )
    except Exception as e:
        logger.error(f"Error formatting search results: {e}")
        return json.dumps({"error": str(e)})


def format_sql_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Format SQL query response to a standardized structure."""
    try:
        # Check for error response
        if "result-set" in raw_response and "docs" in raw_response["result-set"]:
            docs = raw_response["result-set"]["docs"]
            if len(docs) == 1 and "EXCEPTION" in docs[0]:
                raise QueryError(docs[0]["EXCEPTION"])

        # Return standardized response format
        return {
            "result-set": {
                "docs": raw_response.get("result-set", {}).get("docs", []),
                "numFound": len(raw_response.get("result-set", {}).get("docs", [])),
                "start": 0,
            }
        }
    except QueryError as e:
        raise e
    except Exception as e:
        raise QueryError(f"Error formatting SQL response: {str(e)}")


def format_error_response(error: Exception) -> str:
    """Format error response as JSON string.

    Args:
        error: Exception object

    Returns:
        Error message as JSON string
    """
    error_code = "INTERNAL_ERROR"
    if isinstance(error, QueryError):
        error_code = "QUERY_ERROR"
    elif isinstance(error, SolrError):
        error_code = "SOLR_ERROR"

    return json.dumps({"error": {"code": error_code, "message": str(error)}})
