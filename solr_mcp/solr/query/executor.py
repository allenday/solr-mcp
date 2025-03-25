"""Query execution for SolrCloud."""

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from loguru import logger

from solr_mcp.solr.exceptions import (
    DocValuesError,
    QueryError,
    SolrError,
    SQLExecutionError,
    SQLParseError,
)
from solr_mcp.solr.utils.formatting import format_sql_response
from solr_mcp.solr.vector import VectorSearchResults

logger = logging.getLogger(__name__)


class QueryExecutor:
    """Executes queries against Solr."""

    def __init__(self, base_url: str):
        """Initialize with Solr base URL.

        Args:
            base_url: Base URL for Solr instance
        """
        self.base_url = base_url.rstrip("/")

    async def execute_select_query(self, query: str, collection: str) -> Dict[str, Any]:
        """Execute a SQL SELECT query against Solr using the SQL interface.

        Args:
            query: SQL query to execute
            collection: Collection to query

        Returns:
            Query results

        Raises:
            SQLExecutionError: If the query fails
        """
        try:
            # Build SQL endpoint URL with aggregationMode
            sql_url = f"{self.base_url}/{collection}/sql?aggregationMode=facet"
            logger.debug(f"SQL URL: {sql_url}")

            # Execute SQL query with URL-encoded form data
            payload = {"stmt": query.strip()}
            logger.debug(f"Request payload: {payload}")

            response = requests.post(
                sql_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response text: {response.text}")

            if response.status_code != 200:
                raise SQLExecutionError(
                    f"SQL query failed with status {response.status_code}: {response.text}"
                )

            response_json = response.json()

            # Check for Solr SQL exception in response
            if "result-set" in response_json and "docs" in response_json["result-set"]:
                docs = response_json["result-set"]["docs"]
                if docs and "EXCEPTION" in docs[0]:
                    exception_msg = docs[0]["EXCEPTION"]
                    response_time = docs[0].get("RESPONSE_TIME")

                    # Raise appropriate exception type based on error message
                    if "must have DocValues to use this feature" in exception_msg:
                        raise DocValuesError(exception_msg, response_time)
                    elif "parse failed:" in exception_msg:
                        raise SQLParseError(exception_msg, response_time)
                    else:
                        raise SQLExecutionError(exception_msg, response_time)

            return format_sql_response(response_json)

        except (DocValuesError, SQLParseError, SQLExecutionError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise SQLExecutionError(f"SQL query failed: {str(e)}")

    async def execute_vector_select_query(
        self,
        query: str,
        vector: List[float],
        field: str,
        collection: str,
        vector_results: VectorSearchResults,
    ) -> Dict[str, Any]:
        """Execute SQL query filtered by vector similarity search.

        Args:
            query: SQL query to execute
            vector: Query vector for similarity search
            field: Vector field to search against
            collection: Collection to query
            vector_results: Results from vector search to filter SQL results

        Returns:
            Query results

        Raises:
            QueryError: If the query fails
        """
        try:
            # Build SQL endpoint URL
            sql_url = f"{self.base_url}/{collection}/sql?aggregationMode=facet"

            # Build SQL query with vector results
            doc_ids = vector_results.get_doc_ids()

            # Execute SQL query using aiohttp
            async with aiohttp.ClientSession() as session:
                # Add vector result filtering
                stmt = query  # Start with original query

                # Check if query already has WHERE clause
                has_where = "WHERE" in stmt.upper()
                has_limit = "LIMIT" in stmt.upper()

                # Extract limit part if present to reposition it
                limit_part = ""
                if has_limit:
                    # Use case-insensitive find and split
                    limit_index = stmt.upper().find("LIMIT")
                    stmt_before_limit = stmt[:limit_index].strip()
                    limit_part = stmt[limit_index + 5 :].strip()  # +5 to skip "LIMIT"
                    stmt = stmt_before_limit  # This is everything before LIMIT

                # Add WHERE clause at the proper position
                if doc_ids:
                    # Add filter query if present
                    if has_where:
                        stmt = f"{stmt} AND id IN ({','.join(doc_ids)})"
                    else:
                        stmt = f"{stmt} WHERE id IN ({','.join(doc_ids)})"
                else:
                    # No vector search results, return empty result set
                    if has_where:
                        stmt = f"{stmt} AND 1=0"  # Always false condition
                    else:
                        stmt = f"{stmt} WHERE 1=0"  # Always false condition

                # Add limit back at the end if it was present or add default limit
                if limit_part:
                    stmt = f"{stmt} LIMIT {limit_part}"
                elif not has_limit:
                    stmt = f"{stmt} LIMIT 10"

                logger.debug(f"Executing SQL query: {stmt}")
                async with session.post(
                    sql_url,
                    data={"stmt": stmt},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise QueryError(f"SQL query failed: {error_text}")

                    content_type = response.headers.get("Content-Type", "")
                    response_text = await response.text()

                    try:
                        if "application/json" in content_type:
                            response_json = json.loads(response_text)
                        else:
                            # For text/plain responses, try to parse as JSON first
                            try:
                                response_json = json.loads(response_text)
                            except json.JSONDecodeError:
                                # If not JSON, wrap in a basic result structure
                                response_json = {
                                    "result-set": {
                                        "docs": [],
                                        "numFound": 0,
                                        "start": 0,
                                    }
                                }

                        return format_sql_response(response_json)
                    except Exception as e:
                        raise QueryError(
                            f"Failed to parse response: {str(e)}, Response: {response_text[:200]}"
                        )

        except Exception as e:
            if isinstance(e, QueryError):
                raise
            raise QueryError(f"Error executing vector query: {str(e)}")
