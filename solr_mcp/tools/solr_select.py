"""Tool for executing SQL SELECT queries against Solr."""

from typing import Dict

from solr_mcp.tools.tool_decorator import tool


@tool()
async def execute_select_query(mcp, query: str) -> Dict:
    """Execute SQL queries against Solr collections.

    Executes SQL queries against Solr collections with the following Solr-specific behaviors:

    Collection/Field Rules:
    - Collections are used as table names (case-insensitive)
    - Field names are case-sensitive and must exist in Solr schema
    - SELECT * only allowed with LIMIT clause
    - Unlimited queries restricted to docValues-enabled fields
    - Reserved words must be backtick-escaped

    WHERE Clause Differences:
    - Field must be on one side of predicate
    - No comparing two constants or two fields
    - No subqueries
    - Solr syntax in values:
      - '[0 TO 100]' for ranges
      - '(term1 term2)' for non-phrase OR search
    - String literals use single-quotes

    Supported Features:
    - Operators: =, <>, >, >=, <, <=, IN, LIKE (wildcards), BETWEEN, IS [NOT] NULL
    - Functions: COUNT(*), COUNT(DISTINCT), MIN, MAX, SUM, AVG
    - GROUP BY: Uses faceting (fast) for low cardinality, map_reduce (slow) for high cardinality
    - ORDER BY: Requires docValues-enabled fields
    - LIMIT/OFFSET: Use 'OFFSET x FETCH NEXT y ROWS ONLY' syntax
      - Performance of OFFSET degrades beyond 10k docs per shard

    Args:
        mcp: SolrMCPServer instance
        query: SQL query to execute

    Returns:
        Query results
    """
    solr_client = mcp.solr_client
    return await solr_client.execute_select_query(query)
