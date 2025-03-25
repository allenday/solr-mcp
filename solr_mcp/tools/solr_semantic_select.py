"""Tool for executing semantic search queries against Solr collections."""

from typing import Dict

from solr_mcp.tools.tool_decorator import tool

@tool()
async def execute_semantic_select_query(mcp, query: str, text: str, field: str) -> Dict:
    """Execute semantic search queries against Solr collections.
    
    Extends solr_select tool with semantic search capabilities.
    
    Additional Parameters:
    - text: Natural language text that is converted to vector, which will be used to match against other vector fields
    - field: Name of the DenseVector field to search against
    
    The query results will be ranked based on semantic similarity to the provided text. Therefore, ORDER BY is not allowed.
    
    Collection/Field Rules:
    - Vector field must be a dense_vector field type
    - The specified field must exist in the collection schema
    
    Supported Features:
    - All standard SELECT query features except ORDER BY
      - Results are ordered by semantic similarity
    - Hybrid search combining keyword (SQL WHERE clauses) and vector distance (text parameter)
    
    Args:
        mcp: SolrMCPServer instance
        query: SQL query to execute
        text: Search text to convert to vector
        field: Name of the DenseVector field to search against
    
    Returns:
        Query results
    """
    solr_client = mcp.solr_client
    return await solr_client.execute_semantic_select_query(query, text, field)
