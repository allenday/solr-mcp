"""Tool for executing vector search queries against Solr collections."""

from typing import Dict, List

from solr_mcp.tools.tool_decorator import tool

@tool()
async def execute_vector_select_query(mcp, query: str, vector: List[float]) -> Dict:
    """Execute vector search queries against Solr collections.
    
    Extends solr_select tool with vector search capabilities.
    
    Additional Parameters:
    - vector: Used to match against the collection's vector field, intended for embeddings search.
    
    The query results will be ranked based on distance to the provided vector. Therefore, ORDER BY is not allowed.
    
    Collection/Field Rules:
    - Vector field is a dense_vector field type
    
    Supported Features:
    - All standard SELECT query features except ORDER BY
      - Results are ordered by vector distance
    - Hybrid search combining keyword (SQL WHERE clauses) and vector distance (vector parameter)
    
    Args:
        mcp: SolrMCPServer instance
        query: SQL query to execute
        vector: Query vector for similarity search
    
    Returns:
        Query results
    """
    solr_client = mcp.solr_client
    return await solr_client.execute_vector_select_query(query, vector)
