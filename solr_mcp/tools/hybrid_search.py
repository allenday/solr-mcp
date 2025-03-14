"""Solr hybrid search tool definition."""

# Tool definition for Solr hybrid search
SOLR_HYBRID_SEARCH_TOOL = {
    "name": "solr_hybrid_search",
    "description": (
        "Performs a hybrid search combining keyword and vector similarity search in Solr. "
        "This combines the precision of keyword matching with the semantic understanding "
        "of vector search, providing more relevant results for complex queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The text query to search for, used both for keyword matching and generating the vector."
            },
            "collection": {
                "type": "string",
                "description": "The Solr collection to search. If not specified, uses the default collection."
            },
            "vector_field": {
                "type": "string",
                "description": "The field containing the vector embeddings to search against.",
                "default": "embedding"
            },
            "blend_factor": {
                "type": "number",
                "description": "Blending factor for hybrid search (0.0=keyword only, 1.0=vector only). Default is 0.5.",
                "default": 0.5
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Specific fields to return in the results. If not specified, returns all fields."
            },
            "filter_query": {
                "type": "string",
                "description": "Filter query to apply to both searches."
            },
            "rows": {
                "type": "integer",
                "description": "Number of results to return. Default is 10.",
                "default": 10
            }
        },
        "required": ["query"]
    }
}