"""Solr vector search tool definition."""

# Tool definition for Solr vector search
SOLR_VECTOR_SEARCH_TOOL = {
    "name": "solr_vector_search",
    "description": (
        "Performs a semantic vector search using dense vectors and KNN in Solr. "
        "This tool is ideal for finding semantically similar content based on embeddings. "
        "Use this for semantic search, recommendation, or finding similar documents "
        "based on meaning rather than keywords."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "vector": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "The dense vector representation (embedding) to search with."
            },
            "vector_field": {
                "type": "string",
                "description": "The field containing the vector embeddings to search against.",
                "default": "embedding"
            },
            "collection": {
                "type": "string",
                "description": "The Solr collection to search. If not specified, uses the default collection."
            },
            "k": {
                "type": "integer",
                "description": "Number of nearest neighbors to retrieve. Default is 10.",
                "default": 10
            },
            "filter_query": {
                "type": "string",
                "description": "Optional filter query to limit the search space."
            },
            "return_fields": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Fields to return in the results. If not specified, returns all fields."
            }
        },
        "required": ["vector"]
    }
}