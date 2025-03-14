"""Solr suggestions tool definition."""

# Tool definition for Solr suggestions
SOLR_SUGGESTIONS_TOOL = {
    "name": "solr_suggestions",
    "description": (
        "Gets search suggestions for a partial query from Solr. Use this tool to "
        "find auto-complete suggestions, spelling corrections, or query expansions "
        "based on the content indexed in Solr."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The partial query text to get suggestions for."
            },
            "collection": {
                "type": "string",
                "description": "The Solr collection to use. If not specified, uses the default collection."
            },
            "suggestion_field": {
                "type": "string",
                "description": "The suggestion handler name in Solr. Default is 'suggest'.",
                "default": "suggest"
            },
            "count": {
                "type": "integer",
                "description": "Number of suggestions to return. Default is 5.",
                "default": 5
            }
        },
        "required": ["query"]
    }
}