"""Solr search tool definition."""

# Tool definition for Solr search
SOLR_SEARCH_TOOL = {
    "name": "solr_search",
    "description": (
        "Performs a search across documents in Solr, ideal for retrieving relevant "
        "documents, articles, or structured data. Use this tool to find information "
        "stored in the search index based on keywords, phrases, or specific field values. "
        "Supports filtering, field selection, pagination, and sorting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string. Can use Solr query syntax for advanced queries."
            },
            "collection": {
                "type": "string",
                "description": "The Solr collection to search. If not specified, uses the default collection."
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Specific fields to return in the results. If not specified, returns all fields."
            },
            "filters": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Filter queries to apply (e.g. ['category:books', 'year:[2020 TO 2023]'])."
            },
            "rows": {
                "type": "integer",
                "description": "Number of results to return. Default is 10.",
                "default": 10
            },
            "start": {
                "type": "integer",
                "description": "Start offset for pagination. Default is 0.",
                "default": 0
            },
            "sort": {
                "type": "string",
                "description": "Sort order (e.g. 'score desc', 'date_created desc')."
            }
        },
        "required": ["query"]
    }
}