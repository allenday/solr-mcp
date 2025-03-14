"""Solr facets tool definition."""

# Tool definition for Solr facets
SOLR_FACETS_TOOL = {
    "name": "solr_facets",
    "description": (
        "Retrieves facet information from Solr for a given query. Facets provide "
        "counts of terms or ranges within search results, allowing for analytics "
        "and guided navigation. Use this tool to understand the distribution of "
        "values across fields in the search results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to get facets for."
            },
            "facet_fields": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Fields to get facet counts for."
            },
            "collection": {
                "type": "string",
                "description": "The Solr collection to use. If not specified, uses the default collection."
            },
            "facet_limit": {
                "type": "integer",
                "description": "Maximum number of facet values to return per field. Default is 10.",
                "default": 10
            },
            "facet_mincount": {
                "type": "integer",
                "description": "Minimum count for facet values to be included. Default is 1.",
                "default": 1
            }
        },
        "required": ["query", "facet_fields"]
    }
}