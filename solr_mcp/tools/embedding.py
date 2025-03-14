"""Solr embedding and indexing tool definition."""

# Tool definition for document embedding and indexing
SOLR_EMBEDDING_TOOL = {
    "name": "solr_embed_and_index",
    "description": (
        "Generates embeddings for text using Ollama's nomic-embed-text model "
        "and indexes the document with its vector representation in Solr. "
        "Use this tool to add documents to the search index with vector embeddings "
        "for semantic search capabilities."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "document": {
                "type": "object",
                "description": "The document to index. Must contain at least 'id' and 'text' fields."
            },
            "collection": {
                "type": "string",
                "description": "The Solr collection to index the document in. If not specified, uses the default collection."
            },
            "text_field": {
                "type": "string",
                "description": "The field in the document to use for generating the embedding. Default is 'text'.",
                "default": "text"
            },
            "vector_field": {
                "type": "string",
                "description": "The field name to store the vector embedding. Default is 'embedding'.",
                "default": "embedding"
            },
            "commit": {
                "type": "boolean",
                "description": "Whether to commit the index immediately. Default is true.",
                "default": True
            }
        },
        "required": ["document"]
    }
}

# Tool definition for batch embedding and indexing
SOLR_BATCH_EMBEDDING_TOOL = {
    "name": "solr_batch_embed_and_index",
    "description": (
        "Generates embeddings for multiple documents using Ollama's nomic-embed-text model "
        "and indexes them with their vector representations in Solr. "
        "Use this tool to efficiently add multiple documents to the search index with vector "
        "embeddings for semantic search capabilities."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": {
                    "type": "object"
                },
                "description": "The documents to index. Each must contain at least 'id' and 'text' fields."
            },
            "collection": {
                "type": "string",
                "description": "The Solr collection to index the documents in. If not specified, uses the default collection."
            },
            "text_field": {
                "type": "string",
                "description": "The field in the documents to use for generating embeddings. Default is 'text'.",
                "default": "text"
            },
            "vector_field": {
                "type": "string",
                "description": "The field name to store the vector embeddings. Default is 'embedding'.",
                "default": "embedding"
            },
            "commit": {
                "type": "boolean",
                "description": "Whether to commit the index immediately. Default is true.",
                "default": True
            }
        },
        "required": ["documents"]
    }
}