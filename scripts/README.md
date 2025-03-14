# Utility Scripts for Solr MCP

This directory contains utility scripts for working with the Solr MCP server.

## Scripts

### demo_search.py

Demonstrates how to use the MCP client to search for information using both text search and vector search.

**Usage:**

```bash
# Text search
python demo_search.py "bitcoin mining" --collection vectors

# Vector (semantic) search
python demo_search.py "How does Bitcoin prevent double-spending?" --vector --collection vectors

# Specify number of results
python demo_search.py "blockchain" --results 10
```

The script shows how to connect to the MCP server, perform different types of searches, and display the results.

### process_markdown.py

Splits markdown files into sections based on headings and converts them to JSON documents ready for Solr indexing.

**Usage:**

```bash
# Process a markdown file and output to stdout
python process_markdown.py data/document.md

# Process a markdown file and save to a JSON file
python process_markdown.py data/document.md --output data/processed/document_sections.json
```

The script supports markdown files with YAML frontmatter. The frontmatter metadata will be added to each section document.

### index_documents.py

Indexes documents from a JSON file into Solr with vector embeddings generated using Ollama's nomic-embed-text model.

**Usage:**

```bash
# Index documents into the default collection
python index_documents.py data/processed/document_sections.json

# Index documents into a specific collection
python index_documents.py data/processed/document_sections.json --collection my_collection

# Index documents without committing (useful for batch indexing)
python index_documents.py data/processed/document_sections.json --no-commit
```

## Workflow Example

1. Process a markdown file:

```bash
python process_markdown.py data/document.md --output data/processed/document_sections.json
```

2. Start the Docker containers (if not already running):

```bash
docker-compose up -d
```

3. Index the documents with vector embeddings:

```bash
python index_documents.py data/processed/document_sections.json --collection vectors
```

4. Use the MCP server to search the documents:

```bash
# Configure Claude Desktop to use the MCP server
# Then ask questions about the document
```