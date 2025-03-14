# Solr MCP Quick Start Guide

This guide will help you get up and running with the Solr MCP server quickly.

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/allenday/solr-mcp.git
cd solr-mcp
```

## Step 2: Start SolrCloud with Docker

```bash
docker-compose up -d
```

This will start a SolrCloud instance with ZooKeeper and Ollama for embedding generation.

Verify that Solr is running by visiting: http://localhost:8983/solr/

## Step 3: Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Poetry
pip install poetry

# Install dependencies
poetry install
```

## Step 4: Process and Index Sample Documents

The repository includes the Bitcoin whitepaper as a sample document. Let's process and index it:

```bash
# Process the Markdown file into sections
python scripts/process_markdown.py data/bitcoin-whitepaper.md --output data/processed/bitcoin_sections.json

# Create a unified collection
python scripts/create_unified_collection.py unified

# Index the sections with embeddings
python scripts/unified_index.py data/processed/bitcoin_sections.json --collection unified
```

## Step 5: Run the MCP Server

```bash
poetry run python -m solr_mcp.server
```

By default, the server will run on http://localhost:8000

## Step 6: Test the Search Functionality

You can test the different search capabilities using the demo scripts:

```bash
# Test keyword search
python scripts/simple_search.py "double spend" --collection unified

# Test vector search
python scripts/vector_search.py "how does bitcoin prevent fraud" --collection unified

# Test hybrid search (combining keyword and vector)
python scripts/simple_mcp_test.py
```

## Using with Claude Desktop

To use the MCP server with Claude Desktop:

1. Make sure the MCP server is running
2. In Claude Desktop, go to Settings > Tools
3. Add a new tool with:
   - Name: Solr Search
   - URL: http://localhost:8000
   - Working Directory: /path/to/solr-mcp

Now you can ask Claude queries like:
- "Search for information about double spending in the Bitcoin whitepaper"
- "Find sections related to consensus mechanisms"
- "What does the whitepaper say about transaction verification?"

## Troubleshooting

If you encounter issues:

1. Check that Solr is running: http://localhost:8983/solr/
2. Verify the collection exists: http://localhost:8983/solr/#/~collections
3. Run the diagnostic script: `python scripts/diagnose_search.py`
4. Check the server logs for errors