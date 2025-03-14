# Data Examples for Solr MCP

This directory contains example data for testing and demonstrating the Solr MCP server.

## Bitcoin Whitepaper Example

The Bitcoin whitepaper by Satoshi Nakamoto is included as an example document for testing semantic search capabilities.

### Files

- `bitcoin-whitepaper.md`: The original Bitcoin whitepaper in markdown format
- `processed/bitcoin_sections.json`: The whitepaper split into sections, ready for indexing
- `processed/bitcoin_metadata.md`: Example with proper YAML frontmatter metadata
- `processed/bitcoin_metadata.json`: Processed version with metadata included

### Using the Bitcoin Whitepaper Example

1. **Process the whitepaper into sections** (already done):

```bash
python scripts/process_markdown.py data/bitcoin-whitepaper.md --output data/processed/bitcoin_sections.json
```

2. **Start the Docker containers**:

```bash
docker-compose up -d
```

3. **Index the sections with vector embeddings**:

```bash
python scripts/index_documents.py data/processed/bitcoin_sections.json --collection vectors
```

4. **Search using Claude Desktop**:

Configure Claude Desktop to use your MCP server, then ask questions like:

- "How does Bitcoin solve the double-spending problem?"
- "Explain Bitcoin's proof-of-work system"
- "What is the incentive for nodes to support the network?"

The MCP server will find the most semantically relevant sections from the whitepaper and return them to Claude.

## Adding Your Own Documents

You can add your own documents to this directory and process them using the same workflow:

1. Add markdown documents to the `data/` directory
2. Process them into sections:

```bash
python scripts/process_markdown.py data/your-document.md --output data/processed/your-document_sections.json
```

3. Index them into Solr:

```bash
python scripts/index_documents.py data/processed/your-document_sections.json --collection vectors
```

### YAML Frontmatter

For better document organization, add YAML frontmatter to your markdown files:

```markdown
---
title: "Document Title"
author: "Author Name"
date: "2023-01-01"
tags:
  - tag1
  - tag2
categories:
  - category1
  - category2
---

# Your Document Content
...
```

This metadata will be included in the indexed documents and can be used for filtering searches.