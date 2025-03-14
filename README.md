# Solr MCP

A Python package for accessing Apache Solr indexes via Model Context Protocol (MCP). This integration allows AI assistants like Claude to perform powerful search queries against your Solr indexes, combining both keyword and vector search capabilities.

## Features

- **MCP Server**: Implements the Model Context Protocol for integration with AI assistants
- **Hybrid Search**: Combines keyword search precision with vector search semantic understanding
- **Vector Embeddings**: Generates embeddings for documents using Ollama with nomic-embed-text
- **Unified Collections**: Store both document content and vector embeddings in the same collection
- **Docker Integration**: Easy setup with Docker and docker-compose

## Quick Start

1. Clone this repository
2. Start SolrCloud with Docker:
   ```bash
   docker-compose up -d
   ```
3. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install poetry
   poetry install
   ```
4. Process and index the sample document:
   ```bash
   python scripts/process_markdown.py data/bitcoin-whitepaper.md --output data/processed/bitcoin_sections.json
   python scripts/unified_index.py data/processed/bitcoin_sections.json --collection unified
   ```
5. Run the MCP server:
   ```bash
   poetry run python -m solr_mcp.server
   ```

For more detailed setup and usage instructions, see the [QUICKSTART.md](QUICKSTART.md) guide.

## Requirements

- Python 3.10 or higher
- Docker and Docker Compose
- SolrCloud 9.x
- Ollama (for embedding generation)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.