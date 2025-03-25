# CLAUDE.md - Solr MCP Server Guide (Python)

## IMPORTANT NOTE
Before using the search tools, make sure the Bitcoin whitepaper content is properly indexed in the unified collection!
If search queries like "double spend" return no results, you may need to reindex the content:

```bash
python scripts/process_markdown.py data/bitcoin-whitepaper.md --output data/processed/bitcoin_sections.json
python scripts/unified_index.py data/processed/bitcoin_sections.json --collection unified
```

## Project Structure
- Python-based MCP server integrating with SolrCloud
- Uses MCP 1.4.1 framework for protocol implementation
- Provides document search and knowledge retrieval for AI systems
- Supports SolrCloud collections and distributed search
- Vector search/KNN capabilities for semantic search

## Environment Setup
- Python 3.10: `python3.10 -m venv venv`
- Activate: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
- Install Poetry: `pip install poetry`

## Build Commands
- Install all deps: `poetry install`
- Run server: `poetry run python -m solr_mcp.server`
- Debug mode: `poetry run python -m solr_mcp.server --debug`
- Package: `poetry build`

## Test Commands
- Run tests: `poetry run pytest`
- Single test: `poetry run pytest tests/test_file.py::test_function`
- Coverage: `poetry run pytest --cov=solr_mcp`
- Lint: `poetry run flake8 solr_mcp tests`
- Type check: `poetry run mypy solr_mcp tests`
- Format code: `poetry run black solr_mcp tests`
- Sort imports: `poetry run isort solr_mcp tests`
- Run all formatting: `poetry run black solr_mcp tests && poetry run isort solr_mcp tests`

## Docker Commands
- Start SolrCloud: `docker-compose up -d`
- Check logs: `docker-compose logs -f`
- Solr UI: http://localhost:8983/solr/
- Stop SolrCloud: `docker-compose down`
- Cleanup volumes: `docker-compose down -v`

## SolrCloud Integration
- Connection via pysolr with ZooKeeper ensemble
- Support for collection management and configuration
- Handle distributed search with configurable shards and replicas
- Vector search using dense_vector fields and KNN
- Hybrid search combining keyword and vector search capabilities
- Embedding generation via Ollama using nomic-embed-text (768D vectors)
- Unified collections storing both text content and vector embeddings
- Implement retry and fallback logic for resilience

## Code Style Guidelines
- Follow PEP 8 style guide with 88-char line length (Black formatter)
- Use type hints consistently (Python 3.9+ typing)
- Group imports: stdlib → third-party → local
- Document functions, classes and tools with docstrings
- Handle Solr connection errors with appropriate retries
- Log operations with structured logging (JSON format)
- Return well-formatted errors following JSON-RPC 2.0 spec

## Technical Details

Key implementation details:

- Uses MCP 1.4.1 framework for protocol implementation