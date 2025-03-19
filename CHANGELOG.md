# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- MCP server implementation
- Solr client with search, vector search, and hybrid search capabilities
- Embedding generation via Ollama using nomic-embed-text
- Docker configuration for SolrCloud and ZooKeeper
- Demo scripts and utilities for testing
- Bitcoin whitepaper as sample document
- Documentation (README, QUICKSTART, CONTRIBUTING)

### Fixed
- Improved search query transformation for better results
- Fixed phrase proximity searches with `~5` operator
- Proper field naming for Solr compatibility
- Enhanced text analysis for hyphenated terms like "double-spending"
- Improved synonym handling in Solr configuration
- Fixed vector search configuration to use built-in capabilities
- Improved error handling in Ollama embedding client with retries
- Added proper timeout and fallback mechanisms for embedding generation
- Fixed Solr schema URL paths in client implementation
- Enhanced Docker healthcheck for Ollama service

### Changed
- Migrated from FastMCP to MCP 1.4.1

## [0.1.0] - 2024-03-17
### Added
- Initial release
- MCP server implementation
- Integration with SolrCloud
- Support for basic search operations
- Vector search capabilities
- Hybrid search functionality
- Embedding generation and indexing