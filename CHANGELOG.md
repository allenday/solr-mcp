# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- FastMCP server implementation
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