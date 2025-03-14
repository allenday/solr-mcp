# Contributing to Solr MCP

Thank you for your interest in contributing to the Solr MCP project! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment as described in the README
4. Create a new branch for your changes

## Development Workflow

1. Make your changes in your branch
2. Write or update tests for your changes
3. Ensure all tests pass
4. Format your code using Black and isort
5. Submit a pull request

## Code Style Guidelines

- Follow PEP 8 style guide with 88-char line length (Black formatter)
- Use type hints consistently (Python 3.9+ typing)
- Group imports: stdlib → third-party → local
- Document functions, classes, and tools with docstrings

## Testing

Run the test suite with:

```bash
poetry run pytest
```

For test coverage:

```bash
poetry run pytest --cov=solr_mcp
```

## Submitting Pull Requests

1. Update the README.md with details of changes if appropriate
2. Update the CHANGELOG.md following the Keep a Changelog format
3. The version will be updated according to Semantic Versioning by the maintainers
4. Once you have the sign-off of a maintainer, your PR will be merged

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.