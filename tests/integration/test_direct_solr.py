"""Direct integration tests for Solr MCP functionality.

These tests interact directly with the Solr client, bypassing the MCP server.
"""

import asyncio
import logging
import os

# Add the project root to the path
import sys
import time

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from solr_mcp.solr.client import SolrClient
from solr_mcp.solr.config import SolrConfig
from solr_mcp.vector_provider import OllamaVectorProvider

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Get test config from environment or use defaults
TEST_COLLECTION = os.getenv("TEST_COLLECTION", "unified")
TEST_VECTOR_FIELD = os.getenv("TEST_VECTOR_FIELD", "embedding")
SOLR_BASE_URL = os.getenv("SOLR_BASE_URL", "http://localhost:8983/solr")


@pytest_asyncio.fixture
async def solr_client():
    """Create SolrClient for testing."""
    config = SolrConfig(
        solr_base_url=SOLR_BASE_URL,
        zookeeper_hosts=["localhost:2181"],
        default_collection=TEST_COLLECTION,
    )
    client = SolrClient(config=config)
    try:
        yield client
    finally:
        if hasattr(client, "close"):
            await client.close()


@pytest.mark.asyncio
async def test_basic_search(solr_client):
    """Test basic search functionality."""
    # Use the SQL query instead of search for consistency
    result = await solr_client.execute_select_query(
        query=f"SELECT * FROM {TEST_COLLECTION} WHERE id IS NOT NULL LIMIT 5"
    )

    # The result is already a dictionary
    result_dict = result
    logger.info(
        f"Basic search returned {result_dict.get('result-set', {}).get('numFound', 0)} results"
    )

    assert "result-set" in result_dict, "Result should contain 'result-set' key"
    assert "docs" in result_dict["result-set"], "result-set should contain 'docs' key"
    # There should be at least some results for "double spend" in the Bitcoin whitepaper
    assert len(result_dict["result-set"]["docs"]) > 0, "Should return some results"


@pytest.mark.asyncio
async def test_search_with_filters(solr_client):
    """Test search with filters/WHERE clause."""
    # Use SQL query with WHERE clause
    result = await solr_client.execute_select_query(
        query=f"SELECT * FROM {TEST_COLLECTION} WHERE text:blockchain AND id IS NOT NULL LIMIT 5"
    )

    # The result is already a dictionary
    result_dict = result
    logger.info(
        f"Filtered search returned {result_dict.get('result-set', {}).get('numFound', 0)} results"
    )

    assert "result-set" in result_dict, "Result should contain 'result-set' key"
    assert "docs" in result_dict["result-set"], "result-set should contain 'docs' key"
    assert len(result_dict["result-set"]["docs"]) > 0, "Should return some results"


@pytest.mark.asyncio
async def test_vector_search(solr_client):
    """Test vector search functionality."""
    # Initialize vector provider
    vector_provider = OllamaVectorProvider()
    # Generate vector for search text
    search_text = "double spend attack"
    vector = await vector_provider.get_vector(search_text)

    # Perform vector search using the execute_vector_select_query method
    result = await solr_client.execute_vector_select_query(
        query=f"SELECT * FROM {TEST_COLLECTION} LIMIT 5",
        vector=vector,
        field=TEST_VECTOR_FIELD,
    )

    # The result is already a dictionary
    result_dict = result
    logger.info(
        f"Vector search returned {result_dict.get('result-set', {}).get('numFound', 0)} results"
    )

    assert "result-set" in result_dict, "Result should contain 'result-set' key"
    assert "docs" in result_dict["result-set"], "result-set should contain 'docs' key"
    # Note: vector search may not have results with test data, so we just check the docs array exists


@pytest.mark.asyncio
async def test_vector_search_with_filter(solr_client):
    """Test vector search with filters."""
    # Initialize vector provider
    vector_provider = OllamaVectorProvider()
    # Generate vector for search text
    search_text = "double spend attack"
    vector = await vector_provider.get_vector(search_text)

    # Perform vector search with WHERE clause using execute_vector_select_query
    result = await solr_client.execute_vector_select_query(
        query=f"SELECT * FROM {TEST_COLLECTION} WHERE id IS NOT NULL LIMIT 5",
        vector=vector,
        field=TEST_VECTOR_FIELD,
    )

    # The result is already a dictionary
    result_dict = result
    logger.info(
        f"Vector search with filter returned {result_dict.get('result-set', {}).get('numFound', 0)} results"
    )

    assert "result-set" in result_dict, "Result should contain 'result-set' key"
    assert "docs" in result_dict["result-set"], "result-set should contain 'docs' key"
    # Note: vector search may not have results with test data, so we just check the docs array exists


@pytest.mark.asyncio
async def test_hybrid_search(solr_client):
    """Test hybrid search (keyword + vector)."""
    # Use semantic select as hybrid search is no longer available directly
    result = await solr_client.execute_semantic_select_query(
        query=f"SELECT * FROM {TEST_COLLECTION} LIMIT 5",
        text="bitcoin blockchain",
        field=TEST_VECTOR_FIELD,
    )

    # The result is already a dictionary
    result_dict = result
    logger.info(
        f"Hybrid search returned {result_dict.get('result-set', {}).get('numFound', 0)} results"
    )

    assert "result-set" in result_dict, "Result should contain 'result-set' key"
    assert "docs" in result_dict["result-set"], "result-set should contain 'docs' key"

    # If we have results, check that they have scores
    if (
        result_dict["result-set"]["docs"]
        and len(result_dict["result-set"]["docs"]) > 0
        and result_dict["result-set"]["docs"][0].get("EOF") is not True
    ):
        assert (
            "score" in result_dict["result-set"]["docs"][0]
        ), "Results should have scores"


@pytest.mark.asyncio
async def test_sql_execute(solr_client):
    """Test SQL query execution with WHERE clause."""
    # Create a SQL query with WHERE clause before LIMIT
    query = f"SELECT id, title FROM {TEST_COLLECTION} WHERE id IS NOT NULL LIMIT 5"

    # Execute the query via the client's internal query executor
    result = await solr_client.execute_select_query(query)

    logger.info(f"SQL query result: {result}")

    assert "result-set" in result, "Result should contain 'result-set' key"
    assert "docs" in result["result-set"], "Result should contain 'docs' key"
    assert len(result["result-set"]["docs"]) > 0, "Should return some results"
