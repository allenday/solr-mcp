import pytest
import pytest_asyncio
import os
import httpx

# Get test config from environment or use defaults based on docker-compose.yml
TEST_MCP_URL = os.getenv("TEST_MCP_URL", "http://localhost:8000")
TEST_COLLECTION = os.getenv("TEST_COLLECTION", "unified")
TEST_VECTOR_FIELD = os.getenv("TEST_VECTOR_FIELD", "embedding")

@pytest_asyncio.fixture
async def mcp():
    """Create HTTP client for testing."""
    async with httpx.AsyncClient(base_url=TEST_MCP_URL) as client:
        yield client

@pytest.mark.asyncio
async def test_list_collections(mcp):
    """Test listing collections."""
    response = await mcp.post("/tools/list_collections")
    assert response.status_code == 200
    result = response.json()
    collections = result["collections"]
    assert isinstance(collections, list), "Should return a list of collections"
    assert len(collections) > 0, "Should have at least one collection"
    assert TEST_COLLECTION in collections, f"Test collection {TEST_COLLECTION} should exist"

@pytest.mark.asyncio
async def test_list_fields(mcp):
    """Test listing fields in a collection."""
    response = await mcp.post("/tools/list_fields", json={"collection": TEST_COLLECTION})
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict), "Should return a dictionary"
    assert "fields" in result, "Should have fields key in response"
    assert isinstance(result["fields"], list), "Fields should be a list"
    assert len(result["fields"]) > 0, "Should have at least one field"
    assert any(f["name"] == TEST_VECTOR_FIELD for f in result["fields"]), f"Vector field {TEST_VECTOR_FIELD} should exist"

@pytest.mark.asyncio
async def test_select_query(mcp):
    """Test basic SQL select query."""
    query = f"SELECT * FROM {TEST_COLLECTION} LIMIT 5"
    response = await mcp.post("/tools/select_query", json={"query": query})
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict), "Should return a dictionary"
    assert "result-set" in result, "Should have result-set in response"
    assert "docs" in result["result-set"], "Should have docs in result-set"
    assert len(result["result-set"]["docs"]) <= 5, "Should respect LIMIT clause"

@pytest.mark.asyncio
async def test_semantic_select_query(mcp):
    """Test semantic search query."""
    query = f"SELECT * FROM {TEST_COLLECTION} LIMIT 5"
    search_text = "test semantic search"
    response = await mcp.post("/tools/semantic_select_query", json={
        "query": query,
        "text": search_text
    })
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict), "Should return a dictionary"
    assert "result-set" in result, "Should have result-set in response"
    assert "docs" in result["result-set"], "Should have docs in result-set"
    assert len(result["result-set"]["docs"]) <= 5, "Should respect LIMIT clause"

@pytest.mark.asyncio
async def test_vector_select_query(mcp):
    """Test vector search query."""
    query = f"SELECT * FROM {TEST_COLLECTION} LIMIT 5"
    # Create a test vector matching nomic-embed-text dimensions (768)
    test_vector = [0.1] * 768  # nomic-embed-text uses 768 dimensions
    response = await mcp.post("/tools/vector_select_query", json={
        "query": query,
        "vector": test_vector
    })
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict), "Should return a dictionary"
    assert "result-set" in result, "Should have result-set in response"
    assert "docs" in result["result-set"], "Should have docs in result-set"
    assert len(result["result-set"]["docs"]) <= 5, "Should respect LIMIT clause" 