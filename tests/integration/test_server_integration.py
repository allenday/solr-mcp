"""Integration tests for SolrMCP server."""

import json
import pytest
from solr_mcp.server import SolrMCPServer, main
from solr_mcp.tools.search import SOLR_SEARCH_TOOL
import asyncio
import sys
from unittest.mock import patch

@pytest.fixture
def server():
    """Create a server instance for testing."""
    return SolrMCPServer(debug=True)

@pytest.mark.asyncio
async def test_server_initialization(server):
    """Test server initialization with live Solr."""
    # Verify server is initialized
    assert server.server is not None
    assert server.solr_client is not None
    
    # Verify tool registration
    tools = await server.server.list_tools()
    tool_names = [tool.name for tool in tools]
    assert SOLR_SEARCH_TOOL["name"] in tool_names

@pytest.mark.asyncio
async def test_server_search_basic(server):
    """Test basic search functionality through server."""
    # Test basic search
    result = await server.handle_search("*:*", rows=1)
    result_dict = json.loads(result)
    
    assert "numFound" in result_dict
    assert "docs" in result_dict
    assert isinstance(result_dict["docs"], list)

@pytest.mark.asyncio
async def test_server_search_with_params(server):
    """Test search with various parameters through server."""
    # Test search with parameters
    result = await server.handle_search(
        query="*:*",
        fields=["id", "title"],
        filters=["id:*"],
        sort="id asc",
        rows=5
    )
    result_dict = json.loads(result)
    
    assert result_dict["numFound"] >= 0
    if result_dict["docs"]:
        # Verify we got the requested fields
        doc = result_dict["docs"][0]
        assert "id" in doc
        assert len(doc.keys()) <= 2  # Should only have id and possibly title

@pytest.mark.asyncio
async def test_server_search_pagination(server):
    """Test search pagination through server."""
    # Get total count
    result1 = await server.handle_search("*:*", rows=0)
    total_docs = json.loads(result1)["numFound"]
    
    if total_docs > 0:
        # Get first page
        page_size = 2
        result2 = await server.handle_search("*:*", rows=page_size, start=0)
        page1 = json.loads(result2)
        
        # Get second page
        result3 = await server.handle_search("*:*", rows=page_size, start=page_size)
        page2 = json.loads(result3)
        
        # Verify pagination
        assert len(page1["docs"]) <= page_size
        if total_docs > page_size:
            assert len(page2["docs"]) > 0
            # Verify we got different documents
            page1_ids = {doc["id"] for doc in page1["docs"]}
            page2_ids = {doc["id"] for doc in page2["docs"]}
            assert not page1_ids.intersection(page2_ids)

@pytest.mark.asyncio
async def test_server_error_handling(server):
    """Test error handling in server."""
    # Test invalid collection
    with pytest.raises(Exception):
        await server.handle_search("*:*", collection="nonexistent_collection")
    
    # Test malformed filter syntax
    with pytest.raises(Exception):
        await server.handle_search("*:*", filters="{malformed json")
    
    # Test invalid sort field
    with pytest.raises(Exception):
        await server.handle_search("*:*", sort="nonexistent_field asc")

@pytest.mark.asyncio
async def test_server_run(server):
    """Test server run method."""
    # Mock the server's run method to avoid actually starting the server
    with patch.object(server.server, 'run') as mock_run:
        await server.run()
        mock_run.assert_called_once_with("stdio")

def test_main():
    """Test main function."""
    # Mock sys.argv and server run
    test_args = ['solr_mcp', '--debug', '--config', 'test_config.yaml']
    with patch.object(sys, 'argv', test_args), \
         patch('solr_mcp.server.SolrMCPServer') as mock_server:
        
        # Call main
        main()
        
        # Verify server was created with correct args
        mock_server.assert_called_once_with(config_path='test_config.yaml', debug=True)
        # Verify server.run was called
        mock_server.return_value.server.run.assert_called_once()

@pytest.mark.asyncio
async def test_server_search_parameter_variations(server):
    """Test various parameter combinations for search."""
    # Test different sort orders
    sort_tests = [
        "score desc",  # Default sort
        "score asc",   # Reverse score sort
        "id asc",      # Sort by ID ascending
        "id desc",     # Sort by ID descending
        "date_indexed_dt desc"  # Sort by date descending
    ]
    for sort_param in sort_tests:
        result = await server.handle_search("*:*", sort=sort_param, rows=2)
        result_dict = json.loads(result)
        assert "docs" in result_dict
        assert isinstance(result_dict["docs"], list)
    
    # Test field selection variations
    field_tests = [
        None,                   # All fields
        ["id"],                # Single field
        ["id", "title"],       # Multiple fields
        ["id", "nonexistent"], # Mix of existing and nonexistent fields
    ]
    for fields in field_tests:
        result = await server.handle_search("*:*", fields=fields, rows=1)
        result_dict = json.loads(result)
        if result_dict["docs"]:
            doc = result_dict["docs"][0]
            if fields:
                # Should only have the requested fields (except nonexistent ones)
                assert all(key in fields or key == "_version_" for key in doc.keys())
            else:
                # Should have multiple fields when no field list specified
                assert len(doc.keys()) > 1
    
    # Test filter variations
    filter_tests = [
        None,                    # No filters
        "id:*",                 # Single filter
        ["id:*", "title:*"],    # Multiple filters
        'id:*',                 # String filter
        {"id": "*"},            # Dict filter
        ["id:*"]                # List with single filter
    ]
    for filters in filter_tests:
        result = await server.handle_search("*:*", filters=filters, rows=1)
        result_dict = json.loads(result)
        assert "docs" in result_dict
    
    # Test pagination edge cases
    pagination_tests = [
        (0, 0),     # Zero rows
        (1, 0),     # First item
        (10, 0),    # First page
        (10, 10),   # Second page
        (100, 0),   # Large page
        (5, 1000)   # Start beyond total results
    ]
    for rows, start in pagination_tests:
        result = await server.handle_search("*:*", rows=rows, start=start)
        result_dict = json.loads(result)
        assert "docs" in result_dict
        assert len(result_dict["docs"]) <= rows  # Should not exceed requested rows
        
        # For zero rows, verify empty result
        if rows == 0:
            assert len(result_dict["docs"]) == 0
            
    # Test invalid parameters (should raise exceptions)
    with pytest.raises(Exception):
        await server.handle_search("*:*", rows=-1)  # Negative rows
        
    with pytest.raises(Exception):
        await server.handle_search("*:*", start=-1)  # Negative start 

@pytest.mark.asyncio
async def test_server_search_parameter_effects(server):
    """Test that search parameters have their intended effects on results."""
    
    # First get a baseline set of documents to work with
    result = await server.handle_search("*:*", rows=10)
    base_docs = json.loads(result)["docs"]
    if not base_docs:
        pytest.skip("No documents found in Solr for testing")

    # Test that sort actually affects ordering
    sort_fields = ["id", "score", "date_indexed_dt"]
    for field in sort_fields:
        # Get ascending and descending results
        asc_result = await server.handle_search("*:*", sort=f"{field} asc", rows=5)
        desc_result = await server.handle_search("*:*", sort=f"{field} desc", rows=5)
        
        asc_docs = json.loads(asc_result)["docs"]
        desc_docs = json.loads(desc_result)["docs"]
        
        if asc_docs and desc_docs:
            # Verify ordering is different between asc and desc
            if field in asc_docs[0] and field in desc_docs[0]:
                # Skip comparison if all values are identical
                all_values = set(doc[field] for doc in asc_docs + desc_docs if field in doc)
                if len(all_values) > 1:  # Only compare if we have different values
                    assert asc_docs[0][field] != desc_docs[0][field], f"Sort {field} had no effect"
                
                # Verify ascending order
                for i in range(len(asc_docs)-1):
                    if field in asc_docs[i] and field in asc_docs[i+1]:
                        assert asc_docs[i][field] <= asc_docs[i+1][field], f"Ascending sort failed for {field}"
                
                # Verify descending order
                for i in range(len(desc_docs)-1):
                    if field in desc_docs[i] and field in desc_docs[i+1]:
                        assert desc_docs[i][field] >= desc_docs[i+1][field], f"Descending sort failed for {field}"

    # Test that start parameter actually skips records
    all_result = await server.handle_search("*:*", sort="id asc", rows=5, start=0)
    skip_result = await server.handle_search("*:*", sort="id asc", rows=3, start=2)
    
    all_docs = json.loads(all_result)["docs"]
    skip_docs = json.loads(skip_result)["docs"]
    
    if len(all_docs) >= 5:
        # Verify that skip_docs matches the subset of all_docs starting at position 2
        assert all_docs[2]["id"] == skip_docs[0]["id"], "Start parameter did not skip correct number of records"
        
    # Test that rows parameter limits results
    for rows in [1, 2, 5]:
        result = await server.handle_search("*:*", rows=rows)
        docs = json.loads(result)["docs"]
        assert len(docs) <= rows, f"Results exceeded requested rows={rows}"

    # Test that field selection works
    test_fields = ["id", "title", "text"]
    result = await server.handle_search("*:*", fields=test_fields, rows=1)
    docs = json.loads(result)["docs"]
    if docs:
        doc_fields = set(docs[0].keys()) - {"_version_"}  # Ignore _version_ field
        assert doc_fields.issubset(set(test_fields)), "Got fields that weren't requested"

    # Test that filter affects results
    # First get a value we know exists
    if base_docs and "id" in base_docs[0]:
        test_id = base_docs[0]["id"]
        # Search with filter
        result = await server.handle_search("*:*", filters=[f"id:{test_id}"], rows=10)
        filtered_docs = json.loads(result)["docs"]
        assert len(filtered_docs) == 1, "Filter didn't restrict to single document"
        assert filtered_docs[0]["id"] == test_id, "Filter returned wrong document"

@pytest.mark.asyncio
async def test_mcp_client_request_pattern(server):
    """Test search handling in the pattern that MCP clients use."""
    # Simulate MCP client request format
    test_cases = [
        {
            "name": "Basic search",
            "request": {
                "query": "*:*",
                "rows": 5
            },
            "validate": lambda result: (
                isinstance(result, str) and
                "docs" in json.loads(result) and
                "numFound" in json.loads(result)
            )
        },
        {
            "name": "Search with all parameters",
            "request": {
                "query": "*:*",
                "collection": "unified",
                "fields": ["id", "title", "content"],
                "filters": ["id:*"],
                "sort": "score desc",
                "rows": 3,
                "start": 1
            },
            "validate": lambda result: (
                isinstance(result, str) and
                all(key in json.loads(result) for key in ["docs", "numFound", "start"]) and
                len(json.loads(result)["docs"]) <= 3 and
                all("id" in doc for doc in json.loads(result)["docs"])
            )
        },
        {
            "name": "Search with filters",
            "request": {
                "query": "*:*",
                "filters": ["title:*"],
                "rows": 2
            },
            "validate": lambda result: (
                isinstance(result, str) and
                len(json.loads(result)["docs"]) <= 2 and
                all("title" in doc for doc in json.loads(result)["docs"])
            )
        },
        {
            "name": "Search with sorting",
            "request": {
                "query": "*:*",
                "sort": "id asc",
                "rows": 4
            },
            "validate": lambda result: (
                isinstance(result, str) and
                len(json.loads(result)["docs"]) <= 4 and
                # Verify docs are sorted by id
                all(
                    json.loads(result)["docs"][i]["id"] <= json.loads(result)["docs"][i+1]["id"]
                    for i in range(len(json.loads(result)["docs"])-1)
                )
            )
        }
    ]
    
    for test_case in test_cases:
        print(f"\nExecuting test case: {test_case['name']}")
        result = await server.handle_search(**test_case["request"])
        
        # Validate response format and content
        assert test_case["validate"](result), f"Validation failed for {test_case['name']}"
        
        # Common validations for all responses
        response_data = json.loads(result)
        assert "docs" in response_data, "Response missing 'docs' field"
        assert "numFound" in response_data, "Response missing 'numFound' field"
        assert isinstance(response_data["docs"], list), "'docs' should be a list"
        assert isinstance(response_data["numFound"], int), "'numFound' should be an integer"
        
        # Verify response matches request parameters
        if "rows" in test_case["request"]:
            assert len(response_data["docs"]) <= test_case["request"]["rows"], "Got more docs than requested"
        if "start" in test_case["request"]:
            assert response_data["start"] == test_case["request"]["start"], "Start parameter not reflected in response"
        if "fields" in test_case["request"]:
            for doc in response_data["docs"]:
                assert all(
                    key in test_case["request"]["fields"] or key == "_version_"
                    for key in doc.keys()
                ), "Got fields that weren't requested" 