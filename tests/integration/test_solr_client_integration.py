import json
import pytest
from solr_mcp.solr.client import SolrClient, SolrError
import pysolr

@pytest.fixture
def solr_client():
    """Create a client connected to the live Solr server."""
    return SolrClient()

@pytest.mark.asyncio
async def test_live_connection(solr_client):
    """Test that we can connect to the live Solr server and list collections."""
    collections = solr_client.list_collections()
    assert isinstance(collections, list)
    assert len(collections) > 0
    assert "unified" in collections  # Assuming the default collection exists

@pytest.mark.asyncio
async def test_live_search_basic(solr_client):
    """Test basic search functionality against live Solr."""
    result = await solr_client.search("*:*", rows=1)
    result_dict = json.loads(result)
    
    assert "numFound" in result_dict
    assert "docs" in result_dict
    assert isinstance(result_dict["docs"], list)

@pytest.mark.asyncio
async def test_live_search_with_filters(solr_client):
    """Test search with filters against live Solr."""
    result = await solr_client.search(
        query="*:*",
        fields=["id", "title"],
        filters=["id:*"],  # Filter for documents with an id field
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
async def test_live_search_pagination(solr_client):
    """Test search pagination against live Solr."""
    # Get total count
    result1 = await solr_client.search("*:*", rows=0)
    total_docs = json.loads(result1)["numFound"]
    
    if total_docs > 0:
        # Get first page
        page_size = 2
        result2 = await solr_client.search("*:*", rows=page_size, start=0)
        page1 = json.loads(result2)
        
        # Get second page
        result3 = await solr_client.search("*:*", rows=page_size, start=page_size)
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
async def test_live_field_types(solr_client):
    """Test searching and sorting on different field types."""
    # Test date field ascending and descending
    result_asc = await solr_client.search(
        "*:*",
        sort="date_indexed_dt asc",
        rows=5
    )
    result_desc = await solr_client.search(
        "*:*",
        sort="date_indexed_dt desc",
        rows=5
    )
    asc_dict = json.loads(result_asc)
    desc_dict = json.loads(result_desc)
    
    # Verify we got results
    assert "docs" in asc_dict
    assert "docs" in desc_dict
    assert len(asc_dict["docs"]) > 0
    assert len(desc_dict["docs"]) > 0
    
    # Verify dates are properly sorted
    asc_dates = [doc.get("date_indexed_dt") for doc in asc_dict["docs"] if "date_indexed_dt" in doc]
    desc_dates = [doc.get("date_indexed_dt") for doc in desc_dict["docs"] if "date_indexed_dt" in doc]
    
    # If we have dates to compare, verify they're in correct order
    if asc_dates and desc_dates:
        # Verify ascending order
        assert all(asc_dates[i] <= asc_dates[i+1] for i in range(len(asc_dates)-1)), "Ascending sort not in correct order"
        # Verify descending order
        assert all(desc_dates[i] >= desc_dates[i+1] for i in range(len(desc_dates)-1)), "Descending sort not in correct order"
        # Only verify different orders if dates are not all the same
        if len(set(asc_dates)) > 1:
            assert asc_dates != desc_dates, "Different sort orders should return documents in different orders"
    
    # Test sorting by id (which should be unique)
    result_id_asc = await solr_client.search(
        "*:*",
        sort="id asc",
        rows=5
    )
    result_id_desc = await solr_client.search(
        "*:*",
        sort="id desc",
        rows=5
    )
    asc_dict_id = json.loads(result_id_asc)
    desc_dict_id = json.loads(result_id_desc)
    
    # Verify we got results
    assert "docs" in asc_dict_id
    assert "docs" in desc_dict_id
    assert len(asc_dict_id["docs"]) > 0
    assert len(desc_dict_id["docs"]) > 0
    
    # Extract IDs for comparison
    asc_ids = [doc.get("id") for doc in asc_dict_id["docs"] if "id" in doc]
    desc_ids = [doc.get("id") for doc in desc_dict_id["docs"] if "id" in doc]
    
    # IDs should be unique and thus in different orders
    if asc_ids and desc_ids and len(asc_ids) > 1:
        assert asc_ids != desc_ids, "Different sort orders should return documents in different orders for IDs"
        # Verify ascending order
        assert all(asc_ids[i] <= asc_ids[i+1] for i in range(len(asc_ids)-1)), "Ascending ID sort not in correct order"
        # Verify descending order
        assert all(desc_ids[i] >= desc_ids[i+1] for i in range(len(desc_ids)-1)), "Descending ID sort not in correct order"

@pytest.mark.asyncio
async def test_live_error_handling(solr_client):
    """Test error handling with live Solr server."""
    # Test invalid sort field
    with pytest.raises(ValueError, match="Field 'nonexistent' is not sortable"):
        await solr_client.search("*:*", sort="nonexistent asc")

    # Test invalid sort direction
    with pytest.raises(ValueError, match="Invalid sort direction 'invalid' for field 'score'"):
        await solr_client.search("*:*", sort="score invalid")

    # Test malformed filter syntax
    with pytest.raises(SolrError) as exc_info:
        await solr_client.search("*:*", filters="{malformed json")
    assert "Solr syntax error" in str(exc_info.value)

    # Test invalid collection name
    with pytest.raises(SolrError) as exc_info:
        await solr_client.search("*:*", collection="nonexistent_collection")
    assert "Collection not found" in str(exc_info.value) 