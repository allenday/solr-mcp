#!/usr/bin/env python3
"""
Simple MCP client test script.
"""

import sys
import os
import json
import asyncio
import httpx

# Add the project root to your path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solr_mcp.solr.client import SolrClient

async def direct_solr_test():
    """Test direct Solr connection."""
    client = SolrClient()
    
    # Test standard search with different query formats
    print("\n=== Testing direct Solr client search with different query formats ===")
    results1 = await client.search("double spend", collection="unified")
    print(f"Simple search results: {results1}")
    
    results2 = await client.search("content:double content:spend", collection="unified")
    print(f"Field-specific search results: {results2}")
    
    results3 = await client.search("content:\"double spend\"~5", collection="unified")
    print(f"Phrase search results: {results3}")
    
    # Test with HTTP client
    print("\n=== Testing direct HTTP search ===")
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            'http://localhost:8983/solr/unified/select',
            params={
                'q': 'content:"double spend"~5',
                'wt': 'json'
            }
        )
        print(f"HTTP search results: {response.text}")
    
    # Check solr config details
    print("\n=== Solr client configuration ===")
    print(f"Default collection: {client.config.default_collection}")
    print(f"Collections available: {client.list_collections()}")

async def main():
    await direct_solr_test()

if __name__ == "__main__":
    asyncio.run(main())