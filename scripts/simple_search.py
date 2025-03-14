#!/usr/bin/env python3
"""
Simple search script to demonstrate searching in Solr without MCP.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solr_mcp.solr.client import SolrClient
from solr_mcp.embeddings.client import OllamaClient


async def search_by_text(query: str, collection: Optional[str] = None, rows: int = 5):
    """
    Perform a text search using the SolrClient directly.
    
    Args:
        query: Search query
        collection: Collection name (optional)
        rows: Number of results to return
    """
    # Set up Solr client
    solr_client = SolrClient()
    
    try:
        # Perform the search
        print(f"Searching for: '{query}'")
        result = await solr_client.search(
            query=query,
            collection=collection,
            rows=rows
        )
        
        # Display results
        print(f"\n=== Results for text search: '{query}' ===\n")
        display_results(result)
        
    except Exception as e:
        print(f"Error during search: {e}")


async def search_by_vector(query: str, collection: Optional[str] = None, k: int = 5):
    """
    Perform a vector similarity search using the SolrClient directly.
    
    Args:
        query: Text to generate embedding from
        collection: Collection name (optional)
        k: Number of nearest neighbors to return
    """
    # Set up clients
    solr_client = SolrClient()
    ollama_client = OllamaClient()
    
    try:
        # Generate embedding for the query
        print(f"Generating embedding for: '{query}'")
        embedding = await ollama_client.get_embedding(query)
        
        # Perform the vector search
        print(f"Performing vector search")
        result = await solr_client.vector_search(
            vector=embedding,
            collection=collection,
            k=k
        )
        
        # Display results
        print(f"\n=== Results for vector search: '{query}' ===\n")
        display_results(result)
        
    except Exception as e:
        print(f"Error during vector search: {e}")


def display_results(result_json: str):
    """
    Display search results in a readable format.
    
    Args:
        result_json: JSON string with search results
    """
    try:
        data = json.loads(result_json)
        
        if "docs" in data and isinstance(data["docs"], list):
            docs = data["docs"]
            
            if not docs:
                print("No results found.")
                return
            
            for i, doc in enumerate(docs, 1):
                print(f"Result {i}:")
                # Handle title which could be a string or list
                title = doc.get('title', 'No title')
                if isinstance(title, list):
                    title = title[0]
                print(f"  Title: {title}")
                print(f"  ID: {doc.get('id', 'No ID')}")
                
                if "score" in doc:
                    print(f"  Score: {doc['score']}")
                    
                # Show a preview of the content (first 150 chars)
                content = doc.get("content", "")
                if content:
                    preview = content[:150] + "..." if len(content) > 150 else content
                    print(f"  Preview: {preview}")
                
                if "category" in doc:
                    categories = doc["category"] if isinstance(doc["category"], list) else [doc["category"]]
                    print(f"  Categories: {', '.join(categories)}")
                    
                if "tags" in doc:
                    tags = doc["tags"] if isinstance(doc["tags"], list) else [doc["tags"]]
                    print(f"  Tags: {', '.join(tags)}")
                    
                print()
                
            print(f"Total results: {data.get('numFound', len(docs))}")
        else:
            print("Unexpected result format:")
            print(result_json)
    except json.JSONDecodeError:
        print("Could not parse JSON response:")
        print(result_json)
    except Exception as e:
        print(f"Error displaying results: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple search script for Solr")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--vector", "-v", action="store_true", help="Use vector search instead of text search")
    parser.add_argument("--collection", "-c", default="documents", help="Collection name")
    parser.add_argument("--results", "-n", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.vector:
        await search_by_vector(args.query, args.collection, args.results)
    else:
        await search_by_text(args.query, args.collection, args.results)


if __name__ == "__main__":
    asyncio.run(main())