#!/usr/bin/env python3
"""
Demo script showing how to use the MCP client to search for information.
"""

import argparse
import asyncio
import os
import sys
import json
from typing import Dict, List, Optional, Any

from mcp import client
from mcp.transport.stdio import StdioClientTransport
from loguru import logger

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solr_mcp.embeddings.client import OllamaClient


async def search_by_text(query: str, collection: Optional[str] = None, rows: int = 5):
    """
    Perform a text search using the MCP client.
    
    Args:
        query: Search query
        collection: Collection name (optional)
        rows: Number of results to return
    """
    # Set up MCP client
    mcp_command = ["python", "-m", "solr_mcp.server"]
    transport = StdioClientTransport({"command": mcp_command})
    
    try:
        c = client.Client()
        await c.connect(transport)
        
        # Call the solr_search tool
        args = {
            "query": query,
            "rows": rows
        }
        
        if collection:
            args["collection"] = collection
        
        logger.info(f"Searching for: {query}")
        result = await c.request(
            {"name": "solr_search", "arguments": args}
        )
        
        # Display results
        print(f"\n=== Results for text search: '{query}' ===\n")
        display_results(result)
        
    finally:
        await c.close()


async def search_by_vector(query: str, collection: Optional[str] = None, k: int = 5):
    """
    Perform a vector similarity search using the MCP client.
    
    Args:
        query: Text to generate embedding from
        collection: Collection name (optional)
        k: Number of nearest neighbors to return
    """
    # First, generate an embedding for the query
    ollama_client = OllamaClient()
    embedding = await ollama_client.get_embedding(query)
    
    # Set up MCP client
    mcp_command = ["python", "-m", "solr_mcp.server"]
    transport = StdioClientTransport({"command": mcp_command})
    
    try:
        c = client.Client()
        await c.connect(transport)
        
        # Call the solr_vector_search tool
        args = {
            "vector": embedding,
            "k": k
        }
        
        if collection:
            args["collection"] = collection
        
        logger.info(f"Vector searching for: {query}")
        result = await c.request(
            {"name": "solr_vector_search", "arguments": args}
        )
        
        # Display results
        print(f"\n=== Results for vector search: '{query}' ===\n")
        display_results(result)
        
    finally:
        await c.close()


def display_results(result: Dict):
    """
    Display search results in a readable format.
    
    Args:
        result: Response from the MCP server
    """
    if isinstance(result, dict) and "content" in result:
        content = result["content"]
        
        if isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "")
            
            # Try to parse the JSON content
            try:
                data = json.loads(text_content)
                
                if "docs" in data and isinstance(data["docs"], list):
                    docs = data["docs"]
                    
                    if not docs:
                        print("No results found.")
                        return
                    
                    for i, doc in enumerate(docs, 1):
                        print(f"Result {i}:")
                        print(f"  Title: {doc.get('title', 'No title')}")
                        print(f"  ID: {doc.get('id', 'No ID')}")
                        
                        if "score" in doc:
                            print(f"  Score: {doc['score']}")
                            
                        # Show a preview of the text (first 150 chars)
                        text = doc.get("text", "")
                        if text:
                            preview = text[:150] + "..." if len(text) > 150 else text
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
                    print(text_content)
            except json.JSONDecodeError:
                print(text_content)
    else:
        print(result)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Demo search using the MCP client")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--vector", "-v", action="store_true", help="Use vector search instead of text search")
    parser.add_argument("--collection", "-c", help="Collection name")
    parser.add_argument("--results", "-n", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.vector:
        await search_by_vector(args.query, args.collection, args.results)
    else:
        await search_by_text(args.query, args.collection, args.results)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    asyncio.run(main())