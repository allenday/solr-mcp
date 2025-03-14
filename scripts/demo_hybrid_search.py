#!/usr/bin/env python3
"""
Demo script for testing hybrid search functionality with the MCP server.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional, List

from fastmcp import client
from fastmcp.transport.stdio import StdioClientTransport
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def display_results(results_json: str) -> None:
    """
    Display search results in a readable format.
    
    Args:
        results_json: JSON string with search results
    """
    try:
        results = json.loads(results_json)
        
        # Extract docs and metadata
        docs = results.get("docs", [])
        num_found = results.get("numFound", 0)
        
        if not docs:
            print("No matching documents found.")
            return
        
        print(f"Found {num_found} matching document(s):\n")
        
        for i, doc in enumerate(docs, 1):
            print(f"Result {i}:")
            print(f"  ID: {doc.get('id', 'N/A')}")
            
            # Handle title which could be a string or list
            title = doc.get('title', 'N/A')
            if isinstance(title, list) and title:
                title = title[0]
            print(f"  Title: {title}")
            
            # Display scores
            if 'hybrid_score' in doc:
                print(f"  Hybrid Score: {doc.get('hybrid_score', 0):.4f}")
                print(f"  Keyword Score: {doc.get('keyword_score', 0):.4f}")
                print(f"  Vector Score: {doc.get('vector_score', 0):.4f}")
            elif 'score' in doc:
                print(f"  Score: {doc.get('score', 0):.4f}")
            
            # Handle content which could be string or list
            content = doc.get('content', '')
            if not content:
                content = doc.get('text', '')
            if isinstance(content, list) and content:
                content = content[0]
                
            if content:
                preview = content[:150] + "..." if len(content) > 150 else content
                print(f"  Preview: {preview}")
                
            print()
    except Exception as e:
        print(f"Error displaying results: {e}")
        print(f"Raw results: {results_json}")


async def hybrid_search(
    query: str, 
    collection: Optional[str] = None, 
    blend_factor: float = 0.5,
    rows: int = 5
) -> None:
    """
    Perform a hybrid search using the MCP client.
    
    Args:
        query: Search query
        collection: Collection name (optional)
        blend_factor: Blending factor (0=keyword only, 1=vector only)
        rows: Number of results to return
    """
    # Set up MCP client
    mcp_command = ["python", "-m", "solr_mcp.server"]
    transport = StdioClientTransport({"command": mcp_command})
    
    try:
        c = client.Client()
        await c.connect(transport)
        
        # Call the solr_hybrid_search tool
        args = {
            "query": query,
            "blend_factor": blend_factor,
            "rows": rows
        }
        
        if collection:
            args["collection"] = collection
        
        print(f"Hybrid searching for: '{query}' with blend_factor: {blend_factor}")
        print(f"(0.0 = keyword only, 1.0 = vector only)\n")
        
        result = await c.request(
            {"name": "solr_hybrid_search", "arguments": args}
        )
        
        # Display results
        display_results(result)
        
    finally:
        await c.close()


async def keyword_search(query: str, collection: Optional[str] = None, rows: int = 5) -> None:
    """
    Perform a keyword search using the MCP client.
    
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
        
        print(f"Keyword searching for: '{query}'\n")
        
        result = await c.request(
            {"name": "solr_search", "arguments": args}
        )
        
        # Display results
        display_results(result)
        
    finally:
        await c.close()


async def vector_search(query: str, collection: Optional[str] = None, rows: int = 5) -> None:
    """
    Perform a vector search using the MCP client.
    
    Args:
        query: Search query
        collection: Collection name (optional)
        rows: Number of results to return
    """
    # Set up MCP client
    mcp_command = ["python", "-m", "solr_mcp.server"]
    transport = StdioClientTransport({"command": mcp_command})
    
    # First, generate embedding for the query
    from solr_mcp.embeddings.client import OllamaClient
    ollama = OllamaClient()
    embedding = await ollama.get_embedding(query)
    
    try:
        c = client.Client()
        await c.connect(transport)
        
        # Call the solr_vector_search tool
        args = {
            "vector": embedding,
            "k": rows
        }
        
        if collection:
            args["collection"] = collection
        
        print(f"Vector searching for: '{query}'\n")
        
        result = await c.request(
            {"name": "solr_vector_search", "arguments": args}
        )
        
        # Display results
        display_results(result)
        
    finally:
        await c.close()


async def compare_search_methods(query: str, collection: Optional[str] = None, rows: int = 5) -> None:
    """
    Compare different search methods side by side.
    
    Args:
        query: Search query
        collection: Collection name (optional)
        rows: Number of results to return
    """
    print("\n=== Keyword Search ===")
    await keyword_search(query, collection, rows)
    
    print("\n=== Vector Search ===")
    await vector_search(query, collection, rows)
    
    print("\n=== Hybrid Search (50% blend) ===")
    await hybrid_search(query, collection, 0.5, rows)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Demo for hybrid search with MCP server")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--collection", "-c", default="unified", help="Collection name")
    parser.add_argument("--mode", "-m", choices=['keyword', 'vector', 'hybrid', 'compare'], 
                       default='hybrid', help="Search mode")
    parser.add_argument("--blend", "-b", type=float, default=0.5, 
                       help="Blend factor for hybrid search (0=keyword only, 1=vector only)")
    parser.add_argument("--rows", "-r", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.mode == 'keyword':
        await keyword_search(args.query, args.collection, args.rows)
    elif args.mode == 'vector':
        await vector_search(args.query, args.collection, args.rows)
    elif args.mode == 'hybrid':
        await hybrid_search(args.query, args.collection, args.blend, args.rows)
    elif args.mode == 'compare':
        await compare_search_methods(args.query, args.collection, args.rows)


if __name__ == "__main__":
    asyncio.run(main())