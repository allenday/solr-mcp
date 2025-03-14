#!/usr/bin/env python3
"""
Test script for vector search in Solr.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Any
import httpx

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solr_mcp.embeddings.client import OllamaClient


async def generate_query_embedding(query_text: str) -> List[float]:
    """Generate embedding for a query using Ollama.
    
    Args:
        query_text: Query text to generate embedding for
        
    Returns:
        Embedding vector for the query
    """
    client = OllamaClient()
    print(f"Generating embedding for query: '{query_text}'")
    embedding = await client.get_embedding(query_text)
    return embedding


async def vector_search(
    query: str, 
    collection: str = "testvectors",
    vector_field: str = "embedding",
    k: int = 5,
    filter_query: str = None
):
    """
    Perform a vector search in Solr using the generated embedding.
    
    Args:
        query: Search query text
        collection: Solr collection name
        vector_field: Name of the vector field
        k: Number of results to return
        filter_query: Optional filter query
    """
    # Generate embedding for the query
    query_embedding = await generate_query_embedding(query)
    
    # Format the vector as a string that Solr expects for KNN search
    vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    
    # Prepare Solr KNN query
    solr_url = f"http://localhost:8983/solr/{collection}/select"
    
    # Build query parameters
    params = {
        "q": f"{{!knn f={vector_field} topK={k}}}{vector_str}",
        "fl": "id,title,text,score,vector_model",
        "wt": "json"
    }
    
    if filter_query:
        params["fq"] = filter_query
    
    print(f"Executing vector search in collection '{collection}'")
    
    try:
        # Split implementation - try POST first (to handle long vectors), fall back to GET
        async with httpx.AsyncClient() as client:
            try:
                # First try with POST to handle large vectors
                response = await client.post(
                    solr_url,
                    data={"q": params["q"]},
                    params={
                        "fl": params["fl"],
                        "wt": params["wt"]
                    },
                    timeout=30.0
                )
            except Exception as post_error:
                print(f"POST request failed, trying GET: {post_error}")
                
                # Fall back to GET with a shorter vector representation
                # Truncate the vector string if needed
                if len(vector_str) > 800:
                    short_vector = ",".join(str(round(v, 4)) for v in query_embedding[:100])
                    params["q"] = f"{{!knn f={vector_field} topK={k}}}{short_vector}"
                
                response = await client.get(solr_url, params=params, timeout=30.0)
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"Error in vector search: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"Error during vector search: {e}")
        return None


def display_results(results: Dict[str, Any]):
    """Display search results in a readable format.
    
    Args:
        results: Search results from Solr
    """
    if not results or 'response' not in results:
        print("No valid results received")
        return
    
    print("\n=== Vector Search Results ===\n")
    
    docs = results['response']['docs']
    num_found = results['response']['numFound']
    
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
        
        if 'score' in doc:
            print(f"  Score: {doc['score']}")
            
        # Handle text which could be string or list
        text = doc.get('text', '')
        if isinstance(text, list) and text:
            text = text[0]
            
        if text:
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"  Preview: {preview}")
            
        # Print model info if available
        if 'vector_model' in doc:
            print(f"  Model: {doc.get('vector_model')}")
            
        print()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test vector search in Solr")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--collection", "-c", default="vectors", help="Solr collection name")
    parser.add_argument("--field", "-f", default="embedding", help="Vector field name")
    parser.add_argument("--results", "-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--filter", "-fq", help="Optional filter query")
    
    args = parser.parse_args()
    
    results = await vector_search(
        args.query, 
        args.collection, 
        args.field, 
        args.results,
        args.filter
    )
    
    if results:
        display_results(results)


if __name__ == "__main__":
    asyncio.run(main())