#!/usr/bin/env python3
"""
Unified search script for both keyword and vector searches in the same Solr collection.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional
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


async def keyword_search(
    query: str, 
    collection: str = "unified",
    fields: Optional[List[str]] = None,
    filter_query: Optional[str] = None,
    rows: int = 5
) -> Dict[str, Any]:
    """
    Perform a keyword search in the unified collection.
    
    Args:
        query: Search query text
        collection: Solr collection name
        fields: Fields to return
        filter_query: Optional filter query
        rows: Number of results to return
        
    Returns:
        Search results
    """
    if not fields:
        fields = ["id", "title", "content", "source", "score"]
    
    solr_url = f"http://localhost:8983/solr/{collection}/select"
    params = {
        "q": query,
        "fl": ",".join(fields),
        "rows": rows,
        "wt": "json"
    }
    
    if filter_query:
        params["fq"] = filter_query
    
    print(f"Executing keyword search for '{query}' in collection '{collection}'")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(solr_url, params=params, timeout=30.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error in keyword search: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"Error during keyword search: {e}")
        return None


async def vector_search(
    query: str, 
    collection: str = "unified",
    vector_field: str = "embedding",
    fields: Optional[List[str]] = None,
    filter_query: Optional[str] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Perform a vector search in the unified collection.
    
    Args:
        query: Search query text
        collection: Solr collection name
        vector_field: Name of the vector field
        fields: Fields to return
        filter_query: Optional filter query
        k: Number of results to return
        
    Returns:
        Search results
    """
    if not fields:
        fields = ["id", "title", "content", "source", "score", "vector_model_s"]
    
    # Generate embedding for the query
    query_embedding = await generate_query_embedding(query)
    
    # Format the vector as a string that Solr expects for KNN search
    vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    
    # Prepare Solr KNN query
    solr_url = f"http://localhost:8983/solr/{collection}/select"
    params = {
        "q": f"{{!knn f={vector_field} topK={k}}}{vector_str}",
        "fl": ",".join(fields),
        "wt": "json"
    }
    
    if filter_query:
        params["fq"] = filter_query
    
    print(f"Executing vector search for '{query}' in collection '{collection}'")
    
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
                response = await client.get(solr_url, params=params, timeout=30.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error in vector search: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"Error during vector search: {e}")
        return None


async def hybrid_search(
    query: str, 
    collection: str = "unified",
    vector_field: str = "embedding",
    fields: Optional[List[str]] = None,
    filter_query: Optional[str] = None,
    k: int = 5,
    blend_factor: float = 0.5  # 0=keyword only, 1=vector only, between 0-1 blends
) -> Dict[str, Any]:
    """
    Perform a hybrid search combining both keyword and vector search results.
    
    Args:
        query: Search query text
        collection: Solr collection name
        vector_field: Name of the vector field
        fields: Fields to return
        filter_query: Optional filter query
        k: Number of results to return
        blend_factor: Blending factor between keyword and vector results (0-1)
        
    Returns:
        Blended search results
    """
    if not fields:
        fields = ["id", "title", "content", "source", "score", "vector_model_s"]
    
    # Run both searches
    keyword_results = await keyword_search(query, collection, fields, filter_query, k)
    vector_results = await vector_search(query, collection, vector_field, fields, filter_query, k)
    
    if not keyword_results or not vector_results:
        return keyword_results or vector_results
    
    # Extract docs from both result sets
    keyword_docs = keyword_results.get('response', {}).get('docs', [])
    vector_docs = vector_results.get('response', {}).get('docs', [])
    
    # Create a hybrid result set
    hybrid_docs = {}
    max_keyword_score = max([doc.get('score', 0) for doc in keyword_docs]) if keyword_docs else 1
    max_vector_score = max([doc.get('score', 0) for doc in vector_docs]) if vector_docs else 1
    
    # Process keyword results
    for doc in keyword_docs:
        doc_id = doc['id']
        # Normalize score to 0-1 range
        normalized_score = doc.get('score', 0) / max_keyword_score if max_keyword_score > 0 else 0
        hybrid_docs[doc_id] = {
            **doc,
            'keyword_score': normalized_score,
            'vector_score': 0,
            'hybrid_score': normalized_score * (1 - blend_factor)
        }
    
    # Process vector results
    for doc in vector_docs:
        doc_id = doc['id']
        # Normalize score to 0-1 range
        normalized_score = doc.get('score', 0) / max_vector_score if max_vector_score > 0 else 0
        if doc_id in hybrid_docs:
            # Update existing doc with vector score
            hybrid_docs[doc_id]['vector_score'] = normalized_score
            hybrid_docs[doc_id]['hybrid_score'] += normalized_score * blend_factor
        else:
            hybrid_docs[doc_id] = {
                **doc,
                'keyword_score': 0,
                'vector_score': normalized_score,
                'hybrid_score': normalized_score * blend_factor
            }
    
    # Sort by hybrid score
    sorted_docs = sorted(hybrid_docs.values(), key=lambda x: x.get('hybrid_score', 0), reverse=True)
    
    # Create a hybrid result
    hybrid_result = {
        'responseHeader': keyword_results.get('responseHeader', {}),
        'response': {
            'numFound': len(sorted_docs),
            'start': 0,
            'maxScore': 1.0,
            'docs': sorted_docs[:k]
        }
    }
    
    return hybrid_result


def display_results(results: Dict[str, Any], search_type: str):
    """Display search results in a readable format.
    
    Args:
        results: Search results from Solr
        search_type: Type of search performed (keyword, vector, or hybrid)
    """
    if not results or 'response' not in results:
        print("No valid results received")
        return
    
    print(f"\n=== {search_type.title()} Search Results ===\n")
    
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
        
        # Display scores based on search type
        if search_type == 'hybrid':
            print(f"  Hybrid Score: {doc.get('hybrid_score', 0):.4f}")
            print(f"  Keyword Score: {doc.get('keyword_score', 0):.4f}")
            print(f"  Vector Score: {doc.get('vector_score', 0):.4f}")
        else:
            if 'score' in doc:
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
            
        # Print model info if available
        if 'vector_model_s' in doc:
            print(f"  Model: {doc.get('vector_model_s')}")
            
        print()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified search for Solr")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--collection", "-c", default="unified", help="Collection name")
    parser.add_argument("--mode", "-m", choices=['keyword', 'vector', 'hybrid'], default='hybrid',
                       help="Search mode: keyword, vector, or hybrid (default)")
    parser.add_argument("--blend", "-b", type=float, default=0.5, 
                       help="Blend factor for hybrid search (0=keyword only, 1=vector only)")
    parser.add_argument("--results", "-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--filter", "-fq", help="Optional filter query")
    
    args = parser.parse_args()
    
    if args.mode == 'keyword':
        results = await keyword_search(
            args.query, 
            args.collection, 
            None, 
            args.filter, 
            args.results
        )
        if results:
            display_results(results, 'keyword')
            
    elif args.mode == 'vector':
        results = await vector_search(
            args.query, 
            args.collection, 
            'embedding', 
            None, 
            args.filter, 
            args.results
        )
        if results:
            display_results(results, 'vector')
            
    elif args.mode == 'hybrid':
        results = await hybrid_search(
            args.query, 
            args.collection, 
            'embedding', 
            None, 
            args.filter, 
            args.results,
            args.blend
        )
        if results:
            display_results(results, 'hybrid')


if __name__ == "__main__":
    asyncio.run(main())