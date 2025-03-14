#!/usr/bin/env python3
"""
Specialized script for indexing documents with vector embeddings into Solr.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Any
import time
import httpx

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solr_mcp.embeddings.client import OllamaClient


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using Ollama.
    
    Args:
        texts: List of text strings to generate embeddings for
        
    Returns:
        List of embedding vectors
    """
    client = OllamaClient()
    embeddings = []
    
    print(f"Generating embeddings for {len(texts)} documents...")
    
    # Process in smaller batches to avoid overwhelming Ollama
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
        batch_embeddings = await client.get_embeddings(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings


async def index_documents_with_vectors(json_file: str, collection: str = "vectors", commit: bool = True):
    """
    Index documents with vector embeddings into Solr.
    
    Args:
        json_file: Path to the JSON file containing documents
        collection: Solr collection name
        commit: Whether to commit after indexing
    """
    # Load documents
    with open(json_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Extract text for embedding generation
    texts = []
    for doc in documents:
        # Use the 'text' field if it exists, otherwise use 'content'
        if 'text' in doc:
            texts.append(doc['text'])
        elif 'content' in doc:
            texts.append(doc['content'])
        else:
            texts.append(doc.get('title', ''))  # Fallback to title if no text/content
    
    # Generate embeddings
    embeddings = await generate_embeddings(texts)
    
    # Add embeddings to documents
    docs_with_vectors = []
    for i, doc in enumerate(documents):
        doc_copy = doc.copy()
        # Format the vector as a string in Solr's expected format
        vector_str = f"{embeddings[i]}"
        # Clean up the string to match Solr's required format
        vector_str = vector_str.replace("[", "").replace("]", "").replace(" ", "")
        doc_copy['embedding'] = vector_str
        
        # Add metadata about the embedding
        doc_copy['vector_model'] = 'nomic-embed-text'
        doc_copy['dimensions'] = len(embeddings[i])
        doc_copy['vector_type'] = 'dense'
        
        # Handle date fields for Solr compatibility
        if 'date' in doc_copy and isinstance(doc_copy['date'], str):
            if len(doc_copy['date']) == 10 and doc_copy['date'].count('-') == 2:
                doc_copy['date'] += 'T00:00:00Z'
            elif not doc_copy['date'].endswith('Z'):
                doc_copy['date'] += 'Z'
        
        if 'date_indexed' in doc_copy and isinstance(doc_copy['date_indexed'], str):
            if '.' in doc_copy['date_indexed']:  # Has microseconds
                parts = doc_copy['date_indexed'].split('.')
                doc_copy['date_indexed'] = parts[0] + 'Z'
            elif not doc_copy['date_indexed'].endswith('Z'):
                doc_copy['date_indexed'] += 'Z'
        else:
            # Add current time as date_indexed if not present
            doc_copy['date_indexed'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        
        docs_with_vectors.append(doc_copy)
    
    # Export the prepared documents to a temporary file
    output_file = f"{os.path.splitext(json_file)[0]}_with_vectors.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(docs_with_vectors, f, indent=2)
    
    print(f"Prepared {len(docs_with_vectors)} documents with vector embeddings")
    print(f"Output saved to {output_file}")
    
    # Index to Solr
    solr_url = f"http://localhost:8983/solr/{collection}/update"
    headers = {"Content-Type": "application/json"}
    params = {"commit": "true"} if commit else {}
    
    print(f"Indexing to Solr collection '{collection}'...")
    
    try:
        # Use httpx directly for more control over the request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                solr_url,
                json=docs_with_vectors,
                headers=headers,
                params=params,
                timeout=60.0
            )
            
            if response.status_code == 200:
                print(f"Successfully indexed {len(docs_with_vectors)} documents with vectors")
                return True
            else:
                print(f"Error indexing documents: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        print(f"Error during indexing: {e}")
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Index documents with vector embeddings")
    parser.add_argument("json_file", help="Path to the JSON file containing documents")
    parser.add_argument("--collection", "-c", default="vectors", help="Solr collection name")
    parser.add_argument("--no-commit", dest="commit", action="store_false", help="Don't commit after indexing")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.json_file):
        print(f"Error: File {args.json_file} not found")
        sys.exit(1)
    
    result = await index_documents_with_vectors(args.json_file, args.collection, args.commit)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    asyncio.run(main())