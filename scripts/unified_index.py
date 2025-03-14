#!/usr/bin/env python3
"""
Script for indexing documents with both text and vector embeddings in a unified Solr collection.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import httpx
from typing import Dict, List, Any

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


def prepare_field_names(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare field names for Solr using dynamic field naming convention.
    
    Args:
        doc: Original document
        
    Returns:
        Document with properly named fields for Solr
    """
    solr_doc = {}
    
    # Map basic fields (keep as is)
    for field in ['id', 'title', 'content', 'source', 'embedding']:
        if field in doc:
            solr_doc[field] = doc[field]
    
    # Special handling for content if it doesn't exist but text does
    if 'content' not in solr_doc and 'text' in doc:
        solr_doc['content'] = doc['text']
    
    # Map integer fields
    for field in ['section_number', 'dimensions']:
        if field in doc:
            solr_doc[f"{field}_i"] = doc[field]
    
    # Map string fields
    for field in ['author', 'vector_model']:
        if field in doc:
            solr_doc[f"{field}_s"] = doc[field]
    
    # Map date fields
    for field in ['date', 'date_indexed']:
        if field in doc:
            # Format date for Solr
            date_value = doc[field]
            if isinstance(date_value, str):
                if '.' in date_value:  # Has microseconds
                    parts = date_value.split('.')
                    date_value = parts[0] + 'Z'
                elif not date_value.endswith('Z'):
                    date_value = date_value + 'Z'
            solr_doc[f"{field}_dt"] = date_value
    
    # Map multi-valued fields
    for field in ['category', 'tags']:
        if field in doc:
            solr_doc[f"{field}_ss"] = doc[field]
    
    return solr_doc


async def index_documents(json_file: str, collection: str = "unified", commit: bool = True):
    """
    Index documents with both text content and vector embeddings.
    
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
            texts.append(doc.get('title', ''))
    
    # Generate embeddings
    embeddings = await generate_embeddings(texts)
    
    # Prepare documents for indexing
    solr_docs = []
    for i, doc in enumerate(documents):
        doc_copy = doc.copy()
        
        # Add embedding and metadata
        doc_copy['embedding'] = embeddings[i]
        doc_copy['vector_model'] = 'nomic-embed-text'
        doc_copy['dimensions'] = len(embeddings[i])
        
        # Add current time as date_indexed if not present
        if 'date_indexed' not in doc_copy:
            doc_copy['date_indexed'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        
        # Prepare field names according to Solr conventions
        solr_doc = prepare_field_names(doc_copy)
        solr_docs.append(solr_doc)
    
    # Index documents
    print(f"Indexing {len(solr_docs)} documents to collection '{collection}'...")
    
    async with httpx.AsyncClient() as client:
        for i, doc in enumerate(solr_docs):
            solr_url = f"http://localhost:8983/solr/{collection}/update/json/docs"
            params = {"commit": "true"} if (commit and i == len(solr_docs) - 1) else {}
            
            try:
                response = await client.post(
                    solr_url,
                    json=doc,
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    print(f"Error indexing document {doc['id']}: {response.status_code} - {response.text}")
                    return False
                    
                print(f"Indexed document {i+1}/{len(solr_docs)}: {doc['id']}")
                
            except Exception as e:
                print(f"Error indexing document {doc['id']}: {e}")
                return False
    
    print(f"Successfully indexed {len(solr_docs)} documents to collection '{collection}'")
    return True


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Index documents with both text and vector embeddings")
    parser.add_argument("json_file", help="Path to the JSON file containing documents")
    parser.add_argument("--collection", "-c", default="unified", help="Solr collection name")
    parser.add_argument("--no-commit", dest="commit", action="store_false", help="Don't commit after indexing")
    
    args = parser.parse_args()
    
    result = await index_documents(args.json_file, args.collection, args.commit)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    asyncio.run(main())