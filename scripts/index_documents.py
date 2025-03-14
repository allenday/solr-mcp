#!/usr/bin/env python3
"""
Script to index documents in Solr with vector embeddings
generated using Ollama's nomic-embed-text model.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solr_mcp.embeddings.client import OllamaClient
from solr_mcp.solr.client import SolrClient


async def index_documents(json_file: str, collection: str = "vectors", commit: bool = True):
    """
    Index documents from a JSON file into Solr with vector embeddings.
    
    Args:
        json_file: Path to the JSON file containing documents
        collection: Solr collection name
        commit: Whether to commit after indexing
    """
    # Load documents
    with open(json_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Initialize clients
    solr_client = SolrClient()
    
    # Check if collection exists
    collections = solr_client.list_collections()
    if collection not in collections:
        print(f"Warning: Collection '{collection}' not found in Solr. Available collections: {collections}")
        response = input("Do you want to continue with the default collection? (y/N): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
        collection = solr_client.config.default_collection
    
    # Index documents with embeddings
    print(f"Indexing {len(documents)} documents with embeddings...")
    
    try:
        success = await solr_client.batch_index_with_generated_embeddings(
            documents=documents,
            collection=collection,
            commit=commit
        )
        
        if success:
            print(f"Successfully indexed {len(documents)} documents in collection '{collection}'")
        else:
            print("Indexing failed")
            
    except Exception as e:
        print(f"Error indexing documents: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents in Solr with vector embeddings")
    parser.add_argument("json_file", help="Path to the JSON file containing documents")
    parser.add_argument("--collection", "-c", default="vectors", help="Solr collection name")
    parser.add_argument("--no-commit", dest="commit", action="store_false", help="Don't commit after indexing")
    
    args = parser.parse_args()
    
    asyncio.run(index_documents(args.json_file, args.collection, args.commit))