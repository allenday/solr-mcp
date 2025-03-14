#!/usr/bin/env python3
"""
Simple indexing script to demonstrate adding documents to Solr without embeddings.
"""

import argparse
import json
import os
import sys
import time
import pysolr
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def index_documents(json_file: str, collection: str = "documents", commit: bool = True):
    """
    Index documents from a JSON file into Solr without vector embeddings.
    
    Args:
        json_file: Path to the JSON file containing documents
        collection: Solr collection name
        commit: Whether to commit after indexing
    """
    # Load documents
    with open(json_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Initialize Solr client directly
    solr_url = f"http://localhost:8983/solr/{collection}"
    solr = pysolr.Solr(solr_url, always_commit=commit)
    
    print(f"Indexing {len(documents)} documents to {collection} collection...")
    
    try:
        # Add documents to Solr
        solr.add(documents)
        print(f"Successfully indexed {len(documents)} documents in collection '{collection}'")
    except Exception as e:
        print(f"Error indexing documents: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents in Solr without vector embeddings")
    parser.add_argument("json_file", help="Path to the JSON file containing documents")
    parser.add_argument("--collection", "-c", default="documents", help="Solr collection name")
    parser.add_argument("--no-commit", dest="commit", action="store_false", help="Don't commit after indexing")
    
    args = parser.parse_args()
    index_documents(args.json_file, args.collection, args.commit)