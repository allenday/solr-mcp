#!/usr/bin/env python3
"""
Diagnostic script to help debug search issues in Solr collections.
"""

import argparse
import asyncio
import httpx
import json
import os
import sys
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


async def get_collection_schema(collection: str) -> Dict[str, Any]:
    """Get schema details for a collection.
    
    Args:
        collection: Solr collection name
        
    Returns:
        Schema details
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8983/solr/{collection}/schema",
            params={"wt": "json"},
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting schema: {response.status_code} - {response.text}")
            return {}


async def get_collection_status(collection: str) -> Dict[str, Any]:
    """Get status details for a collection.
    
    Args:
        collection: Solr collection name
        
    Returns:
        Collection status
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8983/solr/admin/collections",
            params={"action": "STATUS", "name": collection, "wt": "json"},
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting collection status: {response.status_code} - {response.text}")
            return {}


async def get_document_count(collection: str) -> int:
    """Get document count for a collection.
    
    Args:
        collection: Solr collection name
        
    Returns:
        Document count
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8983/solr/{collection}/select",
            params={"q": "*:*", "rows": 0, "wt": "json"},
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json().get("response", {}).get("numFound", 0)
        else:
            print(f"Error getting document count: {response.status_code} - {response.text}")
            return 0


async def get_document_sample(collection: str, num_docs: int = 3) -> List[Dict[str, Any]]:
    """Get a sample of documents from the collection.
    
    Args:
        collection: Solr collection name
        num_docs: Number of documents to return
        
    Returns:
        List of documents
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8983/solr/{collection}/select",
            params={"q": "*:*", "rows": num_docs, "wt": "json"},
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json().get("response", {}).get("docs", [])
        else:
            print(f"Error getting document sample: {response.status_code} - {response.text}")
            return []


async def test_text_search(collection: str, field: str, search_term: str) -> Dict[str, Any]:
    """Test a text search on a specific field.
    
    Args:
        collection: Solr collection name
        field: Field to search in
        search_term: Term to search for
        
    Returns:
        Search results
    """
    query = f"{field}:{search_term}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8983/solr/{collection}/select",
            params={"q": query, "rows": 5, "wt": "json"},
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error testing text search: {response.status_code} - {response.text}")
            return {}


async def analyze_text(collection: str, field_type: str, text: str) -> Dict[str, Any]:
    """Analyze how a text is processed for a given field type.
    
    Args:
        collection: Solr collection name
        field_type: Field type to analyze with
        text: Text to analyze
        
    Returns:
        Analysis results
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8983/solr/{collection}/analysis/field",
            params={"analysis.fieldtype": field_type, "analysis.fieldvalue": text, "wt": "json"},
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error analyzing text: {response.status_code} - {response.text}")
            return {}


async def diagnose_collection(collection: str, search_term: str = "bitcoin") -> None:
    """Run a comprehensive diagnosis on a collection.
    
    Args:
        collection: Solr collection name
        search_term: Term to use in search tests
    """
    print(f"\n=== Diagnosing Collection: {collection} ===\n")
    
    # Check if collection exists
    status = await get_collection_status(collection)
    if not status or "status" not in status:
        print(f"Error: Collection '{collection}' may not exist.")
        return
    
    # Get document count
    doc_count = await get_document_count(collection)
    print(f"Document count: {doc_count}")
    
    if doc_count == 0:
        print("No documents found in the collection. Please index some documents first.")
        return
    
    # Get schema details
    schema = await get_collection_schema(collection)
    if schema:
        field_types = {ft.get("name"): ft for ft in schema.get("schema", {}).get("fieldTypes", [])}
        fields = {f.get("name"): f for f in schema.get("schema", {}).get("fields", [])}
        
        print("\nText fields in schema:")
        text_fields = []
        for name, field in fields.items():
            field_type = field.get("type")
            if field_type and ("text" in field_type.lower() or field_type == "string"):
                indexed = field.get("indexed", True)
                stored = field.get("stored", True)
                text_fields.append(name)
                print(f"  - {name} (type: {field_type}, indexed: {indexed}, stored: {stored})")
        
        # Get document sample
        print("\nSample documents:")
        docs = await get_document_sample(collection)
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:")
            for key, value in doc.items():
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                elif isinstance(value, list) and len(str(value)) > 100:
                    value = str(value)[:100] + "..."
                print(f"  {key}: {value}")
        
        # Test search on each text field
        print("\nSearch tests:")
        for field in text_fields:
            print(f"\nTesting search on field: {field}")
            results = await test_text_search(collection, field, search_term)
            num_found = results.get("response", {}).get("numFound", 0)
            print(f"  Query: {field}:{search_term}")
            print(f"  Results found: {num_found}")
            
            if num_found > 0:
                print("  First match:")
                doc = results.get("response", {}).get("docs", [{}])[0]
                for key, value in doc.items():
                    if key == field or key in ["id", "title", "score"]:
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"    {key}: {value}")
        
        # Test general search
        print("\nTesting general search:")
        results = await test_text_search(collection, "*", search_term)
        num_found = results.get("response", {}).get("numFound", 0)
        print(f"  Query: {search_term}")
        print(f"  Results found: {num_found}")
        
        if num_found > 0:
            print("  First match:")
            doc = results.get("response", {}).get("docs", [{}])[0]
            for key, value in doc.items():
                if key in ["id", "title", "score", "content"]:
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"    {key}: {value}")
        
        # Analyze text processing
        print("\nText analysis for search term:")
        # Find a text field type to analyze with
        text_field_type = None
        for name, field in fields.items():
            if "text" in field.get("type", "").lower():
                text_field_type = field.get("type")
                break
        
        if text_field_type and text_field_type in field_types:
            print(f"  Using field type: {text_field_type}")
            analysis = await analyze_text(collection, text_field_type, search_term)
            
            if "analysis" in analysis:
                for key, stages in analysis.get("analysis", {}).items():
                    print(f"\n  {key.capitalize()} analysis:")
                    for stage in stages:
                        if "text" in stage:
                            print(f"    - {stage.get('name', 'unknown')}: {stage.get('text', [])}")
    
    print("\n=== Diagnosis Complete ===")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Diagnose Solr search issues")
    parser.add_argument("--collection", "-c", default="unified", help="Collection name")
    parser.add_argument("--term", "-t", default="bitcoin", help="Search term to test with")
    
    args = parser.parse_args()
    
    await diagnose_collection(args.collection, args.term)


if __name__ == "__main__":
    asyncio.run(main())