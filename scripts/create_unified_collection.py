#!/usr/bin/env python3
"""
Script to create a unified collection for both document content and vector embeddings.
"""

import asyncio
import httpx
import json
import sys
import os
import time


async def create_unified_collection(collection_name="unified"):
    """Create a unified collection for both text and vector search."""
    try:
        async with httpx.AsyncClient() as client:
            # Check if collection already exists
            response = await client.get(
                "http://localhost:8983/solr/admin/collections",
                params={"action": "LIST", "wt": "json"},
                timeout=10.0
            )
            
            if response.status_code != 200:
                print(f"Error checking collections: {response.status_code}")
                return False
            
            collections = response.json().get('collections', [])
            
            if collection_name in collections:
                print(f"Collection '{collection_name}' already exists. Deleting it...")
                delete_response = await client.get(
                    "http://localhost:8983/solr/admin/collections",
                    params={
                        "action": "DELETE",
                        "name": collection_name,
                        "wt": "json"
                    },
                    timeout=10.0
                )
                
                if delete_response.status_code != 200:
                    print(f"Error deleting collection: {delete_response.status_code} - {delete_response.text}")
                    return False
                
                print(f"Deleted collection '{collection_name}'")
                # Wait a moment for the deletion to complete
                await asyncio.sleep(3)
            
            # Create the collection with 1 shard and 1 replica for simplicity
            create_response = await client.get(
                "http://localhost:8983/solr/admin/collections",
                params={
                    "action": "CREATE",
                    "name": collection_name,
                    "numShards": 1,
                    "replicationFactor": 1,
                    "wt": "json"
                },
                timeout=30.0
            )
            
            if create_response.status_code != 200:
                print(f"Error creating collection: {create_response.status_code} - {create_response.text}")
                return False
            
            print(f"Created collection '{collection_name}'")
            
            # Wait a moment for the collection to be ready
            await asyncio.sleep(2)
            
            # Define schema fields - both document and vector fields in one schema
            schema_fields = [
                # Document fields
                {
                    "name": "id",
                    "type": "string",
                    "stored": True,
                    "indexed": True,
                    "required": True
                },
                {
                    "name": "title",
                    "type": "text_general",
                    "stored": True,
                    "indexed": True
                },
                {
                    "name": "content",
                    "type": "text_general",
                    "stored": True,
                    "indexed": True
                },
                {
                    "name": "source",
                    "type": "string",
                    "stored": True,
                    "indexed": True
                },
                {
                    "name": "section_number_i",  # Using dynamic field naming
                    "type": "pint",
                    "stored": True,
                    "indexed": True
                },
                {
                    "name": "author_s",  # Using dynamic field naming
                    "type": "string",
                    "stored": True,
                    "indexed": True
                },
                {
                    "name": "date_indexed_dt",  # Using dynamic field naming
                    "type": "pdate",
                    "stored": True,
                    "indexed": True
                },
                {
                    "name": "category_ss",  # Using dynamic field naming for multi-valued
                    "type": "string",
                    "stored": True,
                    "indexed": True,
                    "multiValued": True
                },
                {
                    "name": "tags_ss",  # Using dynamic field naming for multi-valued
                    "type": "string",
                    "stored": True,
                    "indexed": True,
                    "multiValued": True
                },
                # Vector metadata fields
                {
                    "name": "vector_model_s",
                    "type": "string",
                    "stored": True,
                    "indexed": True
                },
                {
                    "name": "dimensions_i",
                    "type": "pint",
                    "stored": True,
                    "indexed": True
                }
            ]
            
            # Add each field to the schema
            for field in schema_fields:
                field_response = await client.post(
                    f"http://localhost:8983/solr/{collection_name}/schema",
                    json={"add-field": field},
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                
                if field_response.status_code != 200:
                    print(f"Error adding field {field['name']}: {field_response.status_code} - {field_response.text}")
                    # Continue with other fields even if one fails (might be an existing field)
                    continue
                
                print(f"Added field {field['name']}")
            
            # Define vector field type for 768D vectors (nomic-embed-text)
            vector_fieldtype = {
                "name": "knn_vector",
                "class": "solr.DenseVectorField",
                "vectorDimension": 768,
                "similarityFunction": "cosine"
            }
            
            # Add vector field type
            fieldtype_response = await client.post(
                f"http://localhost:8983/solr/{collection_name}/schema",
                json={"add-field-type": vector_fieldtype},
                headers={"Content-Type": "application/json"},
                timeout=10.0
            )
            
            if fieldtype_response.status_code != 200:
                print(f"Error adding field type: {fieldtype_response.status_code} - {fieldtype_response.text}")
                return False
            
            print(f"Added field type {vector_fieldtype['name']}")
            
            # Define the main vector embedding field
            vector_field = {
                "name": "embedding",
                "type": "knn_vector",
                "stored": True,
                "indexed": True
            }
            
            # Add vector field
            vector_field_response = await client.post(
                f"http://localhost:8983/solr/{collection_name}/schema",
                json={"add-field": vector_field},
                headers={"Content-Type": "application/json"},
                timeout=10.0
            )
            
            if vector_field_response.status_code != 200:
                print(f"Error adding vector field: {vector_field_response.status_code} - {vector_field_response.text}")
                return False
            
            print(f"Added field {vector_field['name']}")
            
            print(f"Collection '{collection_name}' created and configured successfully")
            return True
    
    except Exception as e:
        print(f"Error creating unified collection: {e}")
        return False


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
    else:
        collection_name = "unified"
    
    success = await create_unified_collection(collection_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())