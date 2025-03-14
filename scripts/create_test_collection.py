#!/usr/bin/env python3
"""
Script to create a test collection with optimized schema for vector search.
"""

import asyncio
import httpx
import json
import sys
import os
import time


async def create_collection(collection_name="testvectors"):
    """Create a test collection for vector search."""
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
            
            # Create the collection with 1 shard and 1 replica
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
            
            # Define schema fields
            schema_fields = [
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
                    "name": "text",
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
                    "name": "vector_model",
                    "type": "string",
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
                    continue
            
            # Define vector field type
            vector_fieldtype = {
                "name": "knn_vector",
                "class": "solr.DenseVectorField",
                "vectorDimension": 768,  # Adjusted to match actual dimensions from Ollama's nomic-embed-text
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
            
            # Define vector field
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
        print(f"Error creating collection: {e}")
        return False


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
    else:
        collection_name = "testvectors"
    
    success = await create_collection(collection_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())