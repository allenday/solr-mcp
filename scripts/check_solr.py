#!/usr/bin/env python3
"""
Script to check Solr configuration and status.
"""

import asyncio
import httpx
import json
import sys


async def check_solr_collections():
    """Check Solr collections and their configuration."""
    try:
        async with httpx.AsyncClient() as client:
            # Get list of collections
            response = await client.get(
                "http://localhost:8983/solr/admin/collections",
                params={"action": "LIST", "wt": "json"},
                timeout=10.0
            )
            
            if response.status_code != 200:
                print(f"Error getting collections: {response.status_code} - {response.text}")
                return
            
            collections_data = response.json()
            
            if 'collections' in collections_data:
                collections = collections_data['collections']
                print(f"Found {len(collections)} collections: {', '.join(collections)}")
                
                # Check each collection
                for collection in collections:
                    # Get schema information
                    schema_response = await client.get(
                        f"http://localhost:8983/solr/{collection}/schema",
                        params={"wt": "json"},
                        timeout=10.0
                    )
                    
                    if schema_response.status_code != 200:
                        print(f"Error getting schema for {collection}: {schema_response.status_code}")
                        continue
                    
                    schema_data = schema_response.json()
                    
                    # Check for vector field type
                    field_types = schema_data.get('schema', {}).get('fieldTypes', [])
                    vector_type = None
                    for ft in field_types:
                        if ft.get('class') == 'solr.DenseVectorField':
                            vector_type = ft
                            break
                    
                    if vector_type:
                        print(f"\nCollection '{collection}' has vector field type:")
                        print(f"  Name: {vector_type.get('name')}")
                        print(f"  Class: {vector_type.get('class')}")
                        print(f"  Vector Dimension: {vector_type.get('vectorDimension')}")
                        print(f"  Similarity Function: {vector_type.get('similarityFunction')}")
                    else:
                        print(f"\nCollection '{collection}' does not have a vector field type")
                        
                    # Check for vector fields
                    fields = schema_data.get('schema', {}).get('fields', [])
                    vector_fields = [f for f in fields if f.get('type') == 'knn_vector']
                    
                    if vector_fields:
                        print(f"\n  Vector fields in '{collection}':")
                        for field in vector_fields:
                            print(f"    - {field.get('name')} (indexed: {field.get('indexed')}, stored: {field.get('stored')})")
                    else:
                        print(f"\n  No vector fields found in '{collection}'")
            else:
                print("No collections found or invalid response format")
    
    except Exception as e:
        print(f"Error checking Solr: {e}")


if __name__ == "__main__":
    asyncio.run(check_solr_collections())