#!/usr/bin/env python
"""Test script for Ollama embeddings functionality."""

import asyncio
from solr_mcp.embeddings.client import OllamaClient

async def test_embeddings():
    """Test the OllamaClient with sample text."""
    client = OllamaClient(base_url="http://localhost:11434", timeout=60.0)
    print(f"Initialized client with endpoint: {client.embeddings_endpoint}")
    
    test_text = "Bitcoin: A Peer-to-Peer Electronic Cash System"
    
    try:
        print(f"Getting embedding for: '{test_text}'")
        embedding = await client.get_embedding(test_text)
        
        # Print embedding stats rather than full vector
        print(f"Successfully generated embedding with dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Last 5 values: {embedding[-5:]}")
        
        # Test batch embedding
        batch_texts = [
            "Bitcoin: A Peer-to-Peer Electronic Cash System",
            "The Bitcoin network uses proof-of-work to timestamp transactions",
            "A purely peer-to-peer version of electronic cash would allow online payments"
        ]
        
        print(f"\nGetting batch embeddings for {len(batch_texts)} texts")
        batch_embeddings = await client.get_embeddings(batch_texts)
        
        print(f"Successfully generated {len(batch_embeddings)} embeddings")
        for i, emb in enumerate(batch_embeddings):
            print(f"Embedding {i+1}: {len(emb)} dimensions")
        
        return True
    except Exception as e:
        print(f"Error testing embeddings: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_embeddings())
    exit(0 if result else 1)