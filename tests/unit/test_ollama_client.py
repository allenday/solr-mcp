"""Tests for the OllamaClient."""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json

import pytest
import httpx

from solr_mcp.embeddings.client import OllamaClient


class TestOllamaClient:
    """Test suite for OllamaClient."""
    
    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx.AsyncClient."""
        with patch("httpx.AsyncClient") as mock:
            client_instance = AsyncMock()
            mock.return_value.__aenter__.return_value = client_instance
            
            # Mock the response
            response = MagicMock()
            response.raise_for_status = MagicMock()
            response.json.return_value = {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            client_instance.post.return_value = response
            
            yield client_instance
    
    @pytest.mark.asyncio
    async def test_get_embedding(self, mock_httpx_client):
        """Test getting an embedding."""
        # Arrange
        client = OllamaClient(base_url="http://test-ollama:11434")
        text = "This is a test text"
        
        # Act
        embedding = await client.get_embedding(text)
        
        # Assert
        mock_httpx_client.post.assert_called_once_with(
            "http://test-ollama:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.asyncio
    async def test_get_embeddings(self, mock_httpx_client):
        """Test getting multiple embeddings."""
        # Arrange
        client = OllamaClient(base_url="http://test-ollama:11434")
        texts = ["Text 1", "Text 2"]
        
        # Act
        embeddings = await client.get_embeddings(texts)
        
        # Assert
        assert mock_httpx_client.post.call_count == 2
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert embeddings[1] == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_httpx_client):
        """Test error handling."""
        # Arrange
        client = OllamaClient(base_url="http://test-ollama:11434")
        mock_httpx_client.post.side_effect = httpx.RequestError("Connection error")
        
        # Act/Assert
        with pytest.raises(Exception):
            await client.get_embedding("Test text")