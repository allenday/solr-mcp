"""Tests for solr_default_vectorizer tool."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from solr_mcp.tools.solr_default_vectorizer import get_default_text_vectorizer
from solr_mcp.vector_provider.constants import DEFAULT_OLLAMA_CONFIG, MODEL_DIMENSIONS

class TestDefaultVectorizerTool:
    """Test cases for default_text_vectorizer tool."""
    
    @pytest.mark.asyncio
    async def test_get_default_text_vectorizer_with_server(self):
        """Test getting default vectorizer with a server instance."""
        # Create mock server
        mock_vector_manager = MagicMock()
        mock_vector_manager.client.model = "nomic-embed-text"
        mock_vector_manager.client.base_url = "http://test-host:8888"
        
        mock_solr_client = MagicMock()
        mock_solr_client.vector_manager = mock_vector_manager
        
        mock_server = MagicMock()
        mock_server.solr_client = mock_solr_client
        
        # Execute tool
        result = await get_default_text_vectorizer(mock_server)
        
        # Verify result
        assert result["vector_provider_model"] == "nomic-embed-text"
        assert result["vector_provider_dimension"] == 768
        assert result["vector_provider_host"] == "test-host"
        assert result["vector_provider_port"] == 8888
        assert result["vector_provider_url"] == "http://test-host:8888"
        assert result["vector_provider_spec"] == "nomic-embed-text@test-host:8888"

    @pytest.mark.asyncio
    async def test_get_default_text_vectorizer_without_server(self):
        """Test getting default vectorizer without a server instance."""
        # Create a server without vector_manager
        mock_server = MagicMock(spec=['no_solr_client'])
        
        # Use patch to avoid trying to parse MagicMock as URL
        with patch('solr_mcp.vector_provider.constants.DEFAULT_OLLAMA_CONFIG', {
            'model': 'nomic-embed-text',
            'base_url': 'http://localhost:11434',
            'timeout': 30,
            'retries': 3
        }):
            # Execute tool
            result = await get_default_text_vectorizer(mock_server)
        
        # Verify result uses defaults
        assert result["vector_provider_model"] == DEFAULT_OLLAMA_CONFIG["model"]
        assert result["vector_provider_dimension"] == MODEL_DIMENSIONS[DEFAULT_OLLAMA_CONFIG["model"]]
        assert result["vector_provider_host"] == "localhost"
        assert result["vector_provider_port"] == 11434
        assert result["vector_provider_url"] == DEFAULT_OLLAMA_CONFIG["base_url"]
        assert result["vector_provider_spec"] == f"{DEFAULT_OLLAMA_CONFIG['model']}@localhost:11434"

    @pytest.mark.asyncio
    async def test_get_default_text_vectorizer_unknown_model(self):
        """Test getting default vectorizer with unknown model."""
        # Create mock server
        mock_vector_manager = MagicMock()
        mock_vector_manager.client.model = "unknown-model"
        mock_vector_manager.client.base_url = "http://test-host:8888"
        
        mock_solr_client = MagicMock()
        mock_solr_client.vector_manager = mock_vector_manager
        
        mock_server = MagicMock()
        mock_server.solr_client = mock_solr_client
        
        # Execute tool
        result = await get_default_text_vectorizer(mock_server)
        
        # Verify result with default dimension for unknown model
        assert result["vector_provider_model"] == "unknown-model"
        assert result["vector_provider_dimension"] == 768  # Default dimension
        assert result["vector_provider_spec"] == "unknown-model@test-host:8888"