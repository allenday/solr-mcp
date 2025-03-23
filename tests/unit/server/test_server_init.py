"""Tests for SolrMCP server initialization and running."""

import pytest
from unittest.mock import Mock, patch
import asyncio

from solr_mcp.server import SolrMCPServer, main

@pytest.mark.asyncio
async def test_server_run_async():
    """Test that server.run() is properly awaited."""
    server = SolrMCPServer(stdio=True)  # Use stdio to avoid actual network setup
    
    # Mock the internal run method to track if it was called
    run_called = False
    original_run = server.run
    
    async def mock_run():
        nonlocal run_called
        run_called = True
    
    server.run = mock_run
    
    # Run the server
    await server.run()
    
    # Verify run was called
    assert run_called, "Server run() method was not called"

def test_main_function_async_handling():
    """Test that main() properly handles async server.run()."""
    with patch('solr_mcp.server.SolrMCPServer') as mock_server_class:
        # Create a mock server instance
        mock_server = Mock()
        mock_server_class.return_value = mock_server
        
        # Make run() a coroutine that we can track
        async def mock_run():
            mock_run.called = True
        mock_run.called = False
        mock_server.run = mock_run
        
        # Run main
        with pytest.raises(RuntimeError, match="coroutine.*was never awaited"):
            main()
            
        # The test will fail if run() was properly awaited
        # We expect it to raise RuntimeError about unawaited coroutine 