"""Main MCP server implementation for SolrCloud integration."""

import argparse
import os
import sys
from typing import List
import functools
import asyncio

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import FastMCPError
from mcp.server.models import InitializationOptions
from mcp.types import ServerCapabilities
from mcp.server.lowlevel.server import NotificationOptions
from loguru import logger

from solr_mcp.solr.client import SolrClient
from solr_mcp.solr.config import SolrConfig
from solr_mcp.tools import TOOLS_DEFINITION
from solr_mcp import __version__

class SolrMCPServer:
    """Model Context Protocol server for SolrCloud integration."""

    def __init__(
        self,
        mcp_port: int = int(os.getenv("MCP_PORT", 8081)),
        solr_base_url: str = os.getenv("SOLR_BASE_URL", "http://localhost:8983/solr"),
        zookeeper_hosts: List[str] = os.getenv("ZOOKEEPER_HOSTS", "localhost:2181").split(","),
        default_collection: str = os.getenv("DEFAULT_COLLECTION", "default"),
        connection_timeout: int = int(os.getenv("CONNECTION_TIMEOUT", 10)),
        embedding_field: str = os.getenv("EMBEDDING_FIELD", "embedding"),
        default_top_k: int = int(os.getenv("DEFAULT_TOP_K", 10)),
        stdio: bool = False,
    ):
        """Initialize the server.
        
        Args:
            mcp_port: Port for MCP server
            solr_base_url: Base URL for Solr
            zookeeper_hosts: List of ZooKeeper hosts
            default_collection: Default Solr collection
            connection_timeout: Connection timeout in seconds
            embedding_field: Field name for embeddings
            default_top_k: Default number of results to return
            stdio: Use stdio instead of HTTP
        """
        self.port = mcp_port
        self.config = SolrConfig(
            solr_base_url=solr_base_url,
            zookeeper_hosts=zookeeper_hosts,
            default_collection=default_collection,
            connection_timeout=connection_timeout,
            embedding_field=embedding_field,
            default_top_k=default_top_k
        )
        self.stdio = stdio
        self.__setup_server()

    def __setup_server(self):
        """Set up the MCP server and Solr client."""
        try:
            self.__connect_to_solr()
        except Exception as e:
            logger.error(f"Solr connection error: {e}")
            sys.exit(1)

        logger.info(f"Server starting on port {self.port}")
        
        # Create FastMCP instance with initialization options
        self.mcp = FastMCP(
            name="Solr MCP Server",
            instructions="""This server provides tools for interacting with SolrCloud:
- List collections
- Execute SQL queries
- Execute semantic search queries
- Execute vector search queries""",
            debug=True,
            port=self.port
        )
        
        # Register tools
        self.__setup_tools()

        # Wait for initialization
        logger.info("Waiting for server initialization...")

    def __connect_to_solr(self):
        """Initialize Solr client connection."""
        self.solr_client = SolrClient(config=self.config)

    def __transform_tool_params(self, tool_name: str, params: dict) -> dict:
        """Transform tool parameters before they are passed to the tool.
        
        This method handles parameter translation, such as converting server name strings
        to actual server instances.
        
        Args:
            tool_name: Name of the tool being called
            params: Parameters passed to the tool
            
        Returns:
            Transformed parameters
        """
        if "mcp" in params:
            if isinstance(params["mcp"], str):
                # If mcp is passed as a string (server name), use self as the server instance
                params["mcp"] = self
        return params

    def __wrap_tool(self, tool):
        """Wrap a tool to handle parameter transformation.
        
        Args:
            tool: Tool function to wrap
            
        Returns:
            Wrapped tool function
        """
        @functools.wraps(tool)
        async def wrapper(*args, **kwargs):
            # Transform parameters
            kwargs = self.__transform_tool_params(tool._tool_name, kwargs)
            return await tool(*args, **kwargs)
        
        # Copy tool metadata
        wrapper._is_tool = True
        wrapper._tool_name = tool._tool_name
        wrapper._tool_description = tool._tool_description if hasattr(tool, "_tool_description") else ""
        wrapper._tool_parameters = tool._tool_parameters if hasattr(tool, "_tool_parameters") else {}
        
        return wrapper

    def __setup_tools(self):
        """Register MCP tools."""
        for tool in TOOLS_DEFINITION:
            # Wrap the tool to handle parameter transformation
            wrapped_tool = self.__wrap_tool(tool)
            self.mcp.tool()(wrapped_tool)

    def run(self) -> None:
        """Run the SolrMCP server."""
        logger.info("Starting SolrMCP server...")
        if self.stdio:
            self.mcp.run("stdio")
        else:
            self.mcp.run("sse")

    async def close(self):
        """Clean up resources."""
        if hasattr(self.solr_client, 'close'):
            await self.solr_client.close()
        if hasattr(self.mcp, 'close'):
            await self.mcp.close()

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SolrMCP Server")
    parser.add_argument("--mcp-port", type=int, help="MCP server port", default=int(os.getenv("MCP_PORT", 8081)))
    parser.add_argument("--solr-base-url", help="Solr base URL", default=os.getenv("SOLR_BASE_URL", "http://localhost:8983/solr"))
    parser.add_argument("--zookeeper-hosts", help="ZooKeeper hosts (comma-separated)", default=os.getenv("ZOOKEEPER_HOSTS", "localhost:2181"))
    parser.add_argument("--default-collection", help="Default Solr collection", default=os.getenv("DEFAULT_COLLECTION", "default"))
    parser.add_argument("--connection-timeout", type=int, help="Connection timeout in seconds", default=int(os.getenv("CONNECTION_TIMEOUT", 10)))
    parser.add_argument("--embedding-field", help="Field name for embeddings", default=os.getenv("EMBEDDING_FIELD", "embedding"))
    parser.add_argument("--default-top-k", type=int, help="Default number of results", default=int(os.getenv("DEFAULT_TOP_K", 10)))
    parser.add_argument("--stdio", help="Use stdio instead of HTTP", action="store_true")
    
    args = parser.parse_args()
    
    server = SolrMCPServer(
        mcp_port=args.mcp_port,
        solr_base_url=args.solr_base_url,
        zookeeper_hosts=args.zookeeper_hosts.split(","),
        default_collection=args.default_collection,
        connection_timeout=args.connection_timeout,
        embedding_field=args.embedding_field,
        default_top_k=args.default_top_k,
        stdio=args.stdio
    )
    server.run()

if __name__ == "__main__":
    main()