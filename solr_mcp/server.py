"""FastMCP server implementation for Solr."""

import argparse
import functools
import logging
import os
import sys
from typing import List

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

from solr_mcp.solr.client import SolrClient
from solr_mcp.solr.config import SolrConfig
from solr_mcp.tools import TOOLS_DEFINITION

logger = logging.getLogger(__name__)


class SolrMCPServer:
    """Model Context Protocol server for SolrCloud integration."""

    def __init__(
        self,
        mcp_port: int = int(os.getenv("MCP_PORT", 8081)),
        solr_base_url: str = os.getenv("SOLR_BASE_URL", "http://localhost:8983/solr"),
        zookeeper_hosts: List[str] = os.getenv(
            "ZOOKEEPER_HOSTS", "localhost:2181"
        ).split(","),
        connection_timeout: int = int(os.getenv("CONNECTION_TIMEOUT", 10)),
        stdio: bool = False,
    ):
        """Initialize the server."""
        self.port = mcp_port
        self.config = SolrConfig(
            solr_base_url=solr_base_url,
            zookeeper_hosts=zookeeper_hosts,
            connection_timeout=connection_timeout,
        )
        self.stdio = stdio
        self._setup_server()

    def _setup_server(self):
        """Set up the MCP server and Solr client."""
        try:
            self._connect_to_solr()
        except Exception as e:
            logger.error(f"Solr connection error: {e}")
            sys.exit(1)

        logger.info(f"Server starting on port {self.port}")

        # Create FastMCP instance
        self.mcp = FastMCP(
            name="Solr MCP Server",
            instructions="""This server provides tools for interacting with SolrCloud:
- List collections
- Execute SQL queries
- Execute semantic search queries
- Execute vector search queries""",
            debug=True,
            port=self.port,
        )

        # Register tools
        self._setup_tools()

    def _connect_to_solr(self):
        """Initialize Solr client connection."""
        self.solr_client = SolrClient(config=self.config)

    def _transform_tool_params(self, tool_name: str, params: dict) -> dict:
        """Transform tool parameters before they are passed to the tool."""
        if "mcp" in params:
            if isinstance(params["mcp"], str):
                # If mcp is passed as a string (server name), use self as the server instance
                params["mcp"] = self
        return params

    def _wrap_tool(self, tool):
        """Wrap a tool to handle parameter transformation."""

        @functools.wraps(tool)
        async def wrapper(*args, **kwargs):
            # Transform parameters
            kwargs = self._transform_tool_params(tool.__name__, kwargs)
            result = await tool(*args, **kwargs)
            return result

        # Copy tool metadata
        wrapper._is_tool = True
        wrapper._tool_name = tool.__name__
        wrapper._tool_description = tool.__doc__ if tool.__doc__ else ""

        return wrapper

    def _setup_tools(self):
        """Register MCP tools."""
        for tool in TOOLS_DEFINITION:
            # Wrap the tool to handle parameter transformation
            wrapped_tool = self._wrap_tool(tool)
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
        if hasattr(self.solr_client, "close"):
            await self.solr_client.close()
        if hasattr(self.mcp, "close"):
            await self.mcp.close()


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided MCP server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SolrMCP Server")
    parser.add_argument(
        "--mcp-port",
        type=int,
        help="MCP server port",
        default=int(os.getenv("MCP_PORT", 8081)),
    )
    parser.add_argument(
        "--solr-base-url",
        help="Solr base URL",
        default=os.getenv("SOLR_BASE_URL", "http://localhost:8983/solr"),
    )
    parser.add_argument(
        "--zookeeper-hosts",
        help="ZooKeeper hosts (comma-separated)",
        default=os.getenv("ZOOKEEPER_HOSTS", "localhost:2181"),
    )
    parser.add_argument(
        "--connection-timeout",
        type=int,
        help="Connection timeout in seconds",
        default=int(os.getenv("CONNECTION_TIMEOUT", 10)),
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport mode (stdio or sse)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (for SSE mode)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on (for SSE mode)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level))

    server = SolrMCPServer(
        mcp_port=args.mcp_port,
        solr_base_url=args.solr_base_url,
        zookeeper_hosts=args.zookeeper_hosts.split(","),
        connection_timeout=args.connection_timeout,
        stdio=(args.transport == "stdio"),
    )

    if args.transport == "stdio":
        server.run()
    else:
        mcp_server = server.mcp._mcp_server  # noqa: WPS437
        starlette_app = create_starlette_app(mcp_server, debug=True)
        import uvicorn

        uvicorn.run(starlette_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
