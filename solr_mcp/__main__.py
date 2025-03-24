"""Main entry point for the Solr MCP server."""

import asyncio
from solr_mcp.server import run_server

def main():
    """Main entry point."""
    asyncio.run(run_server())

if __name__ == "__main__":
    main() 