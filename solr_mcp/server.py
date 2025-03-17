"""Main MCP server implementation for SolrCloud integration."""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import Resource
from loguru import logger

from solr_mcp.solr.client import SolrClient
from solr_mcp.tools.search import SOLR_SEARCH_TOOL
from solr_mcp.tools.suggestions import SOLR_SUGGESTIONS_TOOL
from solr_mcp.tools.facets import SOLR_FACETS_TOOL
from solr_mcp.tools.vector_search import SOLR_VECTOR_SEARCH_TOOL
from solr_mcp.tools.hybrid_search import SOLR_HYBRID_SEARCH_TOOL
from solr_mcp.tools.embedding import SOLR_EMBEDDING_TOOL, SOLR_BATCH_EMBEDDING_TOOL


class SolrMCPServer:
    """Model Context Protocol server for SolrCloud integration."""

    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        """Initialize the SolrMCP server.
        
        Args:
            config_path: Path to the configuration file
            debug: Enable debug mode
        """
        self.debug = debug
        self.config_path = config_path
        
        # Initialize FastMCP
        self.server = FastMCP(name="solr-mcp", debug=debug)
        
        # Initialize Solr client
        self.solr_client = SolrClient(config_path)
        
        # Configure logging
        log_level = "DEBUG" if debug else "INFO"
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        
        # Register tool handlers
        self._register_handlers()
        
        logger.info("SolrMCP server initialized")
    
    def _register_handlers(self) -> None:
        """Register MCP tools."""
        # Add tools with handlers
        self.server.tool(name=SOLR_SEARCH_TOOL["name"], 
                         description=SOLR_SEARCH_TOOL["description"])(self.handle_search)
        
        self.server.tool(name=SOLR_SUGGESTIONS_TOOL["name"], 
                         description=SOLR_SUGGESTIONS_TOOL["description"])(self.handle_suggestions)
        
        self.server.tool(name=SOLR_FACETS_TOOL["name"], 
                         description=SOLR_FACETS_TOOL["description"])(self.handle_facets)
        
        self.server.tool(name=SOLR_VECTOR_SEARCH_TOOL["name"], 
                         description=SOLR_VECTOR_SEARCH_TOOL["description"])(self.handle_vector_search)
        
        self.server.tool(name=SOLR_HYBRID_SEARCH_TOOL["name"], 
                         description=SOLR_HYBRID_SEARCH_TOOL["description"])(self.handle_hybrid_search)
        
        self.server.tool(name=SOLR_EMBEDDING_TOOL["name"], 
                         description=SOLR_EMBEDDING_TOOL["description"])(self.handle_embed_and_index)
        
        self.server.tool(name=SOLR_BATCH_EMBEDDING_TOOL["name"], 
                         description=SOLR_BATCH_EMBEDDING_TOOL["description"])(self.handle_batch_embed_and_index)
    
    async def handle_search(self, query: str, collection: Optional[str] = None, 
                           fields: Optional[List[str]] = None, filters: Optional[List[str]] = None,
                           rows: int = 10, start: int = 0, sort: Optional[str] = None,
                           ctx: Any = None) -> str:
        """Handle solr_search tool requests."""
        logger.debug(f"Handling search request: {query}")
        
        try:
            results = await self.solr_client.search(
                query=query,
                collection=collection,
                fields=fields,
                filters=filters,
                rows=rows,
                start=start,
                sort=sort
            )
            return results
        except Exception as e:
            logger.exception(f"Error in search: {e}")
            raise
    
    async def handle_suggestions(self, query: str, collection: Optional[str] = None,
                                suggestion_field: str = "suggest", count: int = 5,
                                ctx: Any = None) -> str:
        """Handle solr_suggestions tool requests."""
        logger.debug(f"Handling suggestions request: {query}")
        
        try:
            suggestions = await self.solr_client.get_suggestions(
                query=query,
                collection=collection,
                suggestion_field=suggestion_field,
                count=count
            )
            return suggestions
        except Exception as e:
            logger.exception(f"Error in suggestions: {e}")
            raise
    
    async def handle_facets(self, query: str, facet_fields: List[str],
                           collection: Optional[str] = None, facet_limit: int = 10,
                           facet_mincount: int = 1, ctx: Any = None) -> str:
        """Handle solr_facets tool requests."""
        logger.debug(f"Handling facets request: {query}, fields: {facet_fields}")
        
        try:
            facets = await self.solr_client.get_facets(
                query=query,
                facet_fields=facet_fields,
                collection=collection,
                facet_limit=facet_limit,
                facet_mincount=facet_mincount
            )
            return facets
        except Exception as e:
            logger.exception(f"Error in facets: {e}")
            raise
    
    async def handle_vector_search(self, vector: List[float], vector_field: str = "embedding",
                                  collection: Optional[str] = None, k: int = 10,
                                  filter_query: Optional[str] = None, 
                                  return_fields: Optional[List[str]] = None,
                                  ctx: Any = None) -> str:
        """Handle solr_vector_search tool requests."""
        logger.debug(f"Handling vector search request, vector length: {len(vector)}")
        
        try:
            results = await self.solr_client.vector_search(
                vector=vector,
                vector_field=vector_field,
                collection=collection,
                k=k,
                filter_query=filter_query,
                return_fields=return_fields
            )
            return results
        except Exception as e:
            logger.exception(f"Error in vector search: {e}")
            raise
    
    async def handle_embed_and_index(self, document: Dict[str, Any],
                                    collection: Optional[str] = None,
                                    text_field: str = "text",
                                    vector_field: str = "embedding",
                                    commit: bool = True,
                                    ctx: Any = None) -> Dict[str, Any]:
        """Handle solr_embed_and_index tool requests."""
        logger.debug(f"Handling embed and index request: {document.get('id', 'unknown')}")
        
        try:
            success = await self.solr_client.index_document_with_generated_embedding(
                document=document,
                collection=collection,
                text_field=text_field,
                vector_field=vector_field,
                commit=commit
            )
            return {"success": success, "id": document.get("id", "unknown")}
        except Exception as e:
            logger.exception(f"Error in embed and index: {e}")
            raise
    
    async def handle_batch_embed_and_index(self, documents: List[Dict[str, Any]],
                                         collection: Optional[str] = None,
                                         text_field: str = "text",
                                         vector_field: str = "embedding",
                                         commit: bool = True,
                                         ctx: Any = None) -> Dict[str, Any]:
        """Handle solr_batch_embed_and_index tool requests."""
        logger.debug(f"Handling batch embed and index request: {len(documents)} documents")
        
        try:
            success = await self.solr_client.batch_index_with_generated_embeddings(
                documents=documents,
                collection=collection,
                text_field=text_field,
                vector_field=vector_field,
                commit=commit
            )
            return {"success": success, "count": len(documents)}
        except Exception as e:
            logger.exception(f"Error in batch embed and index: {e}")
            raise
    
    async def handle_hybrid_search(self, query: str, collection: Optional[str] = None,
                                  vector_field: str = "embedding", blend_factor: float = 0.5,
                                  fields: Optional[List[str]] = None, filter_query: Optional[str] = None,
                                  rows: int = 10, ctx: Any = None) -> str:
        """Handle solr_hybrid_search tool requests."""
        logger.debug(f"Handling hybrid search request: {query}")
        
        try:
            results = await self.solr_client.hybrid_search(
                query=query,
                collection=collection,
                vector_field=vector_field,
                blend_factor=blend_factor,
                fields=fields,
                filter_query=filter_query,
                rows=rows
            )
            return results
        except Exception as e:
            logger.exception(f"Error in hybrid search: {e}")
            raise
    
    async def run(self) -> None:
        """Run the SolrMCP server."""
        logger.info("Starting SolrMCP server")
        self.server.run("stdio")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SolrMCP Server")
    parser.add_argument("--config", help="Path to config file", default=None)
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    
    args = parser.parse_args()
    
    server = SolrMCPServer(config_path=args.config, debug=args.debug)
    
    # MCP.run() is synchronous, no need for asyncio.run
    server.server.run()


if __name__ == "__main__":
    main()