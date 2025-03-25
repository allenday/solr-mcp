"""Tool for listing fields in a Solr collection."""

from typing import Any, Dict

from solr_mcp.tools.tool_decorator import tool


@tool()
async def execute_list_fields(mcp: str, collection: str) -> Dict[str, Any]:
    """List all fields in a Solr collection.

    This tool provides detailed information about each field in a Solr collection,
    including how fields are related through copyField directives. Pay special
    attention to fields that have 'copies_from' data - these are aggregate fields
    that combine content from multiple source fields.

    For example, the '_text_' field is typically an aggregate field that combines
    content from many text fields to provide a unified search experience. When you
    see a field with 'copies_from' data, it means that field contains a copy of
    the content from all the listed source fields.

    Args:
        mcp: MCP instance name
        collection: Name of the collection to get fields from

    Returns:
        Dictionary containing:
        - fields: List of field definitions with their properties including:
            - name: Field name
            - type: Field type (text_general, string, etc)
            - indexed: Whether the field is indexed for searching
            - stored: Whether the field values are stored
            - docValues: Whether the field can be used for sorting/faceting
            - multiValued: Whether the field can contain multiple values
            - copies_from: List of source fields that copy their content to this field
        - collection: Name of the collection queried
    """
    fields = await mcp.solr_client.list_fields(collection)

    return {"fields": fields, "collection": collection}
