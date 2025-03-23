"""Schema and field management for SolrCloud client."""

import logging
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from solr_mcp.solr.constants import FIELD_TYPE_MAPPING, SYNTHETIC_SORT_FIELDS
from solr_mcp.solr.exceptions import SchemaError, SolrError
from solr_mcp.solr.schema.cache import FieldCache

logger = logging.getLogger(__name__)

class FieldManager:
    """Manages Solr schema fields and field types."""

    def __init__(self, solr_base_url: str):
        """Initialize the field manager.
        
        Args:
            solr_base_url: Base URL for Solr instance
        """
        self.solr_base_url = solr_base_url
        self._schema_cache = {}
        self._field_types_cache = {}
        self.cache = FieldCache()

    def get_schema(self, collection: str) -> Dict:
        """Get schema for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Schema information
            
        Raises:
            SchemaError: If schema cannot be retrieved
        """
        if collection in self._schema_cache:
            return self._schema_cache[collection]
            
        try:
            url = f"{self.solr_base_url}/{collection}/schema"
            response = requests.get(url)
            response.raise_for_status()
            
            schema = response.json()
            if "schema" not in schema:
                raise SchemaError(f"Invalid schema response for collection {collection}")
                
            self._schema_cache[collection] = schema["schema"]
            return self._schema_cache[collection]
            
        except Exception as e:
            logger.error(f"Failed to get schema for collection {collection}: {str(e)}")
            raise SchemaError(f"Failed to get schema for collection {collection}: {str(e)}")

    def get_field_types(self, collection: str) -> Dict[str, str]:
        """Get field types for a collection."""
        if collection in self._field_types_cache:
            return self._field_types_cache[collection]

        schema = self.get_schema(collection)
        field_types = {}
        
        # First map field type names to their definitions
        for field_type in schema.get("fieldTypes", []):
            field_types[field_type["name"]] = field_type["name"]
            
        # Then map fields to their types
        for field in schema.get("fields", []):
            if "name" in field and "type" in field:
                field_types[field["name"]] = field["type"]
                
        self._field_types_cache[collection] = field_types
        return field_types

    def get_field_type(self, collection: str, field_name: str) -> str:
        """Get field type for a specific field."""
        field_types = self.get_field_types(collection)
        if field_name not in field_types:
            raise SchemaError(f"Field not found: {field_name}")
        return field_types[field_name]

    def validate_field_exists(self, collection: str, field: str) -> bool:
        """Validate that a field exists in a collection.
        
        Args:
            collection: Collection name
            field: Field name to validate
            
        Returns:
            True if field exists
            
        Raises:
            SchemaError: If field does not exist
        """
        try:
            field_info = self.get_field_info(collection)
            if field == "*":
                return True
                
            if field not in field_info["searchable_fields"]:
                raise SchemaError(f"Field {field} not found in collection {collection}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating field {field}: {str(e)}")
            raise SchemaError(f"Error validating field {field}: {str(e)}")
            
    def validate_sort_field(self, collection: str, field: str) -> bool:
        """Validate that a field can be used for sorting.
        
        Args:
            collection: Collection name
            field: Field name to validate
            
        Returns:
            True if field can be used for sorting
            
        Raises:
            SchemaError: If field cannot be used for sorting
        """
        try:
            field_info = self.get_field_info(collection)
            if field not in field_info["sortable_fields"]:
                raise SchemaError(f"Field {field} cannot be used for sorting in collection {collection}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating sort field {field}: {str(e)}")
            raise SchemaError(f"Error validating sort field {field}: {str(e)}")

    def get_field_info(self, collection: str, field: Optional[str] = None) -> Dict[str, Any]:
        """Get field information for a collection.
        
        Args:
            collection: Collection name
            field: Optional field name to get specific info for
            
        Returns:
            Field information including searchable and sortable fields
            
        Raises:
            SchemaError: If field info cannot be retrieved
        """
        try:
            schema = self.get_schema(collection)
            
            # Get all fields
            fields = schema.get("fields", [])
            
            # Build field info
            searchable_fields = []
            sortable_fields = {}
            
            for field_def in fields:
                name = field_def.get("name")
                if not name:
                    continue
                    
                # Check if field is searchable
                if field_def.get("indexed", True):
                    searchable_fields.append(name)
                    
                # Check if field is sortable
                if field_def.get("docValues", False) or field_def.get("stored", False):
                    sortable_fields[name] = {
                        "type": field_def.get("type", "string"),
                        "searchable": field_def.get("indexed", True),
                        "directions": ["asc", "desc"],
                        "default_direction": "asc"
                    }
                    
            # Add special fields
            sortable_fields["_docid_"] = {
                "type": "numeric",
                "searchable": False,
                "directions": ["asc", "desc"],
                "default_direction": "asc"
            }
            sortable_fields["score"] = {
                "type": "numeric",
                "searchable": True,
                "directions": ["asc", "desc"],
                "default_direction": "desc"
            }
            
            field_info = {
                "searchable_fields": searchable_fields,
                "sortable_fields": sortable_fields
            }
            
            if field:
                if field in sortable_fields:
                    return sortable_fields[field]
                raise SchemaError(f"Field {field} not found in collection {collection}")
                
            return field_info
            
        except Exception as e:
            logger.error(f"Error getting field info: {str(e)}")
            raise SchemaError(f"Failed to get field info: {str(e)}")

    def validate_collection(self, collection: str) -> bool:
        """Validate that a collection exists.
        
        Args:
            collection: Collection name to validate
            
        Returns:
            True if collection exists
            
        Raises:
            SchemaError: If collection does not exist
        """
        try:
            self.get_schema(collection)
            return True
            
        except Exception as e:
            logger.error(f"Error validating collection {collection}: {str(e)}")
            raise SchemaError(f"Collection {collection} does not exist: {str(e)}")
            
    def clear_cache(self, collection: Optional[str] = None):
        """Clear schema cache.
        
        Args:
            collection: Optional collection name to clear cache for. If None, clears all cache.
        """
        if collection:
            self._schema_cache.pop(collection, None)
            self._field_types_cache.pop(collection, None)
        else:
            self._schema_cache = {}
            self._field_types_cache = {}

    def _get_collection_fields(self, collection: str) -> Dict[str, Any]:
        """Get or load field information for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Dict containing searchable and sortable fields for the collection
        """
        # Check cache first
        if not self.cache.is_stale(collection):
            return self.cache.get(collection)
            
        try:
            searchable_fields = self._get_searchable_fields(collection)
            sortable_fields = self._get_sortable_fields(collection)
            
            field_info = {
                "searchable_fields": searchable_fields,
                "sortable_fields": sortable_fields
            }
            
            # Update cache
            self.cache.set(collection, field_info)
            
            logger.info(f"Loaded field information for collection {collection}")
            logger.debug(f"Searchable fields: {searchable_fields}")
            logger.debug(f"Sortable fields: {sortable_fields}")
            
            return field_info
            
        except Exception as e:
            logger.error(f"Error loading field information for collection {collection}: {e}")
            # Use cached defaults
            return self.cache.get_or_default(collection)

    def _get_searchable_fields(self, collection: str) -> List[str]:
        """Get list of searchable fields for a collection.

        Args:
            collection: Collection name

        Returns:
            List of field names that can be searched
        """
        try:
            # Try schema API first
            schema_url = f"{collection}/schema/fields?wt=json"
            logger.debug(f"Getting searchable fields from schema URL: {schema_url}")
            full_url = f"{self.solr_base_url}/{schema_url}"
            logger.debug(f"Full URL: {full_url}")
            
            response = requests.get(full_url)
            fields_data = response.json()
            
            searchable_fields = []
            for field in fields_data.get("fields", []):
                field_name = field.get("name")
                field_type = field.get("type")
                
                # Skip special fields
                if field_name.startswith("_") and field_name not in ["_text_"]:
                    continue
                    
                # Add text and string fields
                if field_type in ["text_general", "string"] or "text" in field_type:
                    logger.debug(f"Found searchable field: {field_name}, type: {field_type}")
                    searchable_fields.append(field_name)
            
            # Add known content fields
            content_fields = ["content", "title", "_text_"]
            for field in content_fields:
                if field not in searchable_fields:
                    searchable_fields.append(field)
                    
            logger.info(f"Using searchable fields for collection {collection}: {searchable_fields}")
            return searchable_fields
            
        except Exception as e:
            logger.warning(f"Error getting schema fields: {str(e)}")
            logger.info("Fallback: trying direct URL with query that returns field info")
            
            try:
                direct_url = f"{self.solr_base_url}/{collection}/select?q=*:*&rows=0&wt=json"
                logger.debug(f"Trying direct URL: {direct_url}")
                
                response = requests.get(direct_url)
                response_data = response.json()
                
                # Extract fields from response header
                fields = []
                if "responseHeader" in response_data:
                    header = response_data["responseHeader"]
                    if "params" in header and "fl" in header["params"]:
                        fields = header["params"]["fl"].split(",")
                
                # Add known searchable fields
                fields.extend(["content", "title", "_text_"])
                searchable_fields = list(set(fields))  # Remove duplicates
                
            except Exception as e2:
                logger.error(f"Error getting searchable fields: {str(e2)}")
                logger.info("Using fallback searchable fields: ['content', 'title', '_text_']")
                searchable_fields = ["content", "title", "_text_"]
                
            logger.info(f"Using searchable fields for collection {collection}: {searchable_fields}")
            return searchable_fields

    def _get_sortable_fields(self, collection: str) -> Dict[str, Dict[str, Any]]:
        """Get list of sortable fields and their properties for a collection.

        Args:
            collection: Collection name

        Returns:
            Dict mapping field names to their properties
        """
        try:
            # Try schema API first
            schema_url = f"{collection}/schema/fields?wt=json"
            logger.debug(f"Getting sortable fields from schema URL: {schema_url}")
            full_url = f"{self.solr_base_url}/{schema_url}"
            logger.debug(f"Full URL: {full_url}")
            
            response = requests.get(full_url)
            fields_data = response.json()
            
            sortable_fields = {}
            
            # Process schema fields
            for field in fields_data.get("fields", []):
                field_name = field.get("name")
                field_type = field.get("type")
                multi_valued = field.get("multiValued", False)
                doc_values = field.get("docValues", False)
                
                # Skip special fields, multi-valued fields, and fields without a recognized type
                if (field_name.startswith('_') and field_name not in SYNTHETIC_SORT_FIELDS) or \
                   multi_valued or \
                   field_type not in FIELD_TYPE_MAPPING:
                    continue
                
                # Add field to sortable fields
                sortable_fields[field_name] = {
                    "type": FIELD_TYPE_MAPPING[field_type],
                    "directions": ["asc", "desc"],
                    "default_direction": "asc" if FIELD_TYPE_MAPPING[field_type] in ["string", "numeric", "date"] else "desc",
                    "searchable": True  # Regular schema fields are searchable
                }
            
            # Add synthetic fields
            sortable_fields.update(SYNTHETIC_SORT_FIELDS)
            
            return sortable_fields
            
        except Exception as e:
            logger.error(f"Error getting sortable fields: {e}")
            # Return only the guaranteed score field
            return {"score": SYNTHETIC_SORT_FIELDS["score"]}

    def validate_fields(self, collection: str, fields: List[str]) -> None:
        """Validate that the requested fields exist in the collection.
        
        Args:
            collection: Collection name
            fields: List of field names to validate
            
        Raises:
            SchemaError: If any field is not valid for the collection
        """
        collection_info = self._get_collection_fields(collection)
        searchable_fields = collection_info["searchable_fields"]
        sortable_fields = collection_info["sortable_fields"]
        
        # Combine all valid fields
        valid_fields = set(searchable_fields) | set(sortable_fields.keys())
        
        # Check each requested field
        invalid_fields = [f for f in fields if f not in valid_fields]
        if invalid_fields:
            raise SchemaError(f"Invalid fields for collection {collection}: {', '.join(invalid_fields)}")

    def validate_sort_fields(self, collection: str, sort_fields: List[str]) -> None:
        """Validate that the requested sort fields are sortable in the collection.
        
        Args:
            collection: Collection name
            sort_fields: List of field names to validate for sorting
            
        Raises:
            SchemaError: If any field is not sortable in the collection
        """
        collection_info = self._get_collection_fields(collection)
        sortable_fields = collection_info["sortable_fields"]
        
        # Check each sort field
        invalid_fields = [f for f in sort_fields if f not in sortable_fields]
        if invalid_fields:
            raise SchemaError(f"Fields not sortable in collection {collection}: {', '.join(invalid_fields)}")

    def validate_collection_exists(self, collection: str) -> bool:
        """Validate that a collection exists.
        
        Args:
            collection: Collection name to validate
            
        Returns:
            True if collection exists
            
        Raises:
            SchemaError: If collection does not exist
        """
        try:
            self.get_schema(collection)
            return True
        except SchemaError:
            raise SchemaError(f"Collection {collection} does not exist")
            
    def validate_field_exists(self, field: str, collection: str) -> bool:
        """Validate that a field exists in a collection.
        
        Args:
            field: Field name to validate
            collection: Collection name
            
        Returns:
            True if field exists
            
        Raises:
            SchemaError: If field does not exist
        """
        try:
            field_info = self.get_field_info(collection)
            if field not in field_info["searchable_fields"]:
                raise SchemaError(f"Field {field} not found in collection {collection}")
            return True
        except SchemaError:
            raise
            
    def validate_sort_field(self, field: str, collection: str) -> bool:
        """Validate that a field can be used for sorting.
        
        Args:
            field: Field name to validate
            collection: Collection name
            
        Returns:
            True if field is sortable
            
        Raises:
            SchemaError: If field is not sortable
        """
        try:
            field_info = self.get_field_info(collection)
            if field not in field_info["sortable_fields"]:
                raise SchemaError(f"Field {field} is not sortable in collection {collection}")
            return True
        except SchemaError:
            raise 