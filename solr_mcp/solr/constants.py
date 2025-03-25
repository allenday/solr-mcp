"""Constants for SolrCloud client."""

# Field type mapping for sorting
FIELD_TYPE_MAPPING = {
    "string": "string",
    "text_general": "text",
    "text_en": "text",
    "int": "numeric",
    "long": "numeric",
    "float": "numeric",
    "double": "numeric",
    "date": "date",
    "boolean": "boolean",
}

# Synthetic fields that can be used for sorting
SYNTHETIC_SORT_FIELDS = {
    "score": {
        "type": "numeric",
        "directions": ["asc", "desc"],
        "default_direction": "desc",
        "searchable": True,
    },
    "_docid_": {
        "type": "numeric",
        "directions": ["asc", "desc"],
        "default_direction": "asc",
        "searchable": False,
        "warning": "Internal Lucene document ID. Not stable across restarts or reindexing.",
    },
}
