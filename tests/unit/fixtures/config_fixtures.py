"""Configuration fixtures for unit tests."""

import pytest
from unittest.mock import Mock, patch, mock_open

from solr_mcp.solr.config import SolrConfig


@pytest.fixture
def mock_config(request):
    """Parameterized SolrConfig mock.
    
    Args:
        request: Pytest request object that can contain parameters:
            - base_url: Custom Solr base URL
            - zk_hosts: Custom ZooKeeper hosts
            - timeout: Custom connection timeout
    """
    # Get parameters or use defaults
    base_url = getattr(request, 'param', {}).get('base_url', "http://localhost:8983/solr")
    zk_hosts = getattr(request, 'param', {}).get('zk_hosts', ["localhost:2181"])
    timeout = getattr(request, 'param', {}).get('timeout', 10)
    
    config = Mock(spec=SolrConfig)
    config.solr_base_url = base_url
    config.zookeeper_hosts = zk_hosts
    config.connection_timeout = timeout
    
    return config


@pytest.fixture(params=[
    # Format: (fixture_name, content, side_effect)
    ("valid", """
    {
        "solr_base_url": "http://solr:8983/solr",
        "zookeeper_hosts": ["zk1:2181", "zk2:2181"],
        "connection_timeout": 30
    }
    """, None),
    ("invalid_json", "invalid json content", None),
    ("minimal", """
    {
        "zookeeper_hosts": ["zk1:2181"]
    }
    """, None),
    ("missing", None, FileNotFoundError())
])
def mock_config_file(request):
    """Parameterized fixture for different config file scenarios."""
    fixture_name, content, side_effect = request.param
    
    if side_effect:
        with patch("builtins.open", side_effect=side_effect):
            yield fixture_name
    else:
        with patch("builtins.open", mock_open(read_data=content)):
            yield fixture_name


@pytest.fixture
def mock_field_manager_methods():
    """Mock FieldManager methods for testing."""
    mock_fields = {
        "searchable_fields": ["title", "content"],
        "sortable_fields": {
            "id": {
                "directions": ["asc", "desc"],
                "default_direction": "asc"
            },
            "score": {
                "directions": ["asc", "desc"],
                "default_direction": "desc",
                "type": "numeric",
                "searchable": True
            }
        }
    }

    def patch_get_collection_fields(field_manager):
        """Create a context manager for patching _get_collection_fields."""
        return patch.object(field_manager, '_get_collection_fields', return_value=mock_fields)

    def patch_get_searchable_fields(field_manager):
        """Create a context manager for patching _get_searchable_fields."""
        return patch.object(field_manager, '_get_searchable_fields', side_effect=Exception("API error"))

    return {
        "mock_fields": mock_fields,
        "patch_get_collection_fields": patch_get_collection_fields,
        "patch_get_searchable_fields": patch_get_searchable_fields
    }