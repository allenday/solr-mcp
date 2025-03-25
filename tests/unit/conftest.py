"""Test configuration and fixtures.

This module imports and re-exports all fixtures from the fixtures directory,
making them available to all tests without explicit imports.
"""

# Import and expose all fixtures
from unittest.mock import mock_open

from .fixtures.common import *
from .fixtures.config_fixtures import *
from .fixtures.http_fixtures import *
from .fixtures.server_fixtures import *
from .fixtures.solr_fixtures import *
from .fixtures.time_fixtures import *
from .fixtures.vector_fixtures import *
from .fixtures.zookeeper_fixtures import *
