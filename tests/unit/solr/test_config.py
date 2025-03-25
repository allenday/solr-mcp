"""Tests for solr_mcp.solr.config module."""

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import mock_open, patch

import pydantic
import pytest
from pydantic import ValidationError

from solr_mcp.solr.config import SolrConfig
from solr_mcp.solr.exceptions import ConfigurationError


@pytest.fixture
def valid_config_dict() -> Dict[str, Any]:
    """Create a valid configuration dictionary."""
    return {
        "solr_base_url": "http://localhost:8983/solr",
        "zookeeper_hosts": ["localhost:2181"],
        "connection_timeout": 10,
    }


@pytest.fixture
def temp_config_file(tmp_path: Path, valid_config_dict: Dict[str, Any]) -> Path:
    """Create a temporary configuration file."""
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(valid_config_dict, f)
    return config_file


class TestSolrConfig:
    """Test cases for SolrConfig class."""

    def test_init_with_valid_config(self, valid_config_dict):
        """Test initialization with valid configuration."""
        config = SolrConfig(**valid_config_dict)
        assert config.solr_base_url == valid_config_dict["solr_base_url"]
        assert config.zookeeper_hosts == valid_config_dict["zookeeper_hosts"]
        assert config.connection_timeout == valid_config_dict["connection_timeout"]

    def test_init_with_minimal_config(self):
        """Test initialization with minimal required configuration."""
        config = SolrConfig(
            solr_base_url="http://localhost:8983/solr",
            zookeeper_hosts=["localhost:2181"],
        )
        assert config.solr_base_url == "http://localhost:8983/solr"
        assert config.zookeeper_hosts == ["localhost:2181"]
        assert config.connection_timeout == 10

    def test_init_missing_required_fields(self):
        """Test initialization with missing required fields."""
        with pytest.raises(ConfigurationError, match="solr_base_url is required"):
            SolrConfig(zookeeper_hosts=["localhost:2181"])

        with pytest.raises(ConfigurationError, match="zookeeper_hosts is required"):
            SolrConfig(solr_base_url="http://localhost:8983/solr")

    def test_validate_solr_url(self):
        """Test validation of Solr base URL."""
        with pytest.raises(ConfigurationError, match="solr_base_url is required"):
            SolrConfig(solr_base_url="", zookeeper_hosts=["localhost:2181"])

        with pytest.raises(
            ConfigurationError,
            match="Solr base URL must start with http:// or https://",
        ):
            SolrConfig(solr_base_url="invalid_url", zookeeper_hosts=["localhost:2181"])

        # Test HTTPS URL
        config = SolrConfig(
            solr_base_url="https://localhost:8983/solr",
            zookeeper_hosts=["localhost:2181"],
        )
        assert config.solr_base_url == "https://localhost:8983/solr"

    def test_validate_zookeeper_hosts(self):
        """Test validation of ZooKeeper hosts."""
        # Test empty list
        with pytest.raises(ConfigurationError, match="zookeeper_hosts is required"):
            SolrConfig(solr_base_url="http://localhost:8983/solr", zookeeper_hosts=[])

        # Test non-string hosts
        with pytest.raises(ConfigurationError, match="Input should be a valid string"):
            SolrConfig(
                solr_base_url="http://localhost:8983/solr", zookeeper_hosts=[123]
            )

        # Test multiple valid hosts
        config = SolrConfig(
            solr_base_url="http://localhost:8983/solr",
            zookeeper_hosts=["host1:2181", "host2:2181"],
        )
        assert config.zookeeper_hosts == ["host1:2181", "host2:2181"]

    def test_validate_numeric_fields(self):
        """Test validation of numeric fields."""
        # Test zero values
        with pytest.raises(
            ConfigurationError, match="connection_timeout must be positive"
        ):
            SolrConfig(
                solr_base_url="http://localhost:8983/solr",
                zookeeper_hosts=["localhost:2181"],
                connection_timeout=0,
            )

        # Test negative values
        with pytest.raises(
            ConfigurationError, match="connection_timeout must be positive"
        ):
            SolrConfig(
                solr_base_url="http://localhost:8983/solr",
                zookeeper_hosts=["localhost:2181"],
                connection_timeout=-1,
            )

    def test_validate_config(self):
        """Test complete configuration validation."""
        # Test empty solr_base_url
        with pytest.raises(ConfigurationError, match="solr_base_url is required"):
            SolrConfig(solr_base_url="", zookeeper_hosts=["localhost:2181"])

        # Test empty zookeeper_hosts
        with pytest.raises(ConfigurationError, match="zookeeper_hosts is required"):
            SolrConfig(solr_base_url="http://localhost:8983/solr", zookeeper_hosts=[])

        # Test invalid connection_timeout
        with pytest.raises(
            ConfigurationError, match="connection_timeout must be positive"
        ):
            SolrConfig(
                solr_base_url="http://localhost:8983/solr",
                zookeeper_hosts=["localhost:2181"],
                connection_timeout=0,
            )

    def test_load_from_file(self, temp_config_file):
        """Test loading configuration from file."""
        config = SolrConfig.load(str(temp_config_file))
        assert isinstance(config, SolrConfig)
        assert config.solr_base_url == "http://localhost:8983/solr"
        assert config.zookeeper_hosts == ["localhost:2181"]

    def test_load_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            SolrConfig.load("nonexistent.json")

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        invalid_json = tmp_path / "invalid.json"
        with open(invalid_json, "w") as f:
            f.write("invalid json")

        with pytest.raises(
            ConfigurationError, match="Invalid JSON in configuration file"
        ):
            SolrConfig.load(str(invalid_json))

    def test_load_invalid_config(self, tmp_path):
        """Test loading file with invalid configuration."""
        invalid_config = tmp_path / "invalid_config.json"
        with open(invalid_config, "w") as f:
            json.dump({"invalid": "config"}, f)

        with pytest.raises(ConfigurationError, match="solr_base_url is required"):
            SolrConfig.load(str(invalid_config))

    def test_load_with_generic_error(self):
        """Test loading with generic error."""
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = Exception("Generic error")
            with pytest.raises(
                ConfigurationError, match="Failed to load config: Generic error"
            ):
                SolrConfig.load("config.json")

    def test_to_dict(self, valid_config_dict):
        """Test conversion to dictionary."""
        config = SolrConfig(**valid_config_dict)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["solr_base_url"] == valid_config_dict["solr_base_url"]
        assert config_dict["zookeeper_hosts"] == valid_config_dict["zookeeper_hosts"]

    def test_model_validate_method(self):
        """Test model_validate method."""
        config = SolrConfig(
            solr_base_url="http://localhost:8983/solr",
            zookeeper_hosts=["localhost:2181"],
        )

        valid_data = {
            "solr_base_url": "http://localhost:8983/solr",
            "zookeeper_hosts": ["localhost:2181"],
        }
        result = config.model_validate(valid_data)
        assert isinstance(result, SolrConfig)

    def test_model_validate_with_additional_fields(self):
        """Test model validation with additional fields."""
        config = SolrConfig(
            solr_base_url="http://localhost:8983/solr",
            zookeeper_hosts=["localhost:2181"],
        )

        data_with_extra = {
            "solr_base_url": "http://localhost:8983/solr",
            "zookeeper_hosts": ["localhost:2181"],
            "extra_field": "value",
        }
        result = config.model_validate(data_with_extra)
        assert isinstance(result, SolrConfig)
        assert not hasattr(result, "extra_field")

    def test_model_validate_with_type_conversion(self):
        """Test model validation with type conversion."""
        config = SolrConfig(
            solr_base_url="http://localhost:8983/solr",
            zookeeper_hosts=["localhost:2181"],
        )

        data = {
            "solr_base_url": "http://localhost:8983/solr",
            "zookeeper_hosts": ["localhost:2181"],
            "connection_timeout": "20",  # String that should be converted to int
        }
        result = config.model_validate(data)
        assert isinstance(result.connection_timeout, int)
        assert result.connection_timeout == 20
