"""Configuration for Solr client."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pydantic
from pydantic import BaseModel, Field, field_validator, model_validator

from solr_mcp.solr.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class SolrConfig(BaseModel):
    """Configuration for Solr client."""

    solr_base_url: str = Field(description="Base URL for Solr instance")
    zookeeper_hosts: List[str] = Field(description="List of ZooKeeper hosts")
    connection_timeout: int = Field(
        default=10, gt=0, description="Connection timeout in seconds"
    )

    def __init__(self, **data):
        """Initialize SolrConfig with validation error handling."""
        try:
            super().__init__(**data)
        except pydantic.ValidationError as e:
            # Convert Pydantic validation errors to our custom ConfigurationError
            for error in e.errors():
                if error["type"] == "missing":
                    field = error["loc"][0]
                    raise ConfigurationError(f"{field} is required")
                elif error["type"] == "greater_than":
                    field = error["loc"][0]
                    if field == "connection_timeout":
                        raise ConfigurationError("connection_timeout must be positive")
            # If we get here, it's some other validation error
            raise ConfigurationError(str(e))

    @field_validator("solr_base_url")
    def validate_solr_url(cls, v: str) -> str:
        """Validate Solr base URL."""
        if not v:
            raise ConfigurationError("solr_base_url is required")
        if not v.startswith(("http://", "https://")):
            raise ConfigurationError(
                "Solr base URL must start with http:// or https://"
            )
        return v

    @field_validator("zookeeper_hosts")
    def validate_zookeeper_hosts(cls, v: List[str]) -> List[str]:
        """Validate ZooKeeper hosts."""
        if not v:
            raise ConfigurationError("zookeeper_hosts is required")
        if not all(isinstance(host, str) for host in v):
            raise ConfigurationError("ZooKeeper hosts must be strings")
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "SolrConfig":
        """Validate the complete configuration."""
        # Validate solr_base_url
        if not self.solr_base_url:
            raise ConfigurationError("solr_base_url is required")

        # Validate zookeeper_hosts
        if not self.zookeeper_hosts:
            raise ConfigurationError("zookeeper_hosts is required")

        # Validate numeric fields
        if self.connection_timeout <= 0:
            raise ConfigurationError("connection_timeout must be positive")

        return self

    @classmethod
    def load(cls, config_path: str) -> "SolrConfig":
        """Load configuration from JSON file.

        Args:
            config_path: Path to JSON config file

        Returns:
            SolrConfig instance

        Raises:
            ConfigurationError: If file cannot be loaded or is invalid
        """
        try:
            with open(config_path) as f:
                config_dict = json.load(f)

            try:
                return cls(**config_dict)
            except pydantic.ValidationError as e:
                # Convert Pydantic validation errors to our custom ConfigurationError
                for error in e.errors():
                    if error["type"] == "missing":
                        field = error["loc"][0]
                        raise ConfigurationError(f"{field} is required")
                    elif error["type"] == "greater_than":
                        field = error["loc"][0]
                        if field == "connection_timeout":
                            raise ConfigurationError(
                                "connection_timeout must be positive"
                            )
                # If we get here, it's some other validation error
                raise ConfigurationError(str(e))

        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        except json.JSONDecodeError:
            raise ConfigurationError(
                f"Invalid JSON in configuration file: {config_path}"
            )

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load config: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    def model_validate(cls, *args, **kwargs):
        """Override model_validate to handle validation errors."""
        try:
            return super().model_validate(*args, **kwargs)
        except pydantic.ValidationError as e:
            # Convert Pydantic validation errors to our custom ConfigurationError
            for error in e.errors():
                if error["type"] == "missing":
                    field = error["loc"][0]
                    raise ConfigurationError(f"{field} is required")
                elif error["type"] == "greater_than":
                    field = error["loc"][0]
                    if field == "connection_timeout":
                        raise ConfigurationError("connection_timeout must be positive")
            # If we get here, it's some other validation error
            raise ConfigurationError(str(e))
