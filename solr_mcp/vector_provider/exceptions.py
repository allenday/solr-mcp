"""Exceptions for vector provider module."""


class VectorError(Exception):
    """Base exception for vector-related errors."""

    pass


class VectorGenerationError(VectorError):
    """Raised when vector generation fails."""

    pass


class VectorConfigError(VectorError):
    """Raised when there is an error in vector provider configuration."""

    pass


class VectorConnectionError(VectorError):
    """Raised when connection to vector service fails."""

    pass
