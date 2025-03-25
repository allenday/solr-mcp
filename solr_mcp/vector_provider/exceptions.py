"""Exceptions for embeddings module."""

class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass

class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""
    pass

class EmbeddingConfigError(EmbeddingError):
    """Raised when there is an error in embedding configuration."""
    pass

class EmbeddingConnectionError(EmbeddingError):
    """Raised when connection to embedding service fails."""
    pass 