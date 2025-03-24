"""Constants for embeddings module."""

from typing import Dict, Any

# Default configuration for embedding providers
DEFAULT_OLLAMA_CONFIG: Dict[str, Any] = {
    "base_url": "http://localhost:11434",
    "model": "nomic-embed-text",
    "timeout": 30,  # seconds
    "retries": 3
}

# Environment variable names
ENV_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"

# HTTP endpoints
OLLAMA_EMBEDDINGS_PATH = "/api/embeddings"

# Model-specific constants
MODEL_DIMENSIONS = {
    "nomic-embed-text": 768  # 768-dimensional embeddings
} 