"""Mem0 settings configuration."""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True, slots=True)
class Mem0Settings:
    """Configuration for Mem0 persistent memory integration."""

    # Qdrant
    collection_name: str = "ollama-agent"
    host: str = "localhost"
    port: int = 6333
    embedding_model_dims: int = 768

    # LLM
    llm_model: str = "llama3.1:latest"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    ollama_base_url: str = "http://localhost:11434"

    # Embedder
    embedder_model: str = "nomic-embed-text:latest"
    embedder_base_url: str = "http://localhost:11434"

    # User
    user_id: str = "default"

    def to_mem0_config(self) -> dict[str, Any]:
        """Build Mem0 configuration dictionary."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.collection_name,
                    "host": self.host,
                    "port": self.port,
                    "embedding_model_dims": self.embedding_model_dims,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self.llm_model,
                    "temperature": self.llm_temperature,
                    "max_tokens": self.llm_max_tokens,
                    "ollama_base_url": self.ollama_base_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self.embedder_model,
                    "ollama_base_url": self.embedder_base_url,
                },
            },
        }
