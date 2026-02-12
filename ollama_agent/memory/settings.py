"""Mem0 settings configuration."""

from dataclasses import dataclass, field
from typing import Any

from ..settings.paths import MEMORY_DIR


@dataclass(frozen=True, slots=True)
class Mem0Settings:
    """Configuration for Mem0 persistent memory integration."""

    # Qdrant (embedded/local)
    collection_name: str = "ollama-agent"
    qdrant_path: str = field(default_factory=lambda: str(MEMORY_DIR))
    # Always persist Qdrant local storage across runs.
    # This must not be user-configurable because disabling it causes data loss
    # between separate CLI invocations.
    _qdrant_on_disk: bool = field(default=True, init=False, repr=False, compare=False)
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
        qdrant_cfg: dict[str, Any] = {
            "collection_name": self.collection_name,
            "path": self.qdrant_path,
            "on_disk": self._qdrant_on_disk,
            "embedding_model_dims": self.embedding_model_dims,
        }

        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    **qdrant_cfg,
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
