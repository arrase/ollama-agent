"""Mem0 integration for persistent agent memory.

This module provides a clean interface to Mem0 without global singletons.
The MemoryManager is instantiated and owned by the agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mem0 import Memory

from .bootstrap import MemoryBootstrapError, ensure_qdrant_service
from .settings import Mem0Settings

logger = logging.getLogger(__name__)


class Mem0NotConfiguredError(RuntimeError):
    """Raised when Mem0 is used before the integration is initialized."""


class Mem0InitializationError(RuntimeError):
    """Raised when Mem0 cannot be initialized with the provided settings."""


@dataclass
class MemoryManager:
    """Encapsulates Mem0 memory instance and settings.

    This class manages the lifecycle of a Mem0 Memory instance,
    ensuring proper initialization and configuration.

    Attributes:
        settings: Configuration for Mem0.
    """

    settings: Mem0Settings
    _memory: Optional[Memory] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate settings and ensure Qdrant is available."""
        try:
            ensure_qdrant_service(self.settings)
        except MemoryBootstrapError as exc:
            logger.error("Failed to ensure Qdrant service", exc_info=True)
            raise Mem0InitializationError(str(exc)) from exc

    def _build_config(self) -> Dict[str, Any]:
        """Build Mem0 configuration dictionary from settings."""
        s = self.settings
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": s.collection_name,
                    "host": s.host,
                    "port": s.port,
                    "embedding_model_dims": s.embedding_model_dims,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": s.llm_model,
                    "temperature": s.llm_temperature,
                    "max_tokens": s.llm_max_tokens,
                    "ollama_base_url": s.ollama_base_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": s.embedder_model,
                    "ollama_base_url": s.embedder_base_url,
                },
            },
        }

    @property
    def memory(self) -> Memory:
        """Lazy-initialize and return the Memory instance."""
        if self._memory is not None:
            return self._memory

        config = self._build_config()
        try:
            self._memory = Memory.from_config(config)
        except Exception as exc:
            logger.error("Failed to initialize Mem0", exc_info=True)
            raise Mem0InitializationError(str(exc)) from exc
        return self._memory

    def add(self, memory_text: str) -> Dict[str, Any]:
        """Store a memory string for the configured user.

        Args:
            memory_text: The memory content to store.

        Returns:
            Dictionary with the result of the add operation.
        """
        result = self.memory.add(memory_text, user_id=self.settings.user_id)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"results": result}
        return {"results": [result]}

    def search(self, query: str, *, limit: Optional[int] = None) -> Dict[str, Any]:
        """Search stored memories.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            Dictionary with search results.
        """
        kwargs: Dict[str, Any] = {"user_id": self.settings.user_id}
        if limit is not None:
            kwargs["limit"] = limit
        result = self.memory.search(query, **kwargs)
        if isinstance(result, dict):
            return result
        return {"results": result}
