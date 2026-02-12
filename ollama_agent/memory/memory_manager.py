"""Mem0 integration for persistent agent memory."""

from __future__ import annotations

import logging
from typing import Any

from mem0 import Memory  # type: ignore
from .settings import Mem0Settings

logger = logging.getLogger(__name__)


class Mem0InitializationError(RuntimeError):
    """Raised when Mem0 cannot be initialized."""


class MemoryManager:
    """Manages Mem0 memory instance lifecycle."""

    __slots__ = ("settings", "_memory")

    def __init__(self, settings: Mem0Settings) -> None:
        self.settings = settings
        self._memory: Memory | None = None

    @property
    def memory(self) -> Memory:
        """Lazy-initialize and return Memory instance."""
        if self._memory is None:
            try:
                self._memory = Memory.from_config(self.settings.to_mem0_config())
            except Exception as e:
                logger.error("Mem0 initialization failed: %s", e)
                raise Mem0InitializationError(str(e)) from e
        return self._memory

    def add(self, text: str) -> dict[str, Any]:
        """Store a memory for the configured user."""
        result = self.memory.add(text, user_id=self.settings.user_id)
        return self._normalize_result(result)

    def search(self, query: str, limit: int | None = None) -> dict[str, Any]:
        """Search stored memories."""
        kwargs: dict[str, Any] = {"user_id": self.settings.user_id}
        if limit:
            kwargs["limit"] = limit
        result = self.memory.search(query, **kwargs)
        return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result: Any) -> dict[str, Any]:
        """Normalize Mem0 result to dict format."""
        if isinstance(result, dict):
            return result
        return {"results": result if isinstance(result, list) else [result]}
