"""Memory package integrating Mem0 with a Docker-backed Qdrant instance."""

from __future__ import annotations

import logging

from ..settings.configini import Mem0Settings
from .bootstrap import MemoryBootstrapError, ensure_qdrant_service
from .manager import (
    Mem0InitializationError,
    Mem0NotConfiguredError,
    add_memory_entry,
    configure_mem0,
    search_memories,
)

logger = logging.getLogger(__name__)


def bootstrap_memory_backend(settings: Mem0Settings) -> None:
    """Ensure the memory backend is running before the agent starts."""
    try:
        ensure_qdrant_service(settings)
    except MemoryBootstrapError as exc:
        raise Mem0InitializationError(str(exc)) from exc


__all__ = [
    "Mem0InitializationError",
    "Mem0NotConfiguredError",
    "add_memory_entry",
    "bootstrap_memory_backend",
    "configure_mem0",
    "ensure_qdrant_service",
    "search_memories",
]
