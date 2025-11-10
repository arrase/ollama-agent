"""Mem0 integration helpers for persistent agent memory."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict, Optional

from mem0 import Memory

from ..settings.configini import Mem0Settings
from .bootstrap import MemoryBootstrapError, ensure_qdrant_service

logger = logging.getLogger(__name__)


class Mem0NotConfiguredError(RuntimeError):
    """Raised when Mem0 is used before the integration is initialized."""


class Mem0InitializationError(RuntimeError):
    """Raised when Mem0 cannot be initialized with the provided settings."""


_memory_lock = Lock()
_active_settings: Optional[Mem0Settings] = None
_memory_instance: Optional[Memory] = None


def configure_mem0(settings: Mem0Settings) -> None:
    """Register Mem0 settings and reset cached state when they change."""
    global _active_settings, _memory_instance

    with _memory_lock:
        if _active_settings == settings and _memory_instance is not None:
            return

    try:
        ensure_qdrant_service(settings)
    except MemoryBootstrapError as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to ensure Qdrant service", exc_info=True)
        raise Mem0InitializationError(str(exc)) from exc

    with _memory_lock:
        if _active_settings != settings:
            _active_settings = settings
            _memory_instance = None
            logger.debug("Mem0 settings updated; memory cache cleared")


def _require_settings() -> Mem0Settings:
    with _memory_lock:
        if _active_settings is None:
            raise Mem0NotConfiguredError("Mem0 integration is not initialized")
        return _active_settings


def _build_config(settings: Mem0Settings) -> Dict[str, Any]:
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": settings.collection_name,
                "host": settings.host,
                "port": settings.port,
                "embedding_model_dims": settings.embedding_model_dims,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": settings.llm_model,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "ollama_base_url": settings.ollama_base_url,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": settings.embedder_model,
                "ollama_base_url": settings.embedder_base_url,
            },
        },
    }


def _ensure_memory_instance(settings: Mem0Settings) -> Memory:
    global _memory_instance
    with _memory_lock:
        if _memory_instance is not None:
            return _memory_instance

        config = _build_config(settings)
        try:
            _memory_instance = Memory.from_config(config)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize Mem0", exc_info=True)
            raise Mem0InitializationError(str(exc)) from exc
        return _memory_instance


def add_memory_entry(memory_text: str) -> Dict[str, Any]:
    """Store a memory string for the configured single user."""
    settings = _require_settings()
    memory = _ensure_memory_instance(settings)
    result = memory.add(memory_text, user_id=settings.user_id)
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"results": result}
    return {"results": [result]}


def search_memories(query: str, *, limit: Optional[int] = None) -> Dict[str, Any]:
    """Search stored memories for the configured single user."""
    settings = _require_settings()
    memory = _ensure_memory_instance(settings)
    kwargs: Dict[str, Any] = {"user_id": settings.user_id}
    if limit is not None:
        kwargs["limit"] = limit
    result = memory.search(query, **kwargs)
    if isinstance(result, dict):
        return result
    return {"results": result}
