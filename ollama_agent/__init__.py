"""Ollama Agent package."""

from __future__ import annotations

__version__ = "0.10.4"

from .core import (
    DEFAULT_REASONING_EFFORT,
    ModelCapabilityError,
    ReasoningEffortValue,
)
from .settings import Settings, load_settings

__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "ModelCapabilityError",
    "ReasoningEffortValue",
    "Settings",
    "load_settings",
]
