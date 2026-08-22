"""Ollama Agent - AI agent to interact with local models."""

__version__ = "0.7.8"

from .agent import AgentRuntime
from .core import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    ModelCapabilityError,
    ReasoningEffortValue,
)
from .settings import Settings, load_settings

__all__ = [
    "ALLOWED_REASONING_EFFORTS",
    "AgentRuntime",
    "DEFAULT_REASONING_EFFORT",
    "ModelCapabilityError",
    "ReasoningEffortValue",
    "Settings",
    "load_settings",
]
