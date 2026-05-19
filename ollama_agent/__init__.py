"""Ollama Agent - AI agent to interact with local models."""

__version__ = "0.4.1"

from .agent import AgentRuntime, RuntimeResponse
from .core import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    CommandResult,
    ModelCapabilityError,
    ReasoningEffortValue,
)
from .settings import Settings, load_settings

__all__ = [
    "ALLOWED_REASONING_EFFORTS",
    "AgentRuntime",
    "CommandResult",
    "DEFAULT_REASONING_EFFORT",
    "ModelCapabilityError",
    "ReasoningEffortValue",
    "RuntimeResponse",
    "Settings",
    "load_settings",
]
