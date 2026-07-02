"""Ollama Agent - AI agent to interact with local models."""

__version__ = "0.6.0"

from .agent import AgentRuntime
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
    "Settings",
    "load_settings",
]
