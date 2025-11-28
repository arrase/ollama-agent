"""Ollama Agent - AI agent to interact with local models."""

__version__ = "0.1.0"

from .agent import OllamaAgent, create_agent
from .core import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    CommandResult,
    Mem0ToolResult,
    ModelCapabilityError,
    ReasoningEffortValue,
)
from .memory import Mem0Settings, MemoryManager
from .settings import Config, get_config

__all__ = [
    "ALLOWED_REASONING_EFFORTS",
    "CommandResult",
    "Config",
    "DEFAULT_REASONING_EFFORT",
    "Mem0Settings",
    "Mem0ToolResult",
    "MemoryManager",
    "ModelCapabilityError",
    "OllamaAgent",
    "ReasoningEffortValue",
    "create_agent",
    "get_config",
]
