"""Shared type definitions for the application."""

from typing import Any, Dict, Literal, TypedDict

# Reasoning effort types
ReasoningEffortValue = Literal["low", "medium", "high", "disabled"]
ALLOWED_REASONING_EFFORTS: tuple[ReasoningEffortValue, ...] = (
    "low", "medium", "high", "disabled"
)
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "medium"


class CommandResult(TypedDict):
    """Result from executing a shell command."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int


class Mem0ToolResult(TypedDict, total=False):
    """Result from Mem0 operations."""

    success: bool
    data: Dict[str, Any]
    error: str
