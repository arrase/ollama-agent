"""Shared type definitions and utilities for the application."""

from typing import Any, Literal, TypedDict

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
    data: dict[str, Any]
    error: str


def extract_text(content: Any) -> str:
    """Best-effort conversion of agent payload content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(filter(None, map(extract_text, content))).strip()
    if isinstance(content, dict):
        return extract_text(content.get("text") or content.get("content"))
    return ""
