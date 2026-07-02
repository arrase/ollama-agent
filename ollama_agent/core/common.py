"""Shared type definitions and utilities for the application."""

import re
from typing import Any, Literal, TypedDict

# Reasoning effort types
ReasoningEffortValue = Literal["low", "medium", "high", "disabled", "hide", "enabled"]
ALLOWED_REASONING_EFFORTS: tuple[ReasoningEffortValue, ...] = (
    "low",
    "medium",
    "high",
    "disabled",
    "hide",
    "enabled",
)
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "medium"


class CommandResult(TypedDict):
    """Result from executing a shell command."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int


class RAGToolResult(TypedDict, total=False):
    """Result from RAG operations."""

    success: bool
    context: str
    results: list[dict[str, Any]]
    error: str


def extract_text(content: Any, *, sep: str = " ") -> str:
    """Best-effort conversion of agent payload content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return sep.join(
            filter(None, (extract_text(c, sep=sep) for c in content))
        ).strip()
    if isinstance(content, dict):
        return extract_text(content.get("text") or content.get("content"), sep=sep)
    return ""


def assistant_text_from_messages(messages: list[Any]) -> str:
    """Best-effort: return the latest assistant/AI textual content.

    With Responses API, intermediate AI messages can contain only function_call
    blocks; those must be ignored (no fallback to raw str()).
    """
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai":
            return extract_text(getattr(msg, "content", None)) or ""
    return ""


def final_text_from_state(state: dict[str, Any]) -> str:
    """Extract a final, user-facing string from a DeepAgents state payload."""
    messages = state.get("messages")
    if messages:
        text = assistant_text_from_messages(messages)
        if text:
            return text

        # Fallback: preserve prior behavior if we couldn't find assistant text.
        last = messages[-1]
        content = getattr(last, "content", last)
        return extract_text(content) or str(content)
    return str(state)


def validate_identifier(name: str, label: str = "identifier") -> str:
    """Validate that *name* contains only [A-Za-z0-9_-]. Raises ValueError otherwise."""
    name = name.strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid {label}. Use only letters, numbers, '_' and '-'.")
    return name
