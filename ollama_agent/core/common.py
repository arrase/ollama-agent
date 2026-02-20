"""Shared type definitions and utilities for the application."""

import re
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
        return sep.join(filter(None, (extract_text(c, sep=sep) for c in content))).strip()
    if isinstance(content, dict):
        return extract_text(content.get("text") or content.get("content"), sep=sep)
    return ""


def assistant_text_from_messages(messages: list[Any]) -> str:
    """Best-effort: return the latest assistant/AI textual content.

    With Responses API, intermediate AI messages can contain only function_call
    blocks; those must be ignored (no fallback to raw str()).
    """
    for msg in reversed(messages):
        msg_type = str(getattr(msg, "type", "") or "").lower()
        cls_name = getattr(msg, "__class__", type(msg)).__name__.lower()
        if msg_type == "ai" or "aimessage" in cls_name or cls_name == "ai":
            text = extract_text(getattr(msg, "content", None))
            if text:
                return text
    return ""


def final_text_from_state(state: Any) -> str:
    """Extract a final, user-facing string from a DeepAgents state payload."""
    try:
        messages = state.get("messages") if isinstance(state, dict) else None
        if isinstance(messages, list) and messages:
            text = assistant_text_from_messages(messages)
            if text:
                return text

            # Fallback: preserve prior behavior if we couldn't find assistant text.
            last = messages[-1]
            content = getattr(last, "content", last)
            return extract_text(content) or str(content or "")
    except Exception:
        pass
    return str(state)


def resolve_unique_prefix(prefix: str, candidates: list[str]) -> str | None:
    """Resolve a unique candidate starting with prefix.

    Returns the resolved candidate, or None if there is no unique match.
    """
    p = (prefix or "").strip()
    if not p:
        return None
    matches = [c for c in candidates if c.startswith(p)]
    if len(matches) == 1:
        return matches[0]
    return None


def validate_identifier(name: str, label: str = "identifier") -> str:
    """Validate that *name* contains only [A-Za-z0-9_-]. Raises ValueError otherwise."""
    name = (name or "").strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid {label}. Use only letters, numbers, '_' and '-'.")
    return name



