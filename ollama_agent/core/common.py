"""Shared type definitions and utilities for the application."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

# Reasoning effort types
ReasoningEffortValue = Literal["low", "medium", "high", "xhigh", "disabled", "hide", "enabled"]
ALLOWED_REASONING_EFFORTS: tuple[ReasoningEffortValue, ...] = (
    "low",
    "medium",
    "high",
    "xhigh",
    "disabled",
    "hide",
    "enabled",
)
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "medium"


class RAGToolResult(TypedDict, total=False):
    """Result from RAG operations."""

    success: bool
    context: str
    results: list[dict[str, Any]]
    error: str


def extract_text(content: Any, *, sep: str = " ") -> str:
    """Convert agent payload content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return sep.join(filter(None, (extract_text(c, sep=sep) for c in content))).strip()
    if isinstance(content, dict):
        text_val = content.get("text") or content.get("content")
        return extract_text(text_val, sep=sep) if text_val is not None else ""
    return ""


_WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


def validate_identifier(name: str, label: str = "identifier") -> str:
    """Validate that *name* contains only [A-Za-z0-9_-] and is not a reserved system name."""
    name = name.strip()
    if (
        not name
        or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
        or name.upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            f"Invalid {label}. Use only letters, numbers, '_' and '-' (reserved device names not allowed)."
        )
    return name
