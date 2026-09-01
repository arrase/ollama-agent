"""Shared type definitions and utilities for the application."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from ..i18n import _

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
        if text_val is None:
            return ""
        return extract_text(text_val, sep=sep)
    raise TypeError(f"Unsupported content shape for extract_text: {type(content).__name__}")


_WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


def validate_identifier(name: str, label: str | None = None) -> str:
    """Validate that *name* contains only [A-Za-z0-9_-] and is not a reserved system name."""
    name = name.strip()
    if (
        not name
        or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
        or name.upper() in _WINDOWS_RESERVED_NAMES
    ):
        resolved_label = label or _("identifier")
        raise ValueError(
            _("Invalid {label}. Use only letters, numbers, '_' and '-' (reserved device names not allowed).", label=resolved_label)
        )
    return name
