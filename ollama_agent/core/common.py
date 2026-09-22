"""Shared type definitions and utilities for the application."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, TypedDict

from ..i18n import _

ReasoningEffortValue = str
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "default"


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
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        return sep.join(filter(None, (extract_text(item, sep=sep) for item in content))).strip()
    if isinstance(content, dict):
        for key in ("text", "content"):
            if key in content:
                return extract_text(content[key], sep=sep)
        raise TypeError("Unsupported dict content for extract_text: missing 'text' or 'content' key")
    raise TypeError(f"Unsupported content shape for extract_text: {type(content).__name__}")


_WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(10)),
        *(f"LPT{i}" for i in range(10)),
    }
)


def validate_identifier(name: str, label: str = "identifier") -> str:
    """Validate that *name* contains only [A-Za-z0-9_-] and is not a reserved system name."""
    name = name.strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name.upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(
            _(
                "Invalid {label}. Use only letters, numbers, '_' and '-' (reserved device names not allowed).",
                label=label,
            )
        )
    return name


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text to *path* atomically using a temporary file in path.parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tmp_path = Path(tf.name)
            tf.write(text)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
