"""Shared type definitions and utilities for the application."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, select_autoescape

from ..i18n import _

ReasoningEffortValue = str
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "default"


def extract_text(content: object) -> str:
    """Convert agent payload content into plain text.

    Non-text multimodal blocks (image/audio/video/file) contribute no plain text, so
    a conversation that attached one must still be readable. Genuinely malformed
    payloads (e.g. a dict with no recognised key) still raise ``TypeError``.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        return " ".join(filter(None, (extract_text(item) for item in content))).strip()
    if isinstance(content, dict):
        for key in ("text", "content"):
            if key in content:
                return extract_text(content[key])
        if "type" in content:
            return ""
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


QUICK_PREVIEW_LIMIT = 60


def shorten(text: str, limit: int = QUICK_PREVIEW_LIMIT) -> str:
    """Collapse newlines and truncate *text* to *limit* characters with an ellipsis."""
    flat = text.replace("\n", " ")
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to *path* atomically using a temporary file in path.parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tmp_path = Path(tf.name)
            tf.write(text)
            tf.flush()
            os.fsync(tf.fileno())
        tmp_path.replace(path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def render_prompt_template(template_str: str, context: dict[str, Any]) -> str:
    """Render a Jinja2 template string with the provided context dictionary."""
    env = Environment(
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.from_string(template_str)
    return template.render(context)
