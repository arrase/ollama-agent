"""Utility helpers shared across the application."""

from typing import Any


def extract_text(content: Any) -> str:
    """Best-effort conversion of agent payload content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(filter(None, (extract_text(item) for item in content))).strip()
    if isinstance(content, dict):
        return extract_text(content.get("text") or content.get("content"))
    return ""
