"""Utility helpers shared across the application."""

import logging
from typing import Any, Literal, cast

# Type definitions
ReasoningEffortValue = Literal["low", "medium", "high"]
ALLOWED_REASONING_EFFORTS: tuple[ReasoningEffortValue, ...] = (
    "low", "medium", "high")
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "medium"

# Configure logging
logger = logging.getLogger(__name__)


def validate_reasoning_effort(effort: str) -> ReasoningEffortValue:
    """
    Validate and normalize reasoning effort value.

    Args:
        effort: Effort level string to validate.

    Returns:
        Valid reasoning effort value.
    """
    if effort in ALLOWED_REASONING_EFFORTS:
        return cast(ReasoningEffortValue, effort)
    logger.warning(
        f"Invalid reasoning effort '{effort}', using default '{DEFAULT_REASONING_EFFORT}'")
    return DEFAULT_REASONING_EFFORT


def extract_text(content: Any) -> str:
    """Best-effort conversion of agent payload content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [extract_text(item) for item in content]
        return " ".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value

        if "content" in content:
            return extract_text(content["content"])

    return ""
