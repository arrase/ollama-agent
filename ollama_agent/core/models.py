"""Model capabilities and validation logic."""

import logging
from functools import lru_cache
from typing import Iterable, cast

import ollama

from .common import ALLOWED_REASONING_EFFORTS, DEFAULT_REASONING_EFFORT, ReasoningEffortValue

logger = logging.getLogger(__name__)


class ModelCapabilityError(RuntimeError):
    """Raised when the selected model cannot run tool calls."""


@lru_cache(maxsize=None)
def _get_capabilities(model: str) -> set[str]:
    """Fetch and cache capabilities for a model."""
    try:
        response = ollama.show(model)
    except Exception as exc:
        raise ModelCapabilityError(
            f"Failed to fetch metadata for '{model}': {exc}") from exc

    payload = getattr(response, "capabilities", {})
    if isinstance(payload, dict):
        payload = payload.get("capabilities", [])

    if isinstance(payload, Iterable) and not isinstance(payload, str):
        return {str(c).lower() for c in payload if c}

    logger.warning("Model '%s' does not expose capabilities", model)
    return set()


def model_supports_tools(model: str) -> bool:
    """Check if a model supports tool calls."""
    return "tools" in _get_capabilities(model)


def ensure_model_supports_tools(model: str) -> None:
    """Raise ModelCapabilityError if the model doesn't support tools."""
    if not model_supports_tools(model):
        raise ModelCapabilityError(f"Model '{model}' does not support tools.")




def validate_reasoning_effort(effort: str) -> ReasoningEffortValue:
    """Validate and normalize reasoning effort value."""
    if effort in ALLOWED_REASONING_EFFORTS:
        return cast(ReasoningEffortValue, effort)
    logger.warning("Invalid reasoning effort '%s', using '%s'",
                   effort, DEFAULT_REASONING_EFFORT)
    return DEFAULT_REASONING_EFFORT
