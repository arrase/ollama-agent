"""Utility helpers shared across the application."""

import logging
from functools import lru_cache
from typing import Any, Iterable, Literal, cast

import ollama

# Type definitions
ReasoningEffortValue = Literal["low", "medium", "high", "disabled"]
ALLOWED_REASONING_EFFORTS: tuple[ReasoningEffortValue, ...] = (
    "low", "medium", "high", "disabled")
DEFAULT_REASONING_EFFORT: ReasoningEffortValue = "medium"

# Configure logging
logger = logging.getLogger(__name__)


class ModelCapabilityError(RuntimeError):
    """Raised when the selected model cannot run tool calls."""


@lru_cache(maxsize=None)
def _capabilities_for_model(model: str) -> set[str]:
    try:
        response = ollama.show(model)
    except Exception as exc:  # noqa: BLE001
        raise ModelCapabilityError(
            f"Failed to fetch metadata for model '{model}': {exc}"
        ) from exc

    payload: Any = getattr(response, "capabilities", None)
    if isinstance(payload, dict):
        payload = payload.get("capabilities")
    if isinstance(payload, str):
        capabilities = {payload.lower()}
    elif isinstance(payload, Iterable):
        capabilities = {str(item).lower() for item in payload}
    else:
        capabilities = set()
    if not capabilities:
        logger.warning(
            "Model '%s' does not expose capabilities in the Ollama response",
            model,
        )
    return capabilities


def model_supports_tools(model: str) -> bool:
    return "tools" in _capabilities_for_model(model)


def ensure_model_supports_tools(model: str) -> None:
    if not model_supports_tools(model):
        raise ModelCapabilityError(
            f"Model '{model}' does not allow tool usage (requires 'tools' capability)."
        )


def get_tool_compatible_models(preferred: str | None = None) -> list[str]:
    try:
        response = ollama.list()
        models = getattr(response, "models", [])
    except Exception as exc:  # noqa: BLE001
        raise ModelCapabilityError(
            f"Failed to list models: {exc}"
        ) from exc

    names: list[str] = []
    seen: set[str] = set()
    for item in models:
        name = getattr(item, "model", None)
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            if model_supports_tools(name):
                names.append(name)
        except ModelCapabilityError as exc:
            logger.warning("Skipping model '%s': %s", name, exc)

    if preferred:
        ensure_model_supports_tools(preferred)
        if preferred not in names:
            names.insert(0, preferred)

    return names


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
