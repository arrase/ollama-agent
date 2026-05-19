"""Model capabilities, runtime creation, and validation logic."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Callable, Iterable, cast

import ollama
from langchain_ollama import ChatOllama

from .common import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    ReasoningEffortValue,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"

THINKING_MODEL_PREFIXES = (
    "deepseek-r1",
    "deepseek-v3.1",
    "gpt-oss",
    "qwen3",
)


class ModelCapabilityError(RuntimeError):
    """Raised when the selected model cannot run tool calls."""


class ModelContextWindowError(RuntimeError):
    """Raised when the context window for a model cannot be resolved."""


def _show_model(model: str, base_url: str | None = None) -> Any:
    """Fetch and cache Ollama model metadata."""
    return _show_model_cached(model, (base_url or DEFAULT_BASE_URL).rstrip("/"))


@lru_cache(maxsize=None)
def _show_model_cached(model: str, host: str) -> Any:
    try:
        return ollama.Client(host=host).show(model)
    except Exception as exc:
        raise ModelCapabilityError(
            f"Failed to fetch metadata for '{model}': {exc}"
        ) from exc


def _response_field(payload: Any, field: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(field, default)
    return getattr(payload, field, default)


def _parse_num_ctx(text: Any) -> int | None:
    if not isinstance(text, str):
        return None
    match = re.search(
        r"^\s*(?:PARAMETER\s+)?num_ctx\s+(\d+)\s*$", text, re.IGNORECASE | re.MULTILINE
    )
    return int(match.group(1)) if match else None


def _model_context_length(model_info: Any) -> int | None:
    if not isinstance(model_info, dict):
        return None

    values = [
        int(v)
        for k, v in model_info.items()
        if str(k).endswith(".context_length") and str(v).isdigit()
    ]
    return max(values, default=None)


def _get_capabilities(model: str, base_url: str | None = None) -> set[str]:
    """Extract capabilities for a model (metadata is cached in _show_model)."""
    response = _show_model(model, base_url)
    payload = _response_field(response, "capabilities", {})
    if isinstance(payload, dict):
        payload = payload.get("capabilities", [])

    if isinstance(payload, Iterable) and not isinstance(payload, str):
        return {str(c).lower() for c in payload if c}

    logger.warning("Model '%s' does not expose capabilities", model)
    return set()


def model_supports_tools(model: str, base_url: str | None = None) -> bool:
    """Check if a model supports tool calls."""
    return "tools" in _get_capabilities(model, base_url)


def ensure_model_supports_tools(model: str, base_url: str | None = None) -> None:
    """Raise ModelCapabilityError if the model doesn't support tools."""
    if not model_supports_tools(model, base_url):
        raise ModelCapabilityError(f"Model '{model}' does not support tools.")


def model_supports_thinking(model: str, base_url: str | None = None) -> bool:
    """Best-effort detection of Ollama thinking support for a model."""
    capabilities = _get_capabilities(model, base_url)
    if "thinking" in capabilities:
        return True

    model_name = model.lower()
    return any(model_name.startswith(prefix) for prefix in THINKING_MODEL_PREFIXES)


def resolve_context_window(
    model: str,
    context_window: int | None,
    base_url: str | None = None,
) -> int:
    """Resolve the effective context window for a model."""
    if context_window is not None:
        if context_window <= 0:
            raise ModelContextWindowError("context_window must be greater than zero.")
        return context_window

    response = _show_model(model, base_url)

    # 1. Structured info is the most reliable (modern Ollama)
    model_info = _response_field(
        response, "model_info", _response_field(response, "modelinfo", {})
    )
    if resolved := _model_context_length(model_info):
        return resolved

    # 2. Fallback to parameters or modelfile regex
    for field_name in ("parameters", "modelfile"):
        if resolved := _parse_num_ctx(_response_field(response, field_name)):
            return resolved

    raise ModelContextWindowError(
        f"Failed to determine the context window for '{model}'. "
        "Define context_window in the settings or config file."
    )


def resolve_ollama_reasoning(
    model: str,
    effort: ReasoningEffortValue,
    base_url: str | None = None,
    warn_callback: Callable[[str], None] | None = None,
) -> bool | str | None:
    """Translate reasoning_effort to Ollama's native reasoning setting."""
    lower_name = model.lower()
    if lower_name.startswith("gpt-oss"):
        if effort == "disabled":
            warning = (
                "GPT-OSS does not support disabling thinking completely in Ollama; "
                "continuing with the model default thinking behavior."
            )
            if warn_callback is not None:
                warn_callback(warning)
            return None
        return effort

    if not model_supports_thinking(model, base_url):
        return None
    return effort != "disabled"


def create_ollama_chat_model(
    *,
    model: str,
    base_url: str | None,
    api_key: str | None = None,
    context_window: int | None,
    reasoning_effort: ReasoningEffortValue,
    temperature: float = 0,
    warn_callback: Callable[[str], None] | None = None,
) -> ChatOllama:
    """Create a native ChatOllama model with resolved runtime settings."""
    host = (base_url or DEFAULT_BASE_URL).rstrip("/")
    reasoning = resolve_ollama_reasoning(model, reasoning_effort, host, warn_callback)
    num_ctx = resolve_context_window(model, context_window, host)

    kwargs: dict[str, Any] = {
        "base_url": host,
        "model": model,
        "num_ctx": num_ctx,
        "temperature": temperature,
        "profile": {"max_input_tokens": num_ctx},
    }
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    return ChatOllama(**kwargs)


def validate_reasoning_effort(effort: str) -> ReasoningEffortValue:
    """Validate and normalize reasoning effort value."""
    if effort in ALLOWED_REASONING_EFFORTS:
        return cast(ReasoningEffortValue, effort)
    logger.warning(
        "Invalid reasoning effort '%s', using '%s'", effort, DEFAULT_REASONING_EFFORT
    )
    return DEFAULT_REASONING_EFFORT
