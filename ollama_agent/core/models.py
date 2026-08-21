"""Model capabilities, runtime creation, and validation logic."""

from __future__ import annotations

import re
from typing import Any, Callable, cast

import ollama
from langchain_ollama import ChatOllama

from .common import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    ReasoningEffortValue,
)


class ModelCapabilityError(RuntimeError):

    """Raised when the selected model cannot run tool calls."""


class ModelContextWindowError(RuntimeError):
    """Raised when the context window for a model cannot be resolved."""


async def _show_model(model: str, base_url: str) -> Any:
    """Fetch Ollama model metadata asynchronously."""
    host = base_url.rstrip("/")
    try:
        client = ollama.AsyncClient(host=host)
        return await client.show(model)
    except Exception as exc:
        raise ModelCapabilityError(
            f"Failed to fetch metadata for '{model}': {exc}"
        ) from exc


def _parse_num_ctx(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(
        r"^\s*(?:PARAMETER\s+)?num_ctx\s+(\d+)\s*$", text, re.IGNORECASE | re.MULTILINE
    )
    return int(match.group(1)) if match else None


def _model_context_length(model_info: dict[str, Any]) -> int | None:
    values = [
        int(v)
        for k, v in model_info.items()
        if str(k).endswith(".context_length") and str(v).isdigit()
    ]
    return max(values, default=None)


async def get_model_capabilities(model: str, base_url: str) -> set[str]:
    """Extract capabilities for a model."""
    response = await _show_model(model, base_url)
    caps = getattr(response, "capabilities", None)
    if isinstance(caps, dict):
        caps = caps.get("capabilities", [])
    if isinstance(caps, list):
        return {str(c).lower() for c in caps}
    return set()


async def model_supports_tools(model: str, base_url: str) -> bool:
    """Check if a model supports tool calls."""
    return "tools" in await get_model_capabilities(model, base_url)


async def ensure_model_supports_tools(model: str, base_url: str) -> None:
    """Raise ModelCapabilityError if the model doesn't support tools."""
    if not await model_supports_tools(model, base_url):
        raise ModelCapabilityError(f"Model '{model}' does not support tools.")


async def model_supports_thinking(model: str, base_url: str) -> bool:
    """Detection of Ollama thinking support for a model."""
    capabilities = await get_model_capabilities(model, base_url)
    return "thinking" in capabilities


async def resolve_context_window(
    model: str,
    context_window: int | None,
    base_url: str,
) -> int:
    """Resolve the effective context window for a model."""
    if context_window is not None:
        if context_window <= 0:
            raise ModelContextWindowError("context_window must be greater than zero.")
        return context_window

    response = await _show_model(model, base_url)

    # 1. Structured info is the most reliable (modern Ollama)
    model_info = getattr(response, "model_info", None)
    if isinstance(model_info, dict) and (resolved := _model_context_length(model_info)):
        return resolved

    # 2. Fallback to parameters or modelfile regex
    for field_name in ("parameters", "modelfile"):
        if resolved := _parse_num_ctx(getattr(response, field_name, None)):
            return resolved

    raise ModelContextWindowError(
        f"Failed to determine the context window for '{model}'. "
        "Define context_window in the settings or config file."
    )


async def resolve_ollama_reasoning(
    model: str,
    effort: ReasoningEffortValue,
    base_url: str,
    warn_callback: Callable[[str], None] = lambda _: None,
) -> bool | str | None:
    """Translate reasoning_effort to Ollama's native reasoning setting."""
    lower_name = model.lower()
    if "gpt-oss" in lower_name:
        if effort == "disabled":
            warn_callback(
                f"Model '{model}' is a thinking-only model. "
                "reasoning_effort='disabled' is not supported; thinking will remain enabled."
            )
            return None
        if effort == "hide":
            return None
        if effort == "enabled":
            return DEFAULT_REASONING_EFFORT
        if effort == "xhigh":
            return "high"
        return effort

    if bool(re.search(r"qwen[-_.:]?3\.8", lower_name)):
        if effort == "disabled":
            return False
        if effort == "hide":
            return "xhigh"
        if effort in ("high", "xhigh", "enabled"):
            return "xhigh"
        return effort

    caps = await get_model_capabilities(model, base_url)
    if "thinking" not in caps:
        return None

    if effort in ("hide", "enabled"):
        return True
    return effort != "disabled"


async def create_ollama_chat_model(
    *,
    model: str,
    base_url: str,
    context_window: int | None,
    reasoning_effort: ReasoningEffortValue,
    temperature: float = 0,
    warn_callback: Callable[[str], None] = lambda _: None,
) -> ChatOllama:
    """Create a native ChatOllama model with resolved runtime settings."""
    host = base_url.rstrip("/")
    reasoning = await resolve_ollama_reasoning(
        model, reasoning_effort, host, warn_callback
    )
    num_ctx = await resolve_context_window(model, context_window, host)

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
    raise ValueError(
        f"Invalid reasoning effort '{effort}'. Allowed values are: {sorted(ALLOWED_REASONING_EFFORTS)}"
    )
