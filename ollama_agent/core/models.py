"""Model capabilities, runtime creation, and validation logic."""

from __future__ import annotations

import re
from typing import Any, Callable, cast

import ollama
from langchain_ollama import ChatOllama
from pydantic import Field

from ..i18n import _
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
            _("Failed to fetch metadata for '{model}': {exc}", model=model, exc=exc)
        ) from exc


def _parse_modelfile_param(text: str | None, param_name: str) -> str | None:
    if not text:
        return None
    pattern = rf"^\s*(?:PARAMETER\s+)?{re.escape(param_name)}\s+([^\s\n]+)"
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1) if match else None


def _parse_num_ctx(text: str | None) -> int | None:
    val = _parse_modelfile_param(text, "num_ctx")
    return int(val) if val and val.isdigit() else None


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
        raise ModelCapabilityError(_("Model '{model}' does not support tools.", model=model))


async def model_supports_thinking(model: str, base_url: str) -> bool:
    """Detection of Ollama thinking support for a model."""
    return "thinking" in await get_model_capabilities(model, base_url)


async def resolve_context_window(
    model: str,
    context_window: int | None,
    base_url: str,
) -> int:
    """Resolve the effective context window for a model."""
    if context_window is not None:
        if context_window <= 0:
            raise ModelContextWindowError(_("context_window must be greater than zero."))
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
        _("Failed to determine the context window for '{model}'. Define context_window in the settings or config file.", model=model)
    )


async def resolve_ollama_reasoning(
    model: str,
    effort: ReasoningEffortValue,
    base_url: str,
    warn_callback: Callable[[str], None] = lambda _msg: None,
) -> bool | str | None:
    """Translate reasoning_effort to Ollama's native reasoning setting."""
    lower_name = model.lower()
    if "qwen3.8" in lower_name:
        if effort == "disabled":
            return False
        if effort == "hide":
            return True
        if effort == "enabled":
            return "xhigh"
        return effort

    if "gpt-oss" in lower_name:
        if effort == "disabled":
            warn_callback(
                _("Model '{model}' is a thinking-only model. reasoning_effort='disabled' is not supported; thinking will remain enabled.", model=model)
            )
            return None
        if effort == "hide":
            return None
        if effort == "enabled":
            return DEFAULT_REASONING_EFFORT
        return effort

    if not await model_supports_thinking(model, base_url):
        return None

    if effort in ("hide", "enabled"):
        return True
    return effort != "disabled"


OLLAMA_PARAM_DEFAULTS: dict[str, Any] = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.1,
}


async def resolve_model_parameters(
    model: str,
    base_url: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repeat_penalty: float | None = None,
) -> dict[str, tuple[Any, str]]:
    """Resolve model sampling parameters with precedence: User > Modelfile > Ollama Default."""
    user_inputs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repeat_penalty": repeat_penalty,
    }

    response = await _show_model(model, base_url)
    meta_sources = [
        getattr(response, "parameters", None),
        getattr(response, "modelfile", None),
    ]

    resolved: dict[str, tuple[Any, str]] = {}

    for param, user_val in user_inputs.items():
        is_int = param == "top_k"
        if user_val is not None:
            resolved[param] = (int(user_val) if is_int else float(user_val), "user")
            continue

        found_val: Any = None
        for text in meta_sources:
            raw = _parse_modelfile_param(text, param)
            if raw is None and param == "repeat_penalty":
                raw = _parse_modelfile_param(text, "repetition_penalty")
            if raw is not None:
                try:
                    found_val = int(raw) if is_int else float(raw)
                    break
                except ValueError:
                    pass

        if found_val is not None:
            resolved[param] = (found_val, "modelfile")
        else:
            resolved[param] = (OLLAMA_PARAM_DEFAULTS[param], "default")

    return resolved


class OllamaChatModel(ChatOllama):
    """ChatOllama model with support for extended Ollama options and parameter tracking."""

    min_p: float | None = None
    presence_penalty: float | None = None
    effective_params: dict[str, tuple[Any, str]] = Field(default_factory=dict)

    def _chat_params(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params = super()._chat_params(messages, stop=stop, **kwargs)
        options = params.setdefault("options", {})
        if self.min_p is not None:
            options["min_p"] = self.min_p
        if self.presence_penalty is not None:
            options["presence_penalty"] = self.presence_penalty
        return params


async def create_ollama_chat_model(
    *,
    model: str,
    base_url: str,
    context_window: int | None,
    reasoning_effort: ReasoningEffortValue,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repeat_penalty: float | None = None,
    warn_callback: Callable[[str], None] = lambda _msg: None,
) -> OllamaChatModel:
    """Create a native ChatOllama model with resolved runtime settings."""
    host = base_url.rstrip("/")
    reasoning = await resolve_ollama_reasoning(
        model, reasoning_effort, host, warn_callback
    )
    num_ctx = await resolve_context_window(model, context_window, host)
    resolved_params = await resolve_model_parameters(
        model,
        host,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repeat_penalty=repeat_penalty,
    )

    kwargs: dict[str, Any] = {
        "base_url": host,
        "model": model,
        "num_ctx": num_ctx,
        "temperature": resolved_params["temperature"][0],
        "top_p": resolved_params["top_p"][0],
        "top_k": resolved_params["top_k"][0],
        "min_p": resolved_params["min_p"][0],
        "presence_penalty": resolved_params["presence_penalty"][0],
        "repeat_penalty": resolved_params["repeat_penalty"][0],
        "profile": {"max_input_tokens": num_ctx},
        "effective_params": resolved_params,
    }
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    return OllamaChatModel(**kwargs)


def validate_reasoning_effort(effort: str) -> ReasoningEffortValue:
    """Validate and normalize reasoning effort value."""
    if effort in ALLOWED_REASONING_EFFORTS:
        return cast(ReasoningEffortValue, effort)
    raise ValueError(
        _("Invalid reasoning effort '{effort}'. Allowed values are: {allowed}", effort=effort, allowed=sorted(ALLOWED_REASONING_EFFORTS))
    )
