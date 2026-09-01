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


def _parse_modelfile_param(text: str, param_name: str) -> str | None:
    pattern = rf"^\s*(?:PARAMETER\s+)?{re.escape(param_name)}\s+([^\s\n]+)"
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip("\"'") if match else None


def _parse_num_ctx(text: str) -> int | None:
    val = _parse_modelfile_param(text, "num_ctx")
    return int(val) if val and val.isdigit() else None


def _model_context_length(model_info: dict[str, Any]) -> int | None:
    values = [
        int(v)
        for k, v in model_info.items()
        if str(k).endswith("context_length") and str(v).isdigit()
    ]
    return max(values, default=None)


async def get_model_capabilities(
    model: str,
    base_url: str,
    *,
    show_info: Any | None = None,
) -> set[str]:
    """Extract capabilities for a model."""
    response = show_info if show_info is not None else await _show_model(model, base_url)
    caps = getattr(response, "capabilities", None)
    if isinstance(caps, dict):
        caps = caps.get("capabilities", [])
    if isinstance(caps, list):
        return {str(c).lower() for c in caps}
    raise ModelCapabilityError(
        _("Unexpected capabilities format for model '{model}': {type}", model=model, type=type(caps).__name__)
    )


async def model_supports_tools(
    model: str,
    base_url: str,
    *,
    show_info: Any | None = None,
) -> bool:
    """Check if a model supports tool calls."""
    return "tools" in await get_model_capabilities(model, base_url, show_info=show_info)


async def ensure_model_supports_tools(
    model: str,
    base_url: str,
    *,
    show_info: Any | None = None,
) -> None:
    """Raise ModelCapabilityError if the model doesn't support tools."""
    if not await model_supports_tools(model, base_url, show_info=show_info):
        raise ModelCapabilityError(_("Model '{model}' does not support tools.", model=model))


async def model_supports_thinking(
    model: str,
    base_url: str,
    *,
    show_info: Any | None = None,
) -> bool:
    """Detection of Ollama thinking support for a model."""
    return "thinking" in await get_model_capabilities(model, base_url, show_info=show_info)


def _get_model_info(response: Any) -> dict[str, Any] | None:
    """Return the model metadata dict from the supported SDK attribute shapes."""
    for attr in ("model_info", "modelinfo"):
        info = getattr(response, attr, None)
        if isinstance(info, dict):
            return info
    return None


async def resolve_context_window(
    model: str,
    context_window: int | str | None,
    base_url: str,
    *,
    show_info: Any | None = None,
) -> int:
    """Resolve the effective context window for a model."""
    if isinstance(context_window, str):
        cleaned = context_window.strip().lower()
        if cleaned == "max":
            context_window = None
        elif cleaned.isdigit():
            context_window = int(cleaned)
        else:
            raise ModelContextWindowError(
                _("Invalid context_window '{value}'. Expected a positive integer or 'max'.", value=context_window)
            )

    if context_window is not None:
        if context_window <= 0:
            raise ModelContextWindowError(_("context_window must be greater than zero."))
        return context_window

    response = show_info if show_info is not None else await _show_model(model, base_url)

    model_info = _get_model_info(response)
    if model_info is not None and (resolved := _model_context_length(model_info)):
        return resolved

    meta_sources = [
        getattr(response, "parameters", None),
        getattr(response, "modelfile", None),
    ]
    for text in [t for t in meta_sources if t]:
        if resolved := _parse_num_ctx(text):
            return resolved

    raise ModelContextWindowError(
        _("Failed to determine the context window for '{model}'. Define context_window in the settings or config file.", model=model)
    )


async def resolve_ollama_reasoning(
    model: str,
    effort: ReasoningEffortValue,
    base_url: str,
    warn_callback: Callable[[str], None],
    *,
    show_info: Any | None = None,
) -> bool | str | None:
    """Translate reasoning_effort to Ollama's native reasoning setting."""
    lower_name = model.lower()
    if "qwen3.8" in lower_name:
        if effort == "disabled":
            return False
        if effort == "hide":
            return True
        if effort in ("xhigh", "high", "enabled"):
            return "high"
        return effort

    if "gpt-oss" in lower_name:
        if effort == "disabled":
            warn_callback(
                _("Model '{model}' is a thinking-only model. reasoning_effort='disabled' is not supported; thinking will remain enabled.", model=model)
            )
            return True
        if effort in ("hide", "enabled"):
            return True
        if effort == "xhigh":
            return "high"
        return effort

    if not await model_supports_thinking(model, base_url, show_info=show_info):
        return None

    if effort == "disabled":
        return False
    if effort in ("hide", "enabled"):
        return True
    if effort == "xhigh":
        return "high"
    return effort


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
    warn_callback: Callable[[str], None],
    show_info: Any | None = None,
) -> dict[str, tuple[Any, str]]:
    """Resolve model sampling parameters with precedence: User > Modelfile."""
    user_inputs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repeat_penalty": repeat_penalty,
    }

    response = show_info if show_info is not None else await _show_model(model, base_url)
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
        for text in [t for t in meta_sources if t]:
            raw = _parse_modelfile_param(text, param)
            if raw is None and param == "repeat_penalty":
                raw = _parse_modelfile_param(text, "repetition_penalty")
            if raw is not None:
                try:
                    found_val = int(raw) if is_int else float(raw)
                    break
                except ValueError:
                    warn_callback(
                        _("Ignoring invalid value '{raw}' for parameter '{param}'.", raw=raw, param=param)
                    )

        if found_val is not None:
            resolved[param] = (found_val, "modelfile")

    return resolved


class OllamaChatModel(ChatOllama):
    """ChatOllama model with support for extended Ollama options and parameter tracking."""

    min_p: float | None = None
    presence_penalty: float | None = None
    effective_params: dict[str, tuple[Any, str]] = Field(default_factory=dict)
    show_info: Any = Field(default=None, exclude=True)

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
    context_window: int | str | None,
    reasoning_effort: ReasoningEffortValue,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repeat_penalty: float | None = None,
    warn_callback: Callable[[str], None],
) -> OllamaChatModel:
    """Create a native ChatOllama model with resolved runtime settings."""
    host = base_url.rstrip("/")
    show_info = await _show_model(model, host)
    reasoning = await resolve_ollama_reasoning(
        model, reasoning_effort, host, warn_callback, show_info=show_info
    )
    num_ctx = await resolve_context_window(
        model, context_window, host, show_info=show_info
    )
    resolved_params = await resolve_model_parameters(
        model,
        host,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repeat_penalty=repeat_penalty,
        warn_callback=warn_callback,
        show_info=show_info,
    )

    kwargs: dict[str, Any] = {
        "base_url": host,
        "model": model,
        "num_ctx": num_ctx,
        "profile": {"max_input_tokens": num_ctx},
        "effective_params": resolved_params,
        "show_info": show_info,
    }
    for param, (val, _source) in resolved_params.items():
        kwargs[param] = val
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    return OllamaChatModel(**kwargs)


def validate_reasoning_effort(effort: str) -> ReasoningEffortValue:
    """Validate and normalize reasoning effort value."""
    normalized = effort.strip().lower()
    if normalized in ALLOWED_REASONING_EFFORTS:
        return cast(ReasoningEffortValue, normalized)
    raise ValueError(
        _("Invalid reasoning effort '{effort}'. Allowed values are: {allowed}", effort=effort, allowed=sorted(ALLOWED_REASONING_EFFORTS))
    )
