"""Model capabilities, runtime creation, and validation logic."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import httpx
import ollama
from langchain_ollama import ChatOllama
from ollama import ShowResponse
from packaging.version import parse as parse_version
from pydantic import Field

from ..i18n import _
from .common import (
    ReasoningEffortValue,
)

_log = logging.getLogger(__name__)

MIN_OLLAMA_VERSION = "0.34.3"


class ExtendedShowResponse(ShowResponse):
    """Extended ShowResponse capturing the thinking metadata advertised by Ollama."""

    thinking: Any = None


class ModelCapabilityError(RuntimeError):
    """Raised when the selected model cannot run tool calls."""


class ModelContextWindowError(RuntimeError):
    """Raised when the context window for a model cannot be resolved."""


class OllamaVersionError(ModelCapabilityError):
    """Raised when the Ollama server version is lower than the minimum required version."""


def get_ollama_version(base_url: str) -> str:
    """Fetch the Ollama server version string from the /api/version endpoint."""
    url = f"{base_url.rstrip('/')}/api/version"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return str(data["version"])
    except (httpx.HTTPError, OSError) as exc:
        raise ModelCapabilityError(
            _("Could not connect to Ollama at '{base_url}': {exc}", base_url=base_url, exc=exc)
        ) from exc


def check_ollama_version(
    base_url: str,
    min_version: str = MIN_OLLAMA_VERSION,
) -> str:
    """Check that the Ollama server version meets the minimum requirement.

    Returns the detected Ollama version string if valid.
    Raises OllamaVersionError if the version is lower than min_version.
    """
    version_str = get_ollama_version(base_url)
    if parse_version(version_str) < parse_version(min_version):
        raise OllamaVersionError(
            _(
                "Ollama version {version} is lower than required {min_version}. Please update your Ollama installation.",
                version=version_str,
                min_version=min_version,
            )
        )
    return version_str


async def _show_model(model: str, base_url: str) -> ExtendedShowResponse:
    """Fetch Ollama model metadata asynchronously."""
    host = base_url.rstrip("/")
    try:
        client = ollama.AsyncClient(host=host)
        return await client._request(
            ExtendedShowResponse,
            "POST",
            "/api/show",
            json={"model": model},
        )
    except Exception as exc:
        raise ModelCapabilityError(_("Failed to fetch metadata for '{model}': {exc}", model=model, exc=exc)) from exc


def _parse_modelfile_param(text: str, param_name: str) -> str | None:
    pattern = rf"^\s*(?:PARAMETER\s+)?{re.escape(param_name)}\s+([^\s\n]+)"
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip("\"'") if match else None


def _parse_num_ctx(text: str) -> int | None:
    val = _parse_modelfile_param(text, "num_ctx")
    return int(val) if val and val.isdigit() else None


def _model_context_length(model_info: dict[str, Any]) -> int | None:
    values = [int(v) for k, v in model_info.items() if str(k).endswith("context_length") and str(v).isdigit()]
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
        caps = caps["capabilities"]
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


def get_model_thinking_config(response: Any) -> dict[str, Any] | None:
    """Extract thinking configuration from Ollama show metadata."""
    thinking = response.get("thinking") if isinstance(response, dict) else getattr(response, "thinking", None)
    return thinking if isinstance(thinking, dict) else None


async def model_supports_thinking(
    model: str,
    base_url: str,
    *,
    show_info: Any | None = None,
) -> bool:
    """Detection of Ollama thinking support for a model."""
    response = show_info if show_info is not None else await _show_model(model, base_url)
    if get_model_thinking_config(response) is not None:
        return True
    return "thinking" in await get_model_capabilities(model, base_url, show_info=response)


def _get_model_info(response: Any) -> dict[str, Any] | None:
    """Return the model metadata dict from the supported SDK attribute shapes."""
    for attr in ("model_info", "modelinfo"):
        info = getattr(response, attr, None)
        if isinstance(info, dict):
            return info
    return None


def _get_modelfile_sources(response: Any) -> list[str]:
    """Extract non-empty modelfile/parameters text sources from an Ollama response."""
    sources = [
        getattr(response, "parameters", None),
        getattr(response, "modelfile", None),
    ]
    return [s for s in sources if s]


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

    for text in _get_modelfile_sources(response):
        if resolved := _parse_num_ctx(text):
            return resolved

    raise ModelContextWindowError(
        _(
            "Failed to determine the context window for '{model}'. "
            "Define context_window in the settings or config file.",
            model=model,
        )
    )


def _resolve_thinking_from_cfg(
    thinking_cfg: dict[str, Any],
    effort_clean: str,
    is_default: bool,
    model: str,
    warn_callback: Callable[[str], None],
) -> bool | str | None:
    values: list[Any] = thinking_cfg.get("values") or []
    default: Any = thinking_cfg.get("default")

    if is_default:
        return default

    effort_lower = effort_clean.lower()
    is_off = effort_lower in ("false", "0", "disabled", "off")
    is_on = effort_lower in ("true", "1", "enabled", "on")

    for v in values:
        if v is False and is_off:
            return False
        if v is True and is_on:
            return True
        if not isinstance(v, bool) and str(v).lower() == effort_lower:
            return v

    if is_on:
        return default

    if is_off:
        warn_callback(
            _(
                "Model '{model}' is a thinking-only model. reasoning_effort='disabled' is not supported; "
                "thinking will remain enabled.",
                model=model,
            )
        )
        return default

    warn_callback(
        _(
            "Invalid reasoning effort '{effort}'. Allowed values are: {allowed}",
            effort=effort_clean,
            allowed=", ".join(str(v) for v in values),
        )
    )
    return default


async def resolve_ollama_reasoning(
    model: str,
    effort: Any,
    base_url: str,
    warn_callback: Callable[[str], None],
    *,
    show_info: Any | None = None,
) -> bool | str | None:
    """Translate reasoning_effort to Ollama's native reasoning setting using /api/show metadata."""
    response = show_info if show_info is not None else await _show_model(model, base_url)
    thinking_cfg = get_model_thinking_config(response)

    effort_clean = "" if effort is None else str(effort).strip()
    is_default = not effort_clean or effort_clean.lower() == "default"

    if thinking_cfg is not None:
        return _resolve_thinking_from_cfg(thinking_cfg, effort_clean, is_default, model, warn_callback)

    if not await model_supports_thinking(model, base_url, show_info=response):
        return None

    if is_default:
        return True
    effort_lower = effort_clean.lower()
    if effort_lower in ("false", "0", "disabled", "off"):
        return False
    if effort_lower in ("true", "1", "enabled", "on"):
        return True
    return effort_clean


def _find_param_in_modelfile(
    param: str,
    meta_sources: list[str],
    is_int: bool,
    warn_callback: Callable[[str], None],
) -> Any | None:
    for text in meta_sources:
        raw = _parse_modelfile_param(text, param)
        if raw is None and param == "repeat_penalty":
            raw = _parse_modelfile_param(text, "repetition_penalty")
        if raw is not None:
            try:
                return int(raw) if is_int else float(raw)
            except ValueError:
                warn_callback(_("Ignoring invalid value '{raw}' for parameter '{param}'.", raw=raw, param=param))
    return None


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
    meta_sources = _get_modelfile_sources(response)

    resolved: dict[str, tuple[Any, str]] = {}

    for param, user_val in user_inputs.items():
        is_int = param == "top_k"
        if user_val is not None:
            resolved[param] = (int(user_val) if is_int else float(user_val), "user")
            continue

        found_val = _find_param_in_modelfile(param, meta_sources, is_int, warn_callback)
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
    reasoning = await resolve_ollama_reasoning(model, reasoning_effort, host, warn_callback, show_info=show_info)
    num_ctx = await resolve_context_window(model, context_window, host, show_info=show_info)
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


def validate_reasoning_effort(effort: Any) -> ReasoningEffortValue:
    """Normalize and validate reasoning effort value."""
    if isinstance(effort, bool):
        return "true" if effort else "false"
    normalized = str(effort).strip()
    if not normalized:
        raise ValueError(_("{name} cannot be empty.", name="Reasoning effort"))
    return normalized


def get_model_creation_kwargs(
    model_settings: Any,
    *,
    model: str | None = None,
    context_window: int | str | None = None,
    warn_callback: Callable[[str], None] = _log.warning,
) -> dict[str, Any]:
    """Build kwargs dict for create_ollama_chat_model from ModelSettings."""
    return {
        "model": model or model_settings.name,
        "base_url": model_settings.base_url,
        "context_window": context_window or model_settings.context_window,
        "reasoning_effort": validate_reasoning_effort(model_settings.reasoning_effort),
        "temperature": model_settings.temperature,
        "top_p": model_settings.top_p,
        "top_k": model_settings.top_k,
        "min_p": model_settings.min_p,
        "presence_penalty": model_settings.presence_penalty,
        "repeat_penalty": model_settings.repeat_penalty,
        "warn_callback": warn_callback,
    }
