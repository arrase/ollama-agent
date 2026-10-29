"""Model capabilities, runtime creation, and validation logic."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

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

REASONING_OFF_VALUES = ("false", "0", "disabled", "off")
REASONING_HIDE_VALUES = (*REASONING_OFF_VALUES, "hide")
REASONING_ON_VALUES = ("true", "1", "enabled", "on")

_REASONING_EFFORT_NAME = "Reasoning effort"
_EMPTY_FIELD_MSG = "{name} cannot be empty."

# Reasoning effort is forwarded verbatim to Ollama's "think" parameter and written to
# settings.yaml, so it is restricted to a short opaque token.
_REASONING_EFFORT_RE = re.compile(r"[A-Za-z0-9_.:+-]{1,32}")


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
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise ModelCapabilityError(
            _("Could not connect to Ollama at '{base_url}': {exc}", base_url=base_url, exc=exc)
        ) from exc
    # A 200 with a non-JSON or unexpected body must not escape as KeyError/JSONDecodeError.
    version = data.get("version") if isinstance(data, dict) else None
    if not isinstance(version, str) or not version:
        raise ModelCapabilityError(
            _(
                "Could not connect to Ollama at '{base_url}': {exc}",
                base_url=base_url,
                exc="missing 'version' field in /api/version response",
            )
        )
    return version


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
                "Ollama version {version} is lower than required {min_version}. "
                "Please update your Ollama installation.",
                version=version_str,
                min_version=min_version,
            )
        )
    return version_str


OLLAMA_HTTP_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


def create_ollama_async_client(base_url: str) -> ollama.AsyncClient:
    """Build an AsyncClient with a bounded timeout.

    ``ollama.AsyncClient`` defaults to ``timeout=None``, i.e. an httpx client that will
    wait forever. Callers must also close it (``async with``) to release the pool.
    """
    return ollama.AsyncClient(host=base_url.rstrip("/"), timeout=OLLAMA_HTTP_TIMEOUT)


async def _show_model(model: str, base_url: str) -> ExtendedShowResponse:
    """Fetch Ollama model metadata asynchronously."""
    try:
        async with create_ollama_async_client(base_url) as client:
            res: ExtendedShowResponse = await client._request(  # noqa: SLF001
                ExtendedShowResponse,
                "POST",
                "/api/show",
                json={"model": model},
            )
            return res
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
    show_info: ExtendedShowResponse | None = None,
) -> set[str]:
    """Extract capabilities for a model."""
    response = show_info if show_info is not None else await _show_model(model, base_url)
    caps = response.capabilities
    if isinstance(caps, list):
        return {str(c).lower() for c in caps}
    raise ModelCapabilityError(
        _("Unexpected capabilities format for model '{model}': {type}", model=model, type=type(caps).__name__)
    )


async def model_supports_tools(
    model: str,
    base_url: str,
    *,
    show_info: ExtendedShowResponse | None = None,
) -> bool:
    """Check if a model supports tool calls."""
    return "tools" in await get_model_capabilities(model, base_url, show_info=show_info)


async def ensure_model_supports_tools(
    model: str,
    base_url: str,
    *,
    show_info: ExtendedShowResponse | None = None,
) -> None:
    """Raise ModelCapabilityError if the model doesn't support tools."""
    if not await model_supports_tools(model, base_url, show_info=show_info):
        raise ModelCapabilityError(_("Model '{model}' does not support tools.", model=model))


def get_model_thinking_config(response: ExtendedShowResponse | None) -> dict[str, Any] | None:
    """Extract thinking configuration from Ollama show metadata."""
    if response is None:
        return None
    return response.thinking if isinstance(response.thinking, dict) else None


async def model_supports_thinking(
    model: str,
    base_url: str,
    *,
    show_info: ExtendedShowResponse | None = None,
) -> bool:
    """Detection of Ollama thinking support for a model."""
    response = show_info if show_info is not None else await _show_model(model, base_url)
    if get_model_thinking_config(response) is not None:
        return True
    return "thinking" in await get_model_capabilities(model, base_url, show_info=response)


def _get_model_info(response: ExtendedShowResponse) -> dict[str, Any] | None:
    """Return the model metadata dict from the ShowResponse."""
    return dict(response.modelinfo) if isinstance(response.modelinfo, dict) else None


def _get_modelfile_sources(response: ExtendedShowResponse) -> list[str]:
    """Extract non-empty modelfile/parameters text sources from an Ollama response."""
    return [s for s in (response.parameters, response.modelfile) if s]


def parse_context_window_input(value: int | str) -> int | str:
    """Validate and normalize a context_window value to a positive int or 'max'."""
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned == "max":
            return "max"
        if cleaned.isdigit() and int(cleaned) > 0:
            return int(cleaned)
        raise ModelContextWindowError(
            _("Invalid context_window '{value}'. Expected a positive integer or 'max'.", value=value)
        )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelContextWindowError(_("context_window must be greater than zero."))
    return value


async def resolve_context_window(
    model: str,
    context_window: int | str | None,
    base_url: str,
    *,
    show_info: ExtendedShowResponse | None = None,
) -> int:
    """Resolve the effective context window for a model."""
    if context_window is not None:
        parsed = parse_context_window_input(context_window)
        if isinstance(parsed, int):
            return parsed

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


def format_thinking_value(val: object) -> str:
    """Format an Ollama thinking control value as a normalized string."""
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _match_thinking_effort(
    effort_clean: str,
    thinking_cfg: dict[str, Any],
    model_name: str,
) -> tuple[bool, bool | str | None, str | None]:
    """Match an effort token against Ollama thinking metadata.

    Returns ``(matched, resolved_value, warning_msg)``.
    """
    raw_values = thinking_cfg.get("values")
    values: list[Any] = list(raw_values) if isinstance(raw_values, (list, tuple)) else []
    default: bool | str | None = thinking_cfg.get("default")
    effort_lower = effort_clean.lower()

    if not effort_clean or effort_lower in ("default", "hide"):
        return True, default, None

    is_off = effort_lower in REASONING_OFF_VALUES
    is_on = effort_lower in REASONING_ON_VALUES

    for v in values:
        if (v is False and is_off) or (v is True and is_on):
            return True, v, None
        if not isinstance(v, bool) and str(v).lower() == effort_lower:
            return True, str(v), None

    if is_on:
        return True, default, None

    if is_off:
        return (
            False,
            default,
            _(
                "Model '{model}' is a thinking-only model. reasoning_effort='disabled' is not supported; "
                "thinking will remain enabled.",
                model=model_name,
            ),
        )

    return (
        False,
        default,
        _(
            "Invalid reasoning effort '{effort}'. Allowed values are: {allowed}",
            effort=effort_clean,
            allowed=", ".join(format_thinking_value(v) for v in values),
        ),
    )


def _resolve_thinking_from_cfg(
    thinking_cfg: dict[str, Any],
    effort_clean: str,
    model: str,
    warn_callback: Callable[[str], None],
    *,
    is_default: bool,
) -> bool | str | None:
    if is_default:
        default_val: bool | str | None = thinking_cfg.get("default")
        return default_val
    matched, resolved, msg = _match_thinking_effort(effort_clean, thinking_cfg, model)
    if not matched and msg is not None:
        warn_callback(msg)
    return resolved


async def resolve_ollama_reasoning(
    model: str,
    effort: object,
    base_url: str,
    warn_callback: Callable[[str], None],
    *,
    show_info: ExtendedShowResponse | None = None,
) -> bool | str | None:
    """Translate reasoning_effort to Ollama's native reasoning setting using /api/show metadata."""
    response = show_info if show_info is not None else await _show_model(model, base_url)
    thinking_cfg = get_model_thinking_config(response)

    effort_clean = "" if effort is None else str(effort).strip()
    effort_lower = effort_clean.lower()
    is_default = not effort_clean or effort_lower in ("default", "hide")

    if thinking_cfg is not None:
        return _resolve_thinking_from_cfg(thinking_cfg, effort_clean, model, warn_callback, is_default=is_default)

    if "thinking" not in await get_model_capabilities(model, base_url, show_info=response):
        return None

    if is_default:
        return True
    if effort_lower in REASONING_OFF_VALUES:
        return False
    if effort_lower in REASONING_ON_VALUES:
        return True
    return effort_clean


def _find_param_in_modelfile(
    param: str,
    meta_sources: list[str],
    warn_callback: Callable[[str], None],
    *,
    is_int: bool,
) -> int | float | None:
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
    show_info: ExtendedShowResponse | None = None,
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

        found_val = _find_param_in_modelfile(param, meta_sources, warn_callback, is_int=is_int)
        if found_val is not None:
            resolved[param] = (found_val, "modelfile")

    return resolved


class OllamaChatModel(ChatOllama):
    """ChatOllama model with support for extended Ollama options and parameter tracking."""

    num_ctx: int = 0
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
    warn_callback: Callable[[str], None] = _log.warning,
) -> OllamaChatModel:
    """Create a native ChatOllama model with resolved runtime settings."""
    host = base_url.rstrip("/")
    validated_effort = validate_reasoning_effort(reasoning_effort)
    show_info = await _show_model(model, host)
    reasoning = await resolve_ollama_reasoning(model, validated_effort, host, warn_callback, show_info=show_info)
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


def validate_reasoning_effort(effort: object) -> ReasoningEffortValue:
    """Normalize and validate reasoning effort value.

    The value is forwarded verbatim to Ollama's ``think`` parameter and persisted to
    settings.yaml, so it is restricted to a short opaque token rather than accepted
    as arbitrary free text.
    """
    if isinstance(effort, bool):
        return "true" if effort else "false"
    if not isinstance(effort, str):
        raise ValueError(_(_EMPTY_FIELD_MSG, name=_REASONING_EFFORT_NAME))
    normalized = effort.strip()
    if not normalized:
        raise ValueError(_(_EMPTY_FIELD_MSG, name=_REASONING_EFFORT_NAME))
    if not _REASONING_EFFORT_RE.fullmatch(normalized):
        raise ValueError(
            _(
                "Invalid reasoning effort '{effort}'. Expected a short token such as "
                "'low', 'high' or 'default' (letters, digits, '_', '.', ':', '+' or '-').",
                effort=normalized,
            )
        )
    return normalized


def validate_and_normalize_effort(
    effort: str,
    thinking_cfg: dict[str, Any] | None,
    model_name: str = "",
) -> tuple[str | None, str | None]:
    """Validate and normalize effort against model thinking config.

    Returns ``(normalized_effort, message)`` where ``normalized_effort`` is None when
    rejected and ``message`` is the localized warning/error string.
    """
    norm_effort = effort.strip()
    if not norm_effort:
        return None, _(_EMPTY_FIELD_MSG, name=_REASONING_EFFORT_NAME)
    effort_lower = norm_effort.lower()
    if effort_lower in ("default", "hide"):
        return effort_lower, None
    if not (thinking_cfg and thinking_cfg.get("values")):
        return norm_effort, None

    matched, resolved, msg = _match_thinking_effort(norm_effort, thinking_cfg, model_name)
    if not matched:
        return None, msg
    if resolved is None:
        return "default", None
    return format_thinking_value(resolved), None
