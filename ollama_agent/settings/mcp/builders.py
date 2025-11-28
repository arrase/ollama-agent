"""Factory functions for creating MCP server instances."""

from __future__ import annotations

import logging
from typing import Any

from agents import Agent
from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

from ...core import ModelCapabilityError, ensure_model_supports_tools
from .types import DEFAULT_AGENT_INSTRUCTIONS

logger = logging.getLogger(__name__)

# Mapping of config keys to their aliases
_COMMON_KWARGS_MAP = {
    "cache_tools_list": ("cache_tools_list", "cacheToolsList"),
    "client_session_timeout_seconds": (
        "client_session_timeout_seconds",
        "clientSessionTimeoutSeconds",
    ),
    "use_structured_content": ("use_structured_content", "useStructuredContent"),
    "max_retry_attempts": ("max_retry_attempts", "maxRetryAttempts"),
    "retry_backoff_seconds_base": ("retry_backoff_seconds_base", "retryBackoffSecondsBase"),
}

_HTTP_PARAMS_MAP = {
    "headers": ("headers",),
    "timeout": ("timeout",),
    "sse_read_timeout": ("sse_read_timeout", "sseReadTimeout"),
    "terminate_on_close": ("terminate_on_close", "terminateOnClose"),
}


def _get_first(config: dict[str, Any], *keys: str) -> Any:
    """Return the first present key's value from config."""
    for key in keys:
        if key in config:
            return config[key]
    return None


def _extract_kwargs(config: dict[str, Any], mapping: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Extract kwargs from config using a key mapping."""
    return {
        target: value
        for target, keys in mapping.items()
        if (value := _get_first(config, *keys)) is not None
    }


def _create_stdio_server(name: str, config: dict[str, Any]) -> MCPServerStdio | None:
    """Create stdio MCP server from config."""
    command = config.get("command")
    if not command:
        return None

    params = {"command": command}
    for key in ("args", "env", "cwd", "encoding", "encoding_error_handler"):
        if key in config:
            params[key] = config[key]

    return MCPServerStdio(
        name=name,
        params=params,  # type: ignore[arg-type]
        **_extract_kwargs(config, _COMMON_KWARGS_MAP),
    )


def _create_http_server(
    name: str, config: dict[str, Any], use_sse: bool = False
) -> MCPServerSse | MCPServerStreamableHttp | None:
    """Create HTTP-based MCP server (SSE or Streamable HTTP)."""
    url = _get_first(config, "url", "httpUrl")
    if not url:
        return None

    params = {"url": url, **_extract_kwargs(config, _HTTP_PARAMS_MAP)}
    common = _extract_kwargs(config, _COMMON_KWARGS_MAP)

    if use_sse:
        params.pop("terminate_on_close", None)  # Not supported in SSE
        return MCPServerSse(name=name, params=params, **common)  # type: ignore[arg-type]

    return MCPServerStreamableHttp(name=name, params=params, **common)  # type: ignore[arg-type]


def build_server(name: str, config: dict[str, Any]) -> MCPServer | None:
    """Instantiate an MCP server based on configuration."""
    transport = _get_first(config, "type", "transport")
    if isinstance(transport, str):
        transport = transport.lower()

    # Auto-detect transport
    if not transport:
        if config.get("command"):
            transport = "stdio"
        elif _get_first(config, "httpUrl", "url"):
            transport = "streamable_http"

    if transport in {"stdio", "process"}:
        return _create_stdio_server(name, config)
    if transport in {"sse", "http_sse"}:
        return _create_http_server(name, config, use_sse=True)
    if transport in {"streamable_http", "http", "streamable"}:
        return _create_http_server(name, config, use_sse=False)

    logger.warning("Unsupported MCP transport '%s' for '%s'", transport, name)
    return None


def build_mcp_agent(
    name: str,
    server: MCPServer,
    config: dict[str, Any],
    default_model: str | None,
) -> tuple[Agent, str, str] | None:
    """Build an Agent instance to delegate to the MCP server."""
    agent_cfg = config.get("agent", {}) or {}
    model = agent_cfg.get("model") or default_model

    if not model:
        logger.error("Skipping MCP server '%s': missing model", name)
        return None

    try:
        ensure_model_supports_tools(str(model))
    except ModelCapabilityError as e:
        logger.error("Skipping MCP server '%s': %s", name, e)
        return None

    tool_name = agent_cfg.get("tool_name") or f"use_{name}"
    tool_description = (
        agent_cfg.get("tool_description")
        or agent_cfg.get("handoff_description")
        or f"Delegate requests to the '{name}' MCP server"
    )
    instructions = agent_cfg.get("instructions") or DEFAULT_AGENT_INSTRUCTIONS.format(name=name)

    agent = Agent(
        name=agent_cfg.get("name") or f"{name}_agent",
        model=str(model),
        instructions=str(instructions),
        mcp_servers=[server],
        handoff_description=str(agent_cfg.get("handoff_description") or tool_description),
    )

    return agent, str(tool_name), str(tool_description)
