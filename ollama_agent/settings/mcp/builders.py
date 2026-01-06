"""Factory functions for creating MCP server instances."""

from __future__ import annotations

import logging
from typing import Any

from agents import Agent
from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

from ...core import ModelCapabilityError, ensure_model_supports_tools
from .types import DEFAULT_AGENT_INSTRUCTIONS

logger = logging.getLogger(__name__)


def _get(cfg: dict, *keys: str) -> Any:
    """Return first present key's value."""
    return next((cfg[k] for k in keys if k in cfg), None)


def _extract(cfg: dict, mapping: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Extract kwargs from config using key mapping."""
    return {k: v for k, keys in mapping.items() if (v := _get(cfg, *keys)) is not None}


_COMMON_KEYS = {
    "cache_tools_list": ("cache_tools_list", "cacheToolsList"),
    "client_session_timeout_seconds": ("client_session_timeout_seconds", "clientSessionTimeoutSeconds"),
    "use_structured_content": ("use_structured_content", "useStructuredContent"),
    "max_retry_attempts": ("max_retry_attempts", "maxRetryAttempts"),
    "retry_backoff_seconds_base": ("retry_backoff_seconds_base", "retryBackoffSecondsBase"),
}

_HTTP_KEYS = {
    "headers": ("headers",), "timeout": ("timeout",),
    "sse_read_timeout": ("sse_read_timeout", "sseReadTimeout"),
    "terminate_on_close": ("terminate_on_close", "terminateOnClose"),
}


def _create_stdio_server(name: str, config: dict[str, Any]) -> MCPServerStdio | None:
    """Create stdio MCP server from config."""
    if not (command := config.get("command")):
        return None
    params = {"command": command, **{k: config[k] for k in (
        "args", "env", "cwd", "encoding", "encoding_error_handler") if k in config}}
    # type: ignore[arg-type]
    return MCPServerStdio(name=name, params=params, **_extract(config, _COMMON_KEYS))


def _create_http_server(name: str, config: dict[str, Any], use_sse: bool = False) -> MCPServerSse | MCPServerStreamableHttp | None:
    """Create HTTP-based MCP server (SSE or Streamable HTTP)."""
    if not (url := _get(config, "url", "httpUrl")):
        return None
    params, common = {
        "url": url, **_extract(config, _HTTP_KEYS)}, _extract(config, _COMMON_KEYS)
    if use_sse:
        params.pop("terminate_on_close", None)
        # type: ignore[arg-type]
        return MCPServerSse(name=name, params=params, **common)
    # type: ignore[arg-type]
    return MCPServerStreamableHttp(name=name, params=params, **common)


def build_server(name: str, config: dict[str, Any]) -> MCPServer | None:
    """Instantiate an MCP server based on configuration."""
    transport = (_get(config, "type", "transport") or "").lower(
    ) if isinstance(_get(config, "type", "transport"), str) else ""
    if not transport:
        transport = "stdio" if config.get("command") else "streamable_http" if _get(
            config, "httpUrl", "url") else ""

    builders = {
        "stdio": lambda: _create_stdio_server(name, config),
        "process": lambda: _create_stdio_server(name, config),
        "sse": lambda: _create_http_server(name, config, use_sse=True),
        "http_sse": lambda: _create_http_server(name, config, use_sse=True),
        "streamable_http": lambda: _create_http_server(name, config),
        "http": lambda: _create_http_server(name, config),
        "streamable": lambda: _create_http_server(name, config),
    }
    if transport in builders:
        return builders[transport]()
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
    instructions = agent_cfg.get(
        "instructions") or DEFAULT_AGENT_INSTRUCTIONS.format(name=name)

    agent = Agent(
        name=agent_cfg.get("name") or f"{name}_agent",
        model=str(model),
        instructions=str(instructions),
        mcp_servers=[server],
        handoff_description=str(agent_cfg.get(
            "handoff_description") or tool_description),
    )

    return agent, str(tool_name), str(tool_description)
