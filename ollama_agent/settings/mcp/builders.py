"""Factory functions for creating MCP server instances."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents import Agent
from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

from ...core import ModelCapabilityError, ensure_model_supports_tools
from .types import DEFAULT_AGENT_INSTRUCTIONS

logger = logging.getLogger(__name__)


def _get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key from config, falling back to default."""
    for key in keys:
        if key in config:
            return config[key]
    return default


def _extract_common_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Map optional config flags shared across server implementations."""
    mapping = {
        "cache_tools_list": ("cache_tools_list", "cacheToolsList"),
        "client_session_timeout_seconds": (
            "client_session_timeout_seconds",
            "clientSessionTimeoutSeconds",
        ),
        "use_structured_content": ("use_structured_content", "useStructuredContent"),
        "max_retry_attempts": ("max_retry_attempts", "maxRetryAttempts"),
        "retry_backoff_seconds_base": (
            "retry_backoff_seconds_base",
            "retryBackoffSecondsBase",
        ),
    }

    kwargs: dict[str, Any] = {}
    for target, keys in mapping.items():
        value = _get_config_value(config, *keys)
        if value is not None:
            kwargs[target] = value
    return kwargs


def _create_stdio_server(name: str, config: dict[str, Any]) -> Optional[MCPServerStdio]:
    """Create stdio MCP server from config."""
    command = _get_config_value(config, "command")
    if not command:
        return None

    params: dict[str, Any] = {"command": command}
    for key in ("args", "env", "cwd", "encoding", "encoding_error_handler"):
        value = _get_config_value(config, key)
        if value is not None:
            params[key] = value

    return MCPServerStdio(
        name=name,
        params=params,  # type: ignore[arg-type]
        **_extract_common_kwargs(config),
    )


def _create_streamable_http_server(
    name: str, config: dict[str, Any]
) -> Optional[MCPServerStreamableHttp]:
    """Create Streamable HTTP MCP server from config."""
    url = _get_config_value(config, "url", "httpUrl")
    if not url:
        return None

    params: dict[str, Any] = {"url": url}
    for key, aliases in {
        "headers": ("headers",),
        "timeout": ("timeout",),
        "sse_read_timeout": ("sse_read_timeout", "sseReadTimeout"),
        "terminate_on_close": ("terminate_on_close", "terminateOnClose"),
        "httpx_client_factory": ("httpx_client_factory",),
    }.items():
        value = _get_config_value(config, *aliases)
        if value is not None:
            params[key] = value

    return MCPServerStreamableHttp(
        name=name,
        params=params,  # type: ignore[arg-type]
        **_extract_common_kwargs(config),
    )


def _create_sse_server(name: str, config: dict[str, Any]) -> Optional[MCPServerSse]:
    """Create SSE MCP server from config."""
    url = _get_config_value(config, "url", "httpUrl")
    if not url:
        return None

    params: dict[str, Any] = {"url": url}
    for key, aliases in {
        "headers": ("headers",),
        "timeout": ("timeout",),
        "sse_read_timeout": ("sse_read_timeout", "sseReadTimeout"),
    }.items():
        value = _get_config_value(config, *aliases)
        if value is not None:
            params[key] = value

    return MCPServerSse(
        name=name,
        params=params,  # type: ignore[arg-type]
        **_extract_common_kwargs(config),
    )


def build_server(name: str, config: dict[str, Any]) -> Optional[MCPServer]:
    """Instantiate an MCP server based on the configuration payload.

    Args:
        name: Server name for logging and identification.
        config: Configuration dictionary with transport and connection details.

    Returns:
        An MCPServer instance or None if transport is unsupported.
    """
    transport = _get_config_value(config, "type", "transport")
    if isinstance(transport, str):
        transport = transport.lower()

    # Auto-detect transport from config keys
    if not transport:
        if _get_config_value(config, "command"):
            transport = "stdio"
        elif _get_config_value(config, "httpUrl", "url"):
            transport = "streamable_http"

    if transport in {"stdio", "process"}:
        return _create_stdio_server(name, config)
    if transport in {"sse", "http_sse"}:
        return _create_sse_server(name, config)
    if transport in {"streamable_http", "http", "streamable"}:
        return _create_streamable_http_server(name, config)

    logger.warning("Unsupported MCP server transport '%s' for '%s'", transport, name)
    return None


def build_mcp_agent(
    name: str,
    server: MCPServer,
    config: dict[str, Any],
    default_model: Optional[str],
) -> Optional[tuple[Agent, str, str]]:
    """Build an Agent instance to delegate to the MCP server.

    Args:
        name: Server name.
        server: The connected MCPServer.
        config: Server configuration dictionary.
        default_model: Fallback model if not specified in config.

    Returns:
        Tuple of (Agent, tool_name, tool_description) or None if creation fails.
    """
    agent_config = config.get("agent", {})
    if not isinstance(agent_config, dict):
        agent_config = {}

    model = agent_config.get("model") or default_model
    if not model:
        logger.error("Skipping MCP server '%s': missing model for agent", name)
        return None

    try:
        ensure_model_supports_tools(str(model))
    except ModelCapabilityError as exc:
        logger.error("Skipping MCP server '%s': %s", name, exc)
        return None

    instructions = agent_config.get("instructions") or DEFAULT_AGENT_INSTRUCTIONS.format(
        name=name
    )
    agent_name = agent_config.get("name") or f"{name}_agent"
    tool_name = agent_config.get("tool_name") or f"use_{name}"
    tool_description = (
        agent_config.get("tool_description")
        or agent_config.get("handoff_description")
        or f"Delegate requests to the '{name}' MCP server"
    )

    agent = Agent(
        name=agent_name,
        model=str(model),
        instructions=str(instructions),
        mcp_servers=[server],
        handoff_description=str(
            agent_config.get("handoff_description") or tool_description
        ),
    )

    return agent, str(tool_name), str(tool_description)
