"""MCP servers configuration and lifecycle helpers."""

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncContextManager, Awaitable, Callable, Optional, cast

from agents import Agent
from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

from ..utils import ModelCapabilityError, ensure_model_supports_tools

logger = logging.getLogger(__name__)
_agents_mcp_logger = logging.getLogger("openai.agents")
_agents_mcp_logger.setLevel(logging.CRITICAL)

DEFAULT_MCP_CONFIG_PATH = Path.home() / ".ollama-agent" / "mcp_servers.json"
_DEFAULT_AGENT_INSTRUCTIONS = (
    "You operate the '{name}' MCP server. Always fulfill the user's request "
    "by invoking the server tools and return their results directly."
)


@dataclass(slots=True)
class RunningMCPServer:
    """Active MCP server bound to its cleanup coroutine."""

    name: str
    server: MCPServer
    _closer: Callable[[], Awaitable[None]]
    agent: Optional[Agent] = None
    tool_name: Optional[str] = None
    tool_description: Optional[str] = None

    async def shutdown(self) -> None:
        """Tear down the server without raising on failure."""
        try:
            await self._closer()
        except asyncio.CancelledError:
            logger.debug(
                "Cancellation while cleaning up MCP server '%s'", self.name)
        except Exception as cleanup_error:  # pragma: no cover - best effort logging
            logger.debug("Error cleaning up MCP server '%s': %s",
                         self.name, cleanup_error)


def _get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key from ``config`` falling back to ``default``."""

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


def _create_streamable_http_server(name: str, config: dict[str, Any]) -> Optional[MCPServerStreamableHttp]:
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


def _build_server(name: str, config: dict[str, Any]) -> Optional[MCPServer]:
    """Instantiate an MCP server based on the configuration payload."""

    transport = _get_config_value(config, "type", "transport")
    if isinstance(transport, str):
        transport = transport.lower()

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

    logger.warning(
        "Unsupported MCP server transport '%s' for '%s'", transport, name)
    return None


def _build_mcp_agent(
    name: str,
    server: MCPServer,
    config: dict[str, Any],
    default_model: Optional[str],
) -> Optional[tuple[Agent, str, str]]:
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

    instructions = agent_config.get("instructions") or _DEFAULT_AGENT_INSTRUCTIONS.format(name=name)
    agent_name = agent_config.get("name") or f"{name}_agent"
    tool_name = agent_config.get("tool_name") or f"use_{name}"
    tool_description = agent_config.get("tool_description") or agent_config.get("handoff_description") or f"Delegate requests to the '{name}' MCP server"

    agent = Agent(
        name=agent_name,
        model=str(model),
        instructions=str(instructions),
        mcp_servers=[server],
        handoff_description=str(agent_config.get("handoff_description") or tool_description),
    )

    return agent, str(tool_name), str(tool_description)


async def initialize_mcp_servers(
    config_path: Optional[Path] = None,
    *,
    default_model: Optional[str] = None,
) -> list[RunningMCPServer]:
    """Initialize MCP servers declared in the JSON config file."""

    config_path = config_path or DEFAULT_MCP_CONFIG_PATH
    if not config_path.exists():
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as parse_error:
        logger.error("Invalid MCP config JSON in %s: %s",
                     config_path, parse_error)
        return []
    except OSError as io_error:
        logger.error("Unable to read MCP config at %s: %s",
                     config_path, io_error)
        return []

    servers_payload = data.get("mcpServers")
    if not isinstance(servers_payload, dict):
        logger.warning("No 'mcpServers' mapping found in %s", config_path)
        return []

    running_servers: list[RunningMCPServer] = []

    for name, raw_config in servers_payload.items():
        if not isinstance(raw_config, dict):
            logger.warning("Skipping MCP server '%s': expected object, got %s", name, type(
                raw_config).__name__)
            continue

        server = _build_server(name, raw_config)
        if server is None:
            logger.warning(
                "Skipping MCP server '%s': could not determine transport", name)
            continue

        stack = AsyncExitStack()
        try:
            entered_server = await stack.enter_async_context(
                cast(AsyncContextManager[MCPServer], server)
            )
        except Exception as connect_error:
            await stack.aclose()
            logger.error("Failed to initialize MCP server '%s': %s",
                         name, connect_error)
            continue

        agent_bundle = _build_mcp_agent(
            name,
            entered_server,
            raw_config,
            default_model,
        )
        if not agent_bundle:
            await stack.aclose()
            continue

        agent, tool_name, tool_description = agent_bundle

        running_servers.append(
            RunningMCPServer(
                name=name,
                server=entered_server,
                _closer=stack.aclose,
                agent=agent,
                tool_name=tool_name,
                tool_description=tool_description,
            )
        )
        logger.info("Initialized MCP server: %s", name)

    return running_servers


async def cleanup_mcp_servers(servers: list[RunningMCPServer]) -> None:
    """Cleanup MCP server connections."""

    if not servers:
        return

    for entry in servers:
        await entry.shutdown()
