"""MCP server initialization and loading routines."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
from typing import Any

import langchain_mcp_adapters.sessions
from langchain_mcp_adapters.client import MultiServerMCPClient
import mcp.client.stdio

from ..i18n import _
from ..settings import MCP_LOG_PATH, MCP_PATH, SubAgentMCPServer

_log = logging.getLogger(__name__)
_ENV_RE = re.compile(r"\$\{([^}]+)\}|%([^%]+)%")
_KNOWN_TRANSPORTS = {"sse", "websocket", "http", "streamable_http", "streamable-http"}

_orig_stdio_client = mcp.client.stdio.stdio_client


@contextlib.asynccontextmanager
async def _mcp_stdio_client(server: Any, errlog: Any = None) -> Any:
    """Redirect stdio MCP server stderr to mcp.log to keep the TUI pristine."""
    if errlog is not None and errlog is not sys.stderr:
        async with _orig_stdio_client(server, errlog=errlog) as streams:
            yield streams
    else:
        MCP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MCP_LOG_PATH.open("a", encoding="utf-8") as f:
            async with _orig_stdio_client(server, errlog=f) as streams:
                yield streams


mcp.client.stdio.stdio_client = _mcp_stdio_client
langchain_mcp_adapters.sessions.stdio_client = _mcp_stdio_client


class MCPConfigError(RuntimeError):
    """Raised when the MCP configuration is unreadable or invalid."""


def _resolve_env(env: dict[str, str], server_name: str) -> dict[str, str]:
    """Resolve ``${VAR}`` and ``%VAR%`` patterns against ``os.environ``.

    Raises MCPConfigError when a required environment variable is missing.
    """
    if not env:
        return {}

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1) or match.group(2)
        if var not in os.environ:
            raise MCPConfigError(
                _(
                    "MCP server '{name}': missing environment variable '{var}'",
                    name=server_name,
                    var=var,
                )
            )
        return os.environ[var]

    return {key: _ENV_RE.sub(_replace, str(value)) for key, value in env.items()}


def _build_mcp_connection(server_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a MultiServerMCPClient connection dict from a server config.

    Raises MCPConfigError when the config entry is malformed or declares an
    unsupported transport.
    """
    transport = cfg.get("transport") or cfg.get("type")

    if "command" in cfg:
        args = cfg.get("args", [])
        if not isinstance(args, list):
            raise MCPConfigError(
                _("MCP server '{name}': 'args' must be a list", name=server_name)
            )
        out: dict[str, Any] = {
            "transport": "stdio",
            "command": cfg["command"],
            "args": args,
        }
        if "cwd" in cfg:
            out["cwd"] = cfg["cwd"]
        if "env" in cfg:
            env = cfg["env"]
            if not isinstance(env, dict):
                raise MCPConfigError(
                    _("MCP server '{name}': 'env' must be an object", name=server_name)
                )
            resolved = _resolve_env(env, server_name)
            if resolved:
                out["env"] = resolved
        return out

    if "url" in cfg:
        if transport is not None and transport not in _KNOWN_TRANSPORTS:
            raise MCPConfigError(
                _(
                    "MCP server '{name}': unsupported transport '{transport}'",
                    name=server_name,
                    transport=transport,
                )
            )
        out = {"transport": transport if transport is not None else "http", "url": cfg["url"]}
        for k in ("headers", "timeout", "sse_read_timeout", "session_kwargs"):
            if k in cfg:
                out[k] = cfg[k]
        return out

    raise MCPConfigError(
        _("MCP server '{name}': requires either 'command' or 'url'", name=server_name)
    )


async def _read_main_config() -> dict[str, dict[str, Any]]:
    """Read and validate the canonical ``mcp.json`` configuration.

    Returns the ``mcpServers`` mapping (empty when the file does not exist).
    Raises MCPConfigError when the file exists but is unreadable or invalid.
    """
    try:
        raw_json = await asyncio.to_thread(MCP_PATH.read_text, encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise MCPConfigError(
            _("Failed to load MCP config {config_path}: {exc}", config_path=MCP_PATH, exc=exc)
        ) from exc

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise MCPConfigError(
            _("Failed to load MCP config {config_path}: {exc}", config_path=MCP_PATH, exc=exc)
        ) from exc

    if not isinstance(data, dict):
        raise MCPConfigError(
            _("Invalid MCP config {config_path}: expected a JSON object", config_path=MCP_PATH)
        )

    if "mcpServers" not in data:
        return {}

    servers_cfg = data["mcpServers"]
    if not isinstance(servers_cfg, dict):
        raise MCPConfigError(
            _("Invalid MCP config {config_path}: 'mcpServers' must be an object", config_path=MCP_PATH)
        )
    return servers_cfg


async def _connect_and_load(name: str, conn: dict[str, Any]) -> list[Any]:
    """Connect to a single MCP server and return its tools.

    Raises MCPConfigError when the connection or tool loading fails.
    """
    try:
        client = MultiServerMCPClient({name: conn})  # type: ignore[dict-item,arg-type]
        tools = await client.get_tools()
    except Exception as exc:
        raise MCPConfigError(
            _("Failed to load tools from MCP server '{name}': {exc}", name=name, exc=exc)
        ) from exc
    _log.info("Loaded %d MCP tools from server '%s'", len(tools), name)
    return tools


async def _load_tools_from_connections(connections: dict[str, dict[str, Any]]) -> list[Any]:
    """Connect to multiple MCP servers concurrently and return all discovered tools."""
    if not connections:
        return []

    tasks: list[asyncio.Task[list[Any]]] = []
    try:
        async with asyncio.TaskGroup() as tg:
            for name, conn in connections.items():
                tasks.append(tg.create_task(_connect_and_load(name, conn)))
    except ExceptionGroup as eg:
        for exc in eg.exceptions:
            if isinstance(exc, MCPConfigError):
                raise exc from eg
        raise

    return [tool for t in tasks for tool in t.result()]


async def load_main_mcp_tools() -> list[Any]:
    """Load flat MCP tools for the main agent from mcp.json.

    Raises MCPConfigError when the config is unreadable, malformed, or any
    server fails to connect.
    """
    servers_cfg = await _read_main_config()

    connections: dict[str, dict[str, Any]] = {}
    for name, cfg in servers_cfg.items():
        if not isinstance(cfg, dict):
            raise MCPConfigError(
                _("MCP server '{name}': configuration must be an object", name=name)
            )
        connections[name] = _build_mcp_connection(name, cfg)

    return await _load_tools_from_connections(connections)


async def load_subagent_mcp_tools(
    subagent_name: str,
    mcp_servers: list[SubAgentMCPServer],
) -> list[Any]:
    """Load MCP tools for a subagent's MCP servers (one client per server).

    Raises MCPConfigError when any server entry is malformed or fails to connect.
    """
    if not mcp_servers:
        return []

    seen_names: set[str] = set()
    connections: dict[str, dict[str, Any]] = {}
    for srv in mcp_servers:
        name = srv.name.strip()
        if not name:
            raise MCPConfigError(
                _("Subagent '{subagent_name}': MCP server name cannot be empty", subagent_name=subagent_name)
            )
        if name in seen_names:
            raise MCPConfigError(
                _("Subagent '{subagent_name}': duplicate MCP server name '{name}'", subagent_name=subagent_name, name=name)
            )
        seen_names.add(name)
        connections[name] = _build_mcp_connection(name, {"command": srv.command, "args": srv.args, "env": srv.env})

    tools = await _load_tools_from_connections(connections)
    _log.info("Subagent '%s': loaded %d MCP tools", subagent_name, len(tools))
    return tools
