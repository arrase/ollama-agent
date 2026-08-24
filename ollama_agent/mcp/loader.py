"""MCP server initialization and loading routines."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..settings import MCP_PATH, MCP_SERVERS_PATH, SubAgentMCPServer

_log = logging.getLogger(__name__)
_ENV_RE = re.compile(r"\$\{([^}]+)\}|%([^%]+)%")


def get_mcp_config_path() -> Path:
    """Return the active MCP config path (mcp.json if present, else mcp_servers.json or mcp.json)."""
    if MCP_PATH.exists():
        return MCP_PATH
    if MCP_SERVERS_PATH.exists():
        return MCP_SERVERS_PATH
    return MCP_PATH


def _resolve_env(env: dict[str, str]) -> dict[str, str] | None:
    """Resolve ``${VAR}`` and ``%VAR%`` patterns against ``os.environ``.

    Returns the resolved dict (possibly empty) on success, or ``None``
    when required environment variables are missing.
    """
    if not env:
        return {}

    def _replace(match: re.Match[str]) -> str:
        return os.environ[match.group(1) or match.group(2)]

    resolved: dict[str, str] = {}
    for key, value in env.items():
        try:
            resolved[key] = _ENV_RE.sub(_replace, value)
        except KeyError as exc:
            _log.warning("Missing environment variable: %s", exc)
            return None
    return resolved


def _build_mcp_connection(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build a MultiServerMCPClient connection dict from a server config."""
    transport = cfg.get("transport") or cfg.get("type")

    if cfg.get("command"):
        out: dict[str, Any] = {
            "transport": "stdio",
            "command": cfg["command"],
            "args": cfg.get("args", []),
        }
        if "cwd" in cfg:
            out["cwd"] = cfg["cwd"]
        if "env" in cfg:
            resolved = _resolve_env(cfg["env"])
            if resolved is None:
                return None
            if resolved:
                out["env"] = resolved
        return out

    url = cfg.get("url") or cfg.get("httpUrl")
    if url:
        conn_transport = (
            transport
            if transport in {"sse", "websocket", "http", "streamable_http", "streamable-http"}
            else "http"
        )
        out = {"transport": conn_transport, "url": url}
        for k in ("headers", "timeout", "sse_read_timeout", "session_kwargs"):
            if k in cfg:
                out[k] = cfg[k]
        return out

    return None


async def load_main_mcp_tools() -> list[Any]:
    """Load flat MCP tools for the main agent from mcp.json / mcp_servers.json."""
    config_path = get_mcp_config_path()
    if not config_path.exists():
        return []

    try:
        raw_json = await asyncio.to_thread(config_path.read_text, encoding="utf-8")
        data = json.loads(raw_json)
    except (json.JSONDecodeError, OSError) as exc:
        _log.error("Failed to load MCP config %s: %s", config_path, exc)
        return []

    servers_cfg = data.get("mcpServers") or data.get("servers") or {}
    if not isinstance(servers_cfg, dict) or not servers_cfg:
        return []

    # Build MultiServerMCPClient connection dict
    connections: dict[str, dict[str, Any]] = {}
    for name, cfg in servers_cfg.items():
        if not isinstance(cfg, dict):
            continue
        conn = _build_mcp_connection(cfg)
        if conn:
            connections[name] = conn
        else:
            _log.warning(
                "Skipping MCP server '%s': could not determine transport", name
            )

    if not connections:
        return []

    async def _load_single_server(name: str, conn: dict[str, Any]) -> list[Any]:
        """Helper to isolate errors and run in parallel."""
        try:
            client = MultiServerMCPClient({name: conn})  # type: ignore[dict-item,arg-type]
            srv_tools = await client.get_tools()
            _log.info(
                "Loaded %d MCP tools from server '%s'",
                len(srv_tools),
                name,
            )
            return srv_tools
        except Exception as exc:
            _log.error("Failed to load tools from MCP server '%s': %s", name, exc)
            return []

    # Execute all loading tasks in parallel
    tasks = [_load_single_server(name, conn) for name, conn in connections.items()]
    results = await asyncio.gather(*tasks)

    # Flatten the list of lists
    tools: list[Any] = []
    for tool_list in results:
        tools.extend(tool_list)

    return tools


async def load_subagent_mcp_tools(
    subagent_name: str,
    mcp_servers: list[SubAgentMCPServer],
) -> list[Any]:
    """Load MCP tools for a subagent's MCP servers."""
    servers: dict[str, dict[str, Any]] = {}
    for srv in mcp_servers:
        env = _resolve_env(srv.env)
        if env is None:
            _log.warning(
                "Subagent '%s': skipping MCP '%s' (unresolved env vars)",
                subagent_name,
                srv.name,
            )
            continue
        entry: dict[str, Any] = {
            "command": srv.command,
            "args": srv.args,
            "transport": "stdio",
        }
        if env:
            entry["env"] = env
        servers[srv.name] = entry

    if not servers:
        return []

    try:
        client = MultiServerMCPClient(servers)  # type: ignore[arg-type]
        tools = await client.get_tools()
        return tools
    except Exception as exc:
        _log.warning("Subagent '%s': MCP tools failed to load: %s", subagent_name, exc)
        return []
