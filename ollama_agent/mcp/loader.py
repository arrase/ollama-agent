"""MCP server initialization and loading routines."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..settings import MCP_SERVERS_PATH, SubAgentMCPServer

_log = logging.getLogger(__name__)
_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env(env: dict[str, str]) -> dict[str, str] | None:
    """Resolve ``${VAR}`` patterns against ``os.environ``.

    Returns the resolved dict (possibly empty) on success, or ``None``
    when required environment variables are missing.
    """
    if not env:
        return {}
    resolved: dict[str, str] = {}
    for key, value in env.items():
        referenced = _ENV_RE.findall(value)
        missing = [var for var in referenced if var not in os.environ]
        if missing:
            _log.warning("Missing environment variables: %s", ", ".join(missing))
            return None
        resolved[key] = _ENV_RE.sub(lambda m: os.environ[m.group(1)], value)
    return resolved


def _build_mcp_connection(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build a MultiServerMCPClient connection dict from a server config."""
    if cfg.get("command"):
        out: dict[str, Any] = {
            "transport": "stdio",
            "command": cfg["command"],
        }
        if "args" in cfg:
            out["args"] = cfg["args"]
        if "env" in cfg:
            resolved = _resolve_env(cfg["env"])
            if resolved is None:
                return None
            if resolved:
                out["env"] = resolved
        return out

    url = cfg.get("url") or cfg.get("httpUrl")
    if url:
        out = {"transport": "http", "url": url}
        for k in ("headers", "timeout", "sse_read_timeout"):
            if k in cfg:
                out[k] = cfg[k]
        return out

    return None


async def load_main_mcp_tools(exit_stack: AsyncExitStack | None = None) -> list[Any]:
    """Load flat MCP tools for the main agent from mcp_servers.json."""
    if not MCP_SERVERS_PATH.exists():
        return []

    try:
        raw_json = await asyncio.to_thread(MCP_SERVERS_PATH.read_text, encoding="utf-8")
        data = json.loads(raw_json)
    except (json.JSONDecodeError, OSError) as exc:
        _log.error("Failed to load MCP config %s: %s", MCP_SERVERS_PATH, exc)
        return []

    servers_cfg = data.get("mcpServers", data.get("servers", {}))
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

    try:
        client = MultiServerMCPClient(connections)  # type: ignore[arg-type]
        if exit_stack is not None:
            await exit_stack.enter_async_context(client)  # type: ignore[arg-type]
        tools = await client.get_tools()
        _log.info(
            "Loaded %d MCP tools from %d servers",
            len(tools),
            len(connections),
        )
        return tools
    except Exception as exc:
        _log.error("Failed to initialize MCP tools: %s", exc)
        return []


async def load_subagent_mcp_tools(
    subagent_name: str,
    mcp_servers: list[SubAgentMCPServer],
    exit_stack: AsyncExitStack | None = None,
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
        if exit_stack is not None:
            await exit_stack.enter_async_context(client)  # type: ignore[arg-type]
        tools = await client.get_tools()
        return tools
    except Exception as exc:
        _log.warning("Subagent '%s': MCP tools failed to load: %s", subagent_name, exc)
        return []
