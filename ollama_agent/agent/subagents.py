"""Build subagent specifications for create_deep_agent."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from ..settings import ModelSettings, SubAgentMCPServer, SubAgentSettings

_log = logging.getLogger(__name__)
_ENV_RE = re.compile(r"\$\{(\w+)\}")


async def build_subagents(
    subagent_settings: list[SubAgentSettings],
    *,
    model_settings: ModelSettings,
    exit_stack: AsyncExitStack,
) -> list[dict[str, Any]]:
    """Convert ``SubAgentSettings`` into dicts for ``create_deep_agent(subagents=...)``."""
    specs: list[dict[str, Any]] = []
    for sa in subagent_settings:
        spec = await _build_spec(
            sa, model_settings=model_settings, exit_stack=exit_stack
        )
        if spec is not None:
            specs.append(spec)
    return specs


async def _build_spec(
    sa: SubAgentSettings,
    *,
    model_settings: ModelSettings,
    exit_stack: AsyncExitStack,
) -> dict[str, Any] | None:
    """Build a single subagent spec dict."""
    if not sa.name:
        _log.warning("Skipping subagent with empty name")
        return None
    if not sa.description:
        _log.warning("Skipping subagent '%s': missing description", sa.name)
        return None

    spec: dict[str, Any] = {
        "name": sa.name,
        "description": sa.description,
        "system_prompt": sa.system_prompt or sa.description,
    }

    if sa.model or sa.context_window:
        name = sa.model or model_settings.name
        num_ctx = sa.context_window or model_settings.context_window
        spec["model"] = ChatOllama(
            model=name,
            base_url=model_settings.base_url,
            num_ctx=num_ctx,
            profile={"max_input_tokens": num_ctx} if num_ctx else {},
        )

    if sa.skills_paths:
        spec["skills"] = [f"/agent/{p.removeprefix('./')}" for p in sa.skills_paths]

    if sa.mcp_servers:
        tools = await _load_mcp_tools(sa.name, sa.mcp_servers, exit_stack)
        if tools:
            spec["tools"] = tools

    return spec


async def _load_mcp_tools(
    subagent_name: str,
    mcp_servers: list[SubAgentMCPServer],
    exit_stack: AsyncExitStack,
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

        async def _cleanup() -> None:
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception:
                _log.warning("MCP cleanup failed for subagent '%s'", subagent_name)

        exit_stack.push_async_callback(lambda: _cleanup())
        return tools
    except Exception as exc:
        _log.warning(
            "Subagent '%s': MCP tools failed to load: %s", subagent_name, exc
        )
        return []


def _resolve_env(env: dict[str, str]) -> dict[str, str] | None:
    """Resolve ``${VAR}`` patterns against ``os.environ``.

    Returns the resolved dict, or ``None`` when any variable is missing.
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
