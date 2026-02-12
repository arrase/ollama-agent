"""MCP server initialization and cleanup routines.

Loads MCP servers from a JSON config and exposes each server as a single
delegation tool (use_<name> by default), preserving the prior UX.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, cast

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from ...core import ModelCapabilityError, ensure_model_supports_tools, extract_text
from .types import DEFAULT_AGENT_INSTRUCTIONS, DEFAULT_MCP_CONFIG_PATH, RunningMCPServer

logger = logging.getLogger(__name__)


def _get(cfg: dict[str, Any], *keys: str) -> Any:
    return next((cfg[k] for k in keys if k in cfg), None)


def _infer_transport(cfg: dict[str, Any]) -> str:
    t = (_get(cfg, "type", "transport") or "")
    if isinstance(t, str) and t:
        return t.lower()
    if cfg.get("command"):
        return "stdio"
    if _get(cfg, "httpUrl", "url"):
        return "http"
    return ""


def _build_connection(cfg: dict[str, Any]) -> dict[str, Any] | None:
    transport = _infer_transport(cfg)
    if transport in ("process", "stdio"):
        command = cfg.get("command")
        if not command:
            return None
        out: dict[str, Any] = {"transport": "stdio", "command": command}
        if "args" in cfg:
            out["args"] = cfg["args"]
        for k in ("env", "cwd", "encoding", "encoding_error_handler", "session_kwargs"):
            if k in cfg:
                out[k] = cfg[k]
        return out

    if transport in ("http", "streamable_http", "streamable", "sse", "http_sse"):
        url = _get(cfg, "url", "httpUrl")
        if not url:
            return None
        out = {"transport": "http", "url": url}
        for k in ("headers", "timeout", "sse_read_timeout", "terminate_on_close", "session_kwargs", "auth"):
            if k in cfg:
                out[k] = cfg[k]
        return out

    return None


def _final_text_from_state(state: Any) -> str:
    try:
        messages = state.get("messages") if isinstance(state, dict) else None
        if messages:
            last = messages[-1]
            content = getattr(last, "content", last)
            return extract_text(content) or str(content)
    except Exception:
        pass
    return str(state)


async def _init_server(name: str, config: Any, default_model: str | None) -> RunningMCPServer | None:
    if not isinstance(config, dict):
        logger.warning("Skipping MCP server '%s': invalid config type", name)
        return None

    if not (connection := _build_connection(config)):
        logger.warning("Skipping MCP server '%s': could not determine transport", name)
        return None

    agent_cfg = cast(dict[str, Any], config.get("agent", {}) or {})
    model = agent_cfg.get("model") or default_model
    if not model:
        logger.error("Skipping MCP server '%s': missing model", name)
        return None

    try:
        ensure_model_supports_tools(str(model))
    except ModelCapabilityError as exc:
        logger.error("Skipping MCP server '%s': %s", name, exc)
        return None

    tool_name = str(agent_cfg.get("tool_name") or f"use_{name}")
    tool_description = str(
        agent_cfg.get("tool_description")
        or agent_cfg.get("handoff_description")
        or f"Delegate requests to the '{name}' MCP server"
    )
    instructions = str(agent_cfg.get("instructions") or DEFAULT_AGENT_INSTRUCTIONS.format(name=name))

    client = MultiServerMCPClient({name: connection})
    stack = AsyncExitStack()
    try:
        session = await stack.enter_async_context(client.session(name))
        mcp_tools = await load_mcp_tools(session, server_name=name, tool_name_prefix=False)
    except Exception as exc:
        await stack.aclose()
        logger.error("Failed to initialize MCP server '%s': %s", name, exc)
        return None

    llm = ChatOllama(model=str(model), temperature=0)
    delegated_agent = create_deep_agent(model=llm, tools=mcp_tools, system_prompt=instructions)

    @tool(tool_name, description=tool_description)
    async def delegate(prompt: str) -> str:
        state = await delegated_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        return _final_text_from_state(state)

    logger.info("Initialized MCP server: %s", name)
    return RunningMCPServer(
        name=name,
        delegate_tool=delegate,
        _closer=stack.aclose,
        tool_name=tool_name,
        tool_description=tool_description,
    )


async def initialize_mcp_servers(
    config_path: Path | None = None, *, default_model: str | None = None
) -> list[RunningMCPServer]:
    path = config_path or DEFAULT_MCP_CONFIG_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load MCP config %s: %s", path, exc)
        return []

    servers_cfg = data.get("mcpServers", {})
    if not isinstance(servers_cfg, dict):
        logger.warning("Invalid 'mcpServers' in config")
        return []

    servers: list[RunningMCPServer] = []
    for name, cfg in servers_cfg.items():
        if s := await _init_server(str(name), cfg, default_model):
            servers.append(s)
    return servers


async def cleanup_mcp_servers(servers: list[RunningMCPServer]) -> None:
    for server in servers:
        await server.shutdown()
