"""MCP server initialization and cleanup routines.

Loads MCP servers from a JSON config and exposes each server as a Deep Agents
subagent descriptor.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools

from ...core import ModelCapabilityError, ensure_model_supports_tools
from .types import DEFAULT_AGENT_INSTRUCTIONS, DEFAULT_MCP_CONFIG_PATH, RunningMCPServer

logger = logging.getLogger(__name__)


async def _noop_shutdown() -> None:
    return


def _normalize_mcp_tools_for_compat(tools: list[BaseTool]) -> list[BaseTool]:
    normalized: list[BaseTool] = []
    for tool in tools:
        if tool.name != "tavily_search":
            normalized.append(tool)
            continue

        args_schema = getattr(tool, "args_schema", None)
        tool_coro = getattr(tool, "coroutine", None)
        if not isinstance(args_schema, dict) or not callable(tool_coro):
            normalized.append(tool)
            continue

        patched_schema = deepcopy(args_schema)
        topic_schema = patched_schema.get("properties", {}).get("topic")
        if isinstance(topic_schema, dict) and topic_schema.get("const") == "general":
            topic_schema.pop("const", None)
            topic_schema["enum"] = ["general", "news"]

        async def _tavily_search_compat(_tool_coro=tool_coro, runtime: Any = None, **arguments: Any) -> Any:
            topic = arguments.get("topic")
            if isinstance(topic, str) and topic != "general":
                arguments = {**arguments, "topic": "general"}
            return await _tool_coro(runtime=runtime, **arguments)

        normalized.append(
            StructuredTool(
                name=tool.name,
                description=tool.description,
                args_schema=patched_schema,
                coroutine=_tavily_search_compat,
                response_format=getattr(tool, "response_format", "content"),
                metadata=getattr(tool, "metadata", None),
            )
        )

    return normalized


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


async def _init_server(name: str, config: Any, default_model: str | None) -> RunningMCPServer | None:
    if not isinstance(config, dict):
        logger.warning("Skipping MCP server '%s': invalid config type", name)
        return None

    if not (connection := _build_connection(config)):
        logger.warning("Skipping MCP server '%s': could not determine transport", name)
        return None

    raw_agent_cfg = config.get("agent", {})
    agent_cfg = raw_agent_cfg if isinstance(raw_agent_cfg, dict) else {}
    model = agent_cfg.get("model") or default_model
    if not model:
        logger.error("Skipping MCP server '%s': missing model", name)
        return None

    try:
        ensure_model_supports_tools(str(model))
    except ModelCapabilityError as exc:
        logger.error("Skipping MCP server '%s': %s", name, exc)
        return None

    subagent_name = str(agent_cfg.get("name") or "").strip()
    subagent_description = str(agent_cfg.get("description") or "").strip()
    instructions = str(agent_cfg.get("system_prompt") or "").strip()

    if not subagent_name:
        logger.error("Skipping MCP server '%s': missing required field agent.name", name)
        return None
    if not subagent_description:
        logger.error("Skipping MCP server '%s': missing required field agent.description", name)
        return None
    if not instructions:
        instructions = DEFAULT_AGENT_INSTRUCTIONS.format(name=name)

    try:
        # Use per-call MCP sessions instead of a shared long-lived session.
        # This avoids task-affinity/cancellation issues (AnyIO cancel scope
        # errors) when tools are invoked from nested subagents.
        mcp_tools = await load_mcp_tools(
            None,
            connection=connection,
            server_name=name,
            tool_name_prefix=False,
        )
        mcp_tools = _normalize_mcp_tools_for_compat(mcp_tools)
    except Exception as exc:
        logger.error("Failed to initialize MCP server '%s': %s", name, exc)
        return None

    logger.info("Initialized MCP server: %s", name)
    llm = ChatOpenAI(
        **{
            "model_name": str(model),
            "openai_api_base": str(
                config.get("base_url")
                or config.get("openai_api_base")
                or "http://localhost:11434/v1/"
            ),
            "openai_api_key": str(config.get("api_key") or config.get("openai_api_key") or "ollama"),
            "temperature": 0,
            "use_responses_api": False,
            "streaming": False,
        }
    )
    return RunningMCPServer(
        name=name,
        subagent={
            "name": subagent_name,
            "description": subagent_description,
            "system_prompt": instructions,
            "tools": mcp_tools,
            "model": llm,
        },
        _closer=_noop_shutdown,
    )


async def initialize_mcp_servers(
    config_path: Path | None = None,
    *,
    default_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
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
        if isinstance(cfg, dict):
            if base_url and "base_url" not in cfg and "openai_api_base" not in cfg:
                cfg = {**cfg, "base_url": base_url}
            if api_key and "api_key" not in cfg and "openai_api_key" not in cfg:
                cfg = {**cfg, "api_key": api_key}
        if s := await _init_server(str(name), cfg, default_model):
            servers.append(s)
    return servers


async def cleanup_mcp_servers(servers: list[RunningMCPServer]) -> None:
    for server in servers:
        await server.shutdown()
