"""MCP server initialization and cleanup routines."""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, AsyncContextManager, cast

from agents.mcp import MCPServer

from .builders import build_mcp_agent, build_server
from .types import DEFAULT_MCP_CONFIG_PATH, RunningMCPServer

logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("openai.agents").setLevel(logging.CRITICAL)


def _load_config(config_path: Path) -> dict[str, Any] | None:
    """Load MCP configuration file."""
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load MCP config %s: %s", config_path, e)
        return None


async def _init_server(
    name: str, config: Any, default_model: str | None
) -> RunningMCPServer | None:
    """Initialize a single MCP server."""
    if not isinstance(config, dict):
        logger.warning("Skipping MCP server '%s': invalid config type", name)
        return None

    server = build_server(name, config)
    if not server:
        logger.warning("Skipping MCP server '%s': could not determine transport", name)
        return None

    stack = AsyncExitStack()
    try:
        entered = await stack.enter_async_context(
            cast(AsyncContextManager[MCPServer], server)
        )
    except Exception as e:
        await stack.aclose()
        logger.error("Failed to initialize MCP server '%s': %s", name, e)
        return None

    agent_bundle = build_mcp_agent(name, entered, config, default_model)
    if not agent_bundle:
        await stack.aclose()
        return None

    agent, tool_name, tool_description = agent_bundle
    logger.info("Initialized MCP server: %s", name)

    return RunningMCPServer(
        name=name,
        server=entered,
        _closer=stack.aclose,
        agent=agent,
        tool_name=tool_name,
        tool_description=tool_description,
    )


async def initialize_mcp_servers(
    config_path: Path | None = None,
    *,
    default_model: str | None = None,
) -> list[RunningMCPServer]:
    """Initialize MCP servers from config file."""
    data = _load_config(config_path or DEFAULT_MCP_CONFIG_PATH)
    if not data:
        return []

    servers_cfg = data.get("mcpServers", {})
    if not isinstance(servers_cfg, dict):
        logger.warning("Invalid 'mcpServers' in config")
        return []

    results = []
    for name, cfg in servers_cfg.items():
        if server := await _init_server(name, cfg, default_model):
            results.append(server)

    return results


async def cleanup_mcp_servers(servers: list[RunningMCPServer]) -> None:
    """Cleanup MCP server connections."""
    for server in servers:
        await server.shutdown()
