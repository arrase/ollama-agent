"""MCP server initialization and cleanup routines."""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, AsyncContextManager, Optional, cast

from agents.mcp import MCPServer

from .builders import build_mcp_agent, build_server
from .types import DEFAULT_MCP_CONFIG_PATH, RunningMCPServer

logger = logging.getLogger(__name__)

# Suppress noisy logs from the agents library
_agents_mcp_logger = logging.getLogger("openai.agents")
_agents_mcp_logger.setLevel(logging.CRITICAL)


def _load_config(config_path: Path) -> Optional[dict[str, Any]]:
    """Load and validate MCP configuration file.

    Returns:
        Parsed configuration dict or None if loading fails.
    """
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except json.JSONDecodeError as parse_error:
        logger.error("Invalid MCP config JSON in %s: %s", config_path, parse_error)
        return None
    except OSError as io_error:
        logger.error("Unable to read MCP config at %s: %s", config_path, io_error)
        return None


async def initialize_mcp_servers(
    config_path: Optional[Path] = None,
    *,
    default_model: Optional[str] = None,
) -> list[RunningMCPServer]:
    """Initialize MCP servers declared in the JSON config file.

    Args:
        config_path: Path to the MCP configuration file.
        default_model: Default model for MCP agents if not specified in config.

    Returns:
        List of successfully initialized RunningMCPServer instances.
    """
    config_path = config_path or DEFAULT_MCP_CONFIG_PATH
    data = _load_config(config_path)
    if data is None:
        return []

    servers_payload = data.get("mcpServers")
    if not isinstance(servers_payload, dict):
        logger.warning("No 'mcpServers' mapping found in %s", config_path)
        return []

    running_servers: list[RunningMCPServer] = []

    for name, raw_config in servers_payload.items():
        server = await _initialize_single_server(
            name, raw_config, default_model
        )
        if server:
            running_servers.append(server)

    return running_servers


async def _initialize_single_server(
    name: str,
    raw_config: Any,
    default_model: Optional[str],
) -> Optional[RunningMCPServer]:
    """Initialize a single MCP server from configuration.

    Args:
        name: Server name.
        raw_config: Configuration dictionary for this server.
        default_model: Fallback model for the agent.

    Returns:
        A RunningMCPServer if successful, None otherwise.
    """
    if not isinstance(raw_config, dict):
        logger.warning(
            "Skipping MCP server '%s': expected object, got %s",
            name,
            type(raw_config).__name__,
        )
        return None

    server = build_server(name, raw_config)
    if server is None:
        logger.warning(
            "Skipping MCP server '%s': could not determine transport", name
        )
        return None

    stack = AsyncExitStack()
    try:
        entered_server = await stack.enter_async_context(
            cast(AsyncContextManager[MCPServer], server)
        )
    except Exception as connect_error:
        await stack.aclose()
        logger.error("Failed to initialize MCP server '%s': %s", name, connect_error)
        return None

    agent_bundle = build_mcp_agent(name, entered_server, raw_config, default_model)
    if not agent_bundle:
        await stack.aclose()
        return None

    agent, tool_name, tool_description = agent_bundle

    logger.info("Initialized MCP server: %s", name)
    return RunningMCPServer(
        name=name,
        server=entered_server,
        _closer=stack.aclose,
        agent=agent,
        tool_name=tool_name,
        tool_description=tool_description,
    )


async def cleanup_mcp_servers(servers: list[RunningMCPServer]) -> None:
    """Cleanup MCP server connections.

    Args:
        servers: List of running servers to shut down.
    """
    for entry in servers:
        await entry.shutdown()
