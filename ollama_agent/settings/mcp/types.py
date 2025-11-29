"""Data types for MCP server management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from agents import Agent
from agents.mcp import MCPServer

logger = logging.getLogger(__name__)

DEFAULT_MCP_CONFIG_PATH = Path.home() / ".ollama-agent" / "mcp_servers.json"

DEFAULT_AGENT_INSTRUCTIONS = (
    "You operate the '{name}' MCP server. Always fulfill the user's request "
    "by invoking the server tools and return their results directly."
)


@dataclass(slots=True)
class RunningMCPServer:
    """Active MCP server with cleanup capability."""

    name: str
    server: MCPServer
    _closer: Callable[[], Awaitable[None]]
    agent: Agent | None = None
    tool_name: str | None = None
    tool_description: str | None = None

    async def shutdown(self) -> None:
        """Tear down the server without raising on failure."""
        try:
            await self._closer()
        except (asyncio.CancelledError, Exception) as e:
            logger.debug("Error cleaning up MCP server '%s': %s", self.name, e)
