"""Data types for MCP server management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

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
    """Active MCP server bound to its cleanup coroutine.

    Attributes:
        name: Human-readable name for the server.
        server: The connected MCPServer instance.
        _closer: Async function to clean up resources.
        agent: Optional Agent instance for tool delegation.
        tool_name: Name of the tool exposed by this server.
        tool_description: Description of the tool's capabilities.
    """

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
            logger.debug("Cancellation while cleaning up MCP server '%s'", self.name)
        except Exception as cleanup_error:
            logger.debug(
                "Error cleaning up MCP server '%s': %s", self.name, cleanup_error
            )
