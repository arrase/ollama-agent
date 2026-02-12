"""Data types for MCP server management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from langchain.tools import BaseTool

from ..paths import MCP_SERVERS_PATH

logger = logging.getLogger(__name__)

DEFAULT_MCP_CONFIG_PATH = MCP_SERVERS_PATH

DEFAULT_AGENT_INSTRUCTIONS = (
    "You operate the '{name}' MCP server. Always fulfill the user's request "
    "by invoking the server tools and return their results directly."
)


@dataclass(slots=True)
class RunningMCPServer:
    """Active MCP server bundle with cleanup capability."""

    name: str
    delegate_tool: BaseTool
    _closer: Callable[[], Awaitable[None]]
    tool_name: str | None = None
    tool_description: str | None = None

    async def shutdown(self) -> None:
        try:
            await self._closer()
        except (asyncio.CancelledError, Exception) as exc:
            logger.debug("Error cleaning up MCP server '%s': %s", self.name, exc)
