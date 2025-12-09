"""AI agent using openai-agents and Ollama."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, cast

from agents import Agent, ModelSettings, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from ..core import ModelCapabilityError, ReasoningEffortValue, ensure_model_supports_tools, validate_reasoning_effort
from ..memory import Mem0Settings, MemoryManager
from ..settings import RunningMCPServer, cleanup_mcp_servers, initialize_mcp_servers, load_instructions
from ..streaming import event_payloads
from .builtin_tools import BUILTIN_TOOLS, set_memory_manager
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OllamaAgent:
    """AI agent backed by Ollama-compatible API with tool support."""

    model: str
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: ReasoningEffortValue = "medium"
    database_path: Optional[Path] = None
    mcp_config_path: Optional[Path] = None
    mem0_settings: Mem0Settings = field(default_factory=Mem0Settings)

    _mcp_servers: list[RunningMCPServer] = field(default_factory=list, init=False)
    _instructions: str = field(init=False, default="")
    _client: AsyncOpenAI = field(init=False)
    _session_manager: SessionManager = field(init=False)
    _memory_manager: MemoryManager = field(init=False)
    _initialized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(self.reasoning_effort)
        self._instructions = load_instructions()
        self._init_client()
        self._memory_manager = MemoryManager(self.mem0_settings)
        self._session_manager = SessionManager(self.database_path)
        set_memory_manager(self._memory_manager)

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    def _init_client(self) -> None:
        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        set_default_openai_client(self._client, use_for_tracing=False)

    def _get_tools(self) -> list[Any]:
        tools: list[Any] = list(BUILTIN_TOOLS)
        for srv in self._mcp_servers:
            if srv.agent:
                tools.append(srv.agent.as_tool(
                    tool_name=srv.tool_name or f"use_{srv.name}",
                    tool_description=srv.tool_description or f"Delegate to '{srv.name}' MCP agent",
                ))
        return tools

    def _create_agent(self, model: str, effort: ReasoningEffortValue) -> Agent:
        ensure_model_supports_tools(model)
        kwargs: dict[str, Any] = {
            "name": "Ollama Assistant",
            "instructions": self._instructions,
            "model": model,
            "tools": self._get_tools(),
        }
        if effort != "disabled":
            kwargs["model_settings"] = ModelSettings(reasoning=Reasoning(effort=cast(Any, effort)))
        return Agent(**kwargs)

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self.mcp_config_path:
            self._mcp_servers = await initialize_mcp_servers(self.mcp_config_path, default_model=self.model)
        self._initialized = True

    async def cleanup(self) -> None:
        if self._mcp_servers:
            await cleanup_mcp_servers(self._mcp_servers)
            self._mcp_servers.clear()
        self._initialized = False

    @asynccontextmanager
    async def lifespan(self) -> AsyncGenerator["OllamaAgent", None]:
        """Async context manager that guarantees initialize/cleanup pairs."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()

    def _resolve(self, model: Optional[str], effort: Optional[str]) -> tuple[str, ReasoningEffortValue]:
        return (
            model or self.model,
            validate_reasoning_effort(effort) if effort else self.reasoning_effort,
        )

    async def run_async(self, prompt: str, model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> str:
        await self.initialize()
        m, e = self._resolve(model, reasoning_effort)
        try:
            result = await Runner.run(self._create_agent(m, e), input=prompt, session=self._session_manager.get_session())
            return str(result.final_output)
        except (ModelCapabilityError, Exception) as exc:
            logger.error("Agent error: %s", exc)
            return f"Error: {exc}"

    async def run_async_streamed(
        self, prompt: str, model: Optional[str] = None, reasoning_effort: Optional[str] = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        await self.initialize()
        m, e = self._resolve(model, reasoning_effort)
        try:
            result = Runner.run_streamed(self._create_agent(m, e), input=prompt, session=self._session_manager.get_session())
            async for event in result.stream_events():
                for payload in event_payloads(event):
                    yield payload
        except (ModelCapabilityError, Exception) as exc:
            logger.error("Streamed agent error: %s", exc)
            yield {"type": "error", "content": str(exc)}
