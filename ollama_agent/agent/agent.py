"""AI agent using openai-agents and Ollama."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Iterable, Optional, cast

from agents import (
    Agent,
    ModelSettings,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from ..memory import configure_mem0
from ..settings.configini import Mem0Settings, load_instructions
from ..settings.mcp import RunningMCPServer, cleanup_mcp_servers, initialize_mcp_servers
from ..streaming import event_payloads
from .tools import execute_command, mem0_add_memory, mem0_search_memory
from ..models import (
    ModelCapabilityError,
    ReasoningEffortValue,
    ensure_model_supports_tools,
    validate_reasoning_effort,
)
from .session_manager import SessionManager

logger = logging.getLogger(__name__)




@dataclass(slots=True)
class OllamaAgent:
    model: str
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: ReasoningEffortValue = "medium"
    database_path: Optional[Path] = None
    mcp_config_path: Optional[Path] = None
    mem0_settings: Mem0Settings = field(default_factory=Mem0Settings)
    mcp_servers: list[RunningMCPServer] = field(default_factory=list)
    instructions: str = field(init=False)
    client: AsyncOpenAI = field(init=False)
    agent: Agent = field(init=False)
    session_manager: SessionManager = field(init=False)
    _agent_cache: dict[tuple[str, ReasoningEffortValue], Agent] = field(
        init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(
            self.reasoning_effort)
        self.instructions = load_instructions()

        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        set_default_openai_client(self.client, use_for_tracing=False)

        configure_mem0(self.mem0_settings)

        self.session_manager = SessionManager(self.database_path)
        self.agent = self._create_agent()

    def _build_model_settings(
        self, effort: ReasoningEffortValue | None = None
    ) -> ModelSettings | None:
        active_effort = effort or self.reasoning_effort
        if active_effort == "disabled":
            return None
        return ModelSettings(
            reasoning=Reasoning(effort=cast(Any, active_effort))
        )

    def _create_agent(
        self,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffortValue] = None,
    ) -> Agent:
        selected_model = model or self.model
        selected_effort = reasoning_effort or self.reasoning_effort
        ensure_model_supports_tools(selected_model)
        
        key = (selected_model, selected_effort)
        if key in self._agent_cache:
            return self._agent_cache[key]

        tools: list[Any] = [execute_command, mem0_add_memory, mem0_search_memory]
        tools.extend(
            entry.agent.as_tool(
                tool_name=entry.tool_name or f"use_{entry.name}",
                tool_description=entry.tool_description or f"Delegate tasks to the '{entry.name}' MCP agent",
            )
            for entry in self.mcp_servers if entry.agent
        )

        agent_kwargs: dict[str, Any] = {
            "name": "Ollama Assistant",
            "instructions": self.instructions,
            "model": selected_model,
            "tools": tools,
        }
        if (settings := self._build_model_settings(selected_effort)):
            agent_kwargs["model_settings"] = settings

        agent = Agent(**agent_kwargs)
        self._agent_cache[key] = agent
        return agent

    async def initialize(self) -> None:
        """Initialize the agent and its dependencies."""
        await self._ensure_mcp_servers_initialized()

    async def _ensure_mcp_servers_initialized(self) -> None:
        if not self.mcp_servers and self.mcp_config_path:
            self.mcp_servers = await initialize_mcp_servers(self.mcp_config_path, default_model=self.model)
            if self.mcp_servers:
                self._agent_cache.clear()
                self.agent = self._create_agent()

    async def _get_agent(self, model: Optional[str], reasoning_effort: Optional[str]) -> Agent:
        await self._ensure_mcp_servers_initialized()
        if not model and not reasoning_effort:
            return self.agent
        return self._create_agent(
            model=model or self.model, 
            reasoning_effort=validate_reasoning_effort(reasoning_effort) if reasoning_effort else self.reasoning_effort
        )

    async def cleanup(self) -> None:
        if self.mcp_servers:
            await cleanup_mcp_servers(self.mcp_servers)
            self.mcp_servers.clear()

    async def run_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        try:
            agent = await self._get_agent(model, reasoning_effort)
            result = await Runner.run(agent, input=prompt, session=self.session_manager.get_session())
            return str(result.final_output)
        except (ModelCapabilityError, Exception) as exc:
            logger.error("Error running agent: %s", exc)
            return f"Error: {exc}"

    async def run_async_streamed(
        self,
        prompt: str,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            agent = await self._get_agent(model, reasoning_effort)
            result = Runner.run_streamed(agent, input=prompt, session=self.session_manager.get_session())
            async for event in result.stream_events():
                for payload in event_payloads(event):
                    yield payload
        except (ModelCapabilityError, Exception) as exc:
            logger.error("Error running streamed agent: %s", exc)
            yield {"type": "error", "content": str(exc)}

    def reset_session(self) -> str:
        return self.session_manager.reset_session()

    def load_session(self, session_id: str) -> None:
        self.session_manager.load_session(session_id)

    def get_session_id(self) -> Optional[str]:
        return self.session_manager.get_session_id()

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.session_manager.list_sessions()

    async def get_session_history(self, session_id: Optional[str] = None) -> list[Any]:
        return await self.session_manager.get_session_history(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self.session_manager.delete_session(session_id)
