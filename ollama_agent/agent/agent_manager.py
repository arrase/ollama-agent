# ollama_agent/agent/agent_manager.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

from agents import Agent, ModelSettings
from openai.types.shared import Reasoning
from ..settings.mcp import (
    RunningMCPServer,
    cleanup_mcp_servers,
    initialize_mcp_servers,
)
from ..utils import (
    ReasoningEffortValue,
    ensure_model_supports_tools,
    validate_reasoning_effort,
)
from .tools import execute_command

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentManager:
    """Manages the lifecycle of AI agents."""

    model: str
    reasoning_effort: ReasoningEffortValue = "medium"
    mcp_config_path: Optional[Path] = None
    instructions: str = ""
    mcp_servers: list[RunningMCPServer] = field(init=False, default_factory=list)
    _agent_cache: dict[tuple[str, ReasoningEffortValue], Agent] = field(
        init=False, default_factory=dict
    )

    def _build_model_settings(
        self, effort: ReasoningEffortValue | None = None
    ) -> ModelSettings | None:
        active_effort = effort or self.reasoning_effort
        if active_effort == "disabled":
            return None
        return ModelSettings(reasoning=Reasoning(effort=cast(Any, active_effort)))

    def _create_agent(
        self,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffortValue] = None,
    ) -> Agent:
        selected_model = model or self.model
        selected_effort = reasoning_effort or self.reasoning_effort
        ensure_model_supports_tools(selected_model)
        cache_key = (selected_model, selected_effort)
        cached_agent = self_agent_cache.get(cache_key)
        if cached_agent is not None:
            return cached_agent

        model_settings = self._build_model_settings(selected_effort)
        agent_kwargs: dict[str, Any] = {
            "name": "Ollama Assistant",
            "instructions": self.instructions,
            "model": selected_model,
            "tools": [execute_command],
            "mcp_servers": [entry.server for entry in self.mcp_servers],
        }
        if model_settings is not None:
            agent_kwargs["model_settings"] = model_settings

        agent = Agent(**agent_kwargs)
        self._agent_cache[cache_key] = agent
        return agent

    async def _ensure_mcp_servers_initialized(self) -> None:
        if not self.mcp_servers and self.mcp_config_path:
            self.mcp_servers = await initialize_mcp_servers(self.mcp_config_path)
            if self.mcp_servers:
                self._agent_cache.clear()

    async def get_agent(
        self, model: Optional[str], reasoning_effort: Optional[str]
    ) -> Agent:
        await self._ensure_mcp_servers_initialized()
        selected_model = model or self.model
        effort: ReasoningEffortValue = (
            validate_reasoning_effort(reasoning_effort)
            if reasoning_effort
            else self.reasoning_effort
        )
        return self._create_agent(model=selected_model, reasoning_effort=effort)

    async def cleanup(self) -> None:
        if self.mcp_servers:
            await cleanup_mcp_servers(self.mcp_servers)
            self.mcp_servers.clear()
