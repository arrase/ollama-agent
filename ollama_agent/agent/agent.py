"""AI agent using openai-agents and Ollama."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from ..settings.configini import load_instructions
from ..utils import ReasoningEffortValue, validate_reasoning_effort
from .agent_manager import AgentManager
from .runner import run_async_streamed
from .session_manager import SessionManager


@dataclass(slots=True)
class OllamaAgent:
    model: str
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: ReasoningEffortValue = "medium"
    database_path: Optional[Path] = None
    mcp_config_path: Optional[Path] = None
    agent_manager: AgentManager = field(init=False)
    session_manager: SessionManager = field(init=False)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(self.reasoning_effort)
        instructions = load_instructions()

        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        set_default_openai_client(client, use_for_tracing=False)

        self.session_manager = SessionManager(self.database_path)
        self.agent_manager = AgentManager(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            mcp_config_path=self.mcp_config_path,
            instructions=instructions,
        )

    async def cleanup(self) -> None:
        await self.agent_manager.cleanup()

    async def run_async_streamed(
        self,
        prompt: str,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in run_async_streamed(
            self.agent_manager,
            self.session_manager,
            prompt,
            model=model,
            reasoning_effort=reasoning_effort,
        ):
            yield event

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
