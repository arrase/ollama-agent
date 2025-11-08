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
from openai.types.responses import ResponseReasoningTextDeltaEvent, ResponseTextDeltaEvent
from openai.types.shared import Reasoning

from ..settings.configini import load_instructions
from ..settings.mcp import RunningMCPServer, cleanup_mcp_servers, initialize_mcp_servers
from .tools import execute_command
from ..utils import ReasoningEffortValue, validate_reasoning_effort
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


def _raw_event_payloads(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, ResponseReasoningTextDeltaEvent) and data.delta:
        yield {"type": "reasoning_delta", "content": data.delta}
    elif isinstance(data, ResponseTextDeltaEvent) and data.delta:
        yield {"type": "text_delta", "content": data.delta}


def _item_event_payloads(item: Any) -> Iterable[dict[str, Any]]:
    item_type = getattr(item, "type", "")
    if item_type == "tool_call_item":
        yield {"type": "tool_call", "name": getattr(item, "name", "unknown")}
    elif item_type == "tool_call_output_item":
        yield {"type": "tool_output", "output": str(getattr(item, "output", ""))}
    elif item_type == "reasoning":
        summary = getattr(item, "summary", "")
        if summary:
            yield {"type": "reasoning_summary", "content": summary}


def _event_payloads(event: Any) -> Iterable[dict[str, Any]]:
    event_type = getattr(event, "type", "")
    if event_type == "raw_response_event":
        yield from _raw_event_payloads(getattr(event, "data", None))
    elif event_type == "run_item_stream_event":
        yield from _item_event_payloads(getattr(event, "item", None))
    elif event_type == "agent_updated_stream_event":
        agent_name = getattr(getattr(event, "new_agent", None), "name", "unknown")
        yield {"type": "agent_update", "name": agent_name}


@dataclass(slots=True)
class OllamaAgent:
    model: str
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: ReasoningEffortValue = "medium"
    database_path: Optional[Path] = None
    mcp_config_path: Optional[Path] = None
    instructions: str = field(init=False)
    client: AsyncOpenAI = field(init=False)
    agent: Agent = field(init=False)
    mcp_servers: list[RunningMCPServer] = field(
        init=False, default_factory=list)
    session_manager: SessionManager = field(init=False)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(
            self.reasoning_effort)
        self.instructions = load_instructions()

        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        set_default_openai_client(self.client, use_for_tracing=False)

        self.session_manager = SessionManager(self.database_path)
        self.agent = self._create_agent()

    def _build_model_settings(
        self, effort: ReasoningEffortValue | None = None
    ) -> ModelSettings:
        active_effort = effort or self.reasoning_effort
        return ModelSettings(
            reasoning=Reasoning(effort=cast(Any, active_effort))
        )

    def _create_agent(
        self,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffortValue] = None,
    ) -> Agent:
        return Agent(
            name="Ollama Assistant",
            instructions=self.instructions,
            model=model or self.model,
            tools=[execute_command],
            mcp_servers=[entry.server for entry in self.mcp_servers],
            model_settings=self._build_model_settings(reasoning_effort),
        )

    async def _ensure_mcp_servers_initialized(self) -> None:
        if not self.mcp_servers and self.mcp_config_path:
            self.mcp_servers = await initialize_mcp_servers(self.mcp_config_path)
            if self.mcp_servers:
                self.agent = self._create_agent()

    async def _get_agent(self, model: Optional[str], reasoning_effort: Optional[str]) -> Agent:
        await self._ensure_mcp_servers_initialized()
        if not model and not reasoning_effort:
            return self.agent
        effort: ReasoningEffortValue = (
            validate_reasoning_effort(reasoning_effort)
            if reasoning_effort
            else self.reasoning_effort
        )
        return self._create_agent(model=model, reasoning_effort=effort)

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
        agent = await self._get_agent(model, reasoning_effort)
        try:
            result = await Runner.run(agent, input=prompt, session=self.session_manager.get_session())
            return str(result.final_output)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error running agent: %s", exc)
            return f"Error: {exc}"

    async def run_async_streamed(
        self,
        prompt: str,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        agent = await self._get_agent(model, reasoning_effort)
        try:
            result = Runner.run_streamed(
                agent, input=prompt, session=self.session_manager.get_session())
            async for event in result.stream_events():
                for payload in _event_payloads(event):
                    yield payload
        except Exception as exc:  # noqa: BLE001
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
