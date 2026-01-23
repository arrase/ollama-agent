"""AI agent using openai-agents and Ollama."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, cast

from agents import Agent, ModelSettings, RunConfig, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from ..core import ReasoningEffortValue, ensure_model_supports_tools, validate_reasoning_effort
from ..memory import Mem0Settings, MemoryManager
from ..rag import RAGManager, RAGSettings
from ..settings import RunningMCPServer, cleanup_mcp_servers, initialize_mcp_servers, load_instructions
from ..streaming import event_payloads
from .builtin_tools import BUILTIN_TOOLS, set_memory_manager, set_rag_manager
from .session_manager import SessionManager
from ..vision import build_multimodal_responses_input, capture_display_as_base64, extract_display_tokens

logger = logging.getLogger(__name__)


def _maybe_attach_screen_context(prompt: object) -> object:
    """Convert @dpN tokens in string prompts to multimodal input."""
    if not isinstance(prompt, str) or "@dp" not in prompt:
        return prompt
    cleaned, displays = extract_display_tokens(prompt)
    if not displays:
        return prompt
    return build_multimodal_responses_input(cleaned, [capture_display_as_base64(i) for i in displays])


def _merge_session_history_and_input(history: list[Any], new_input: list[Any]) -> list[Any]:
    """Append new input to conversation history (required by openai-agents for list inputs)."""
    return [*history, *new_input]


@dataclass(slots=True)
class OllamaAgent:
    """AI agent backed by Ollama-compatible API with tool support."""

    model: str
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: ReasoningEffortValue = "medium"
    database_path: Path | None = None
    mcp_config_path: Path | None = None
    mem0_settings: Mem0Settings = field(default_factory=Mem0Settings)
    rag_settings: RAGSettings = field(default_factory=RAGSettings)

    _mcp_servers: list[RunningMCPServer] = field(
        default_factory=list, init=False)
    _instructions: str = field(init=False, default="")
    _client: AsyncOpenAI = field(init=False)
    _session_manager: SessionManager = field(init=False)
    _memory_manager: MemoryManager = field(init=False)
    _rag_manager: RAGManager = field(init=False)
    _initialized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(
            self.reasoning_effort)
        self._instructions = load_instructions()
        self._init_client()
        self._memory_manager = MemoryManager(self.mem0_settings)
        self._rag_manager = RAGManager(self.rag_settings)
        self._session_manager = SessionManager(self.database_path)
        set_memory_manager(self._memory_manager)
        set_rag_manager(self._rag_manager)

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def rag_manager(self) -> RAGManager:
        return self._rag_manager

    def _init_client(self) -> None:
        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")
        self._client = AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key)
        set_default_openai_client(self._client, use_for_tracing=False)

    def _get_tools(self) -> list[Any]:
        return [*BUILTIN_TOOLS, *(srv.agent.as_tool(
            tool_name=srv.tool_name or f"use_{srv.name}",
            tool_description=srv.tool_description or f"Delegate to '{srv.name}' MCP agent"
        ) for srv in self._mcp_servers if srv.agent)]

    def _create_agent(self, model: str, effort: ReasoningEffortValue) -> Agent:
        ensure_model_supports_tools(model)
        settings = ModelSettings(reasoning=Reasoning(
            effort=cast(Any, effort))) if effort != "disabled" else None
        return Agent(name="Ollama Assistant", instructions=self._instructions, model=model,
                     tools=self._get_tools(), **(dict(model_settings=settings) if settings else {}))

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

    def _resolve(self, model: str | None, effort: str | None) -> tuple[str, ReasoningEffortValue]:
        return (
            model or self.model,
            validate_reasoning_effort(
                effort) if effort else self.reasoning_effort,
        )

    def _prepare_run(
        self,
        model: str | None,
        reasoning_effort: str | None,
    ) -> tuple[Agent, Any]:
        m, e = self._resolve(model, reasoning_effort)
        return self._create_agent(m, e), self._session_manager.get_session()

    def _prepare_input(self, prompt: object) -> tuple[object, RunConfig | None]:
        prepared = _maybe_attach_screen_context(prompt)
        run_config = (
            RunConfig(session_input_callback=_merge_session_history_and_input)
            if isinstance(prepared, list)
            else None
        )
        return prepared, run_config

    async def run_async(self, prompt: object, model: str | None = None, reasoning_effort: str | None = None) -> str:
        await self.initialize()
        try:
            agent, session = self._prepare_run(model, reasoning_effort)
            prepared_input, run_config = self._prepare_input(prompt)
            result = await Runner.run(
                agent,
                input=prepared_input,
                session=session,
                run_config=run_config,
            )
            return str(result.final_output)
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            return f"Error: {exc}"

    async def run_async_streamed(
        self, prompt: object, model: str | None = None, reasoning_effort: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        await self.initialize()
        try:
            agent, session = self._prepare_run(model, reasoning_effort)
            prepared_input, run_config = self._prepare_input(prompt)
            result = Runner.run_streamed(
                agent,
                input=prepared_input,
                session=session,
                run_config=run_config,
            )
            async for event in result.stream_events():
                for payload in event_payloads(event):
                    yield payload
        except Exception as exc:
            logger.error("Streamed agent error: %s", exc)
            yield {"type": "error", "content": str(exc)}
