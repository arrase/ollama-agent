"""AI agent runtime using LangChain DeepAgents and Ollama."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, cast

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_openai import ChatOpenAI

from ..core import ReasoningEffortValue, assistant_text_from_messages, final_text_from_state
from ..core import validate_reasoning_effort
from ..core import ensure_model_supports_tools
from ..memory import Mem0Settings, MemoryManager
from ..rag import RAGManager, RAGSettings
from ..settings import RunningMCPServer, cleanup_mcp_servers, initialize_mcp_servers, load_instructions
from .builtin_tools import BUILTIN_TOOLS, get_tool_timeout, set_memory_manager, set_rag_manager
from .middleware import stream_tool_events_mw
from .session_manager import SessionManager
from ..streaming.parsers import streaming_reasoning, streaming_text
from ..vision import build_multimodal_responses_input, capture_display_as_base64, extract_display_tokens

logger = logging.getLogger(__name__)


def _maybe_attach_screen_context(prompt: object) -> dict[str, Any]:
    """Convert @dpN tokens to a multimodal user message (LangChain content blocks)."""
    text = prompt if isinstance(prompt, str) else str(prompt)
    if "@dp" not in text:
        return {"role": "user", "content": text}

    cleaned, displays = extract_display_tokens(text)
    if not displays:
        return {"role": "user", "content": text}

    images = [capture_display_as_base64(i) for i in displays]
    # Returns a messages list; take the single user message.
    return build_multimodal_responses_input(cleaned, images)[0]



@dataclass(slots=True)
class OllamaAgent:
    """AI agent backed by Ollama with tool support."""

    model: str
    base_url: str = "http://localhost:11434/v1/"  # OpenAI-compatible Ollama endpoint
    api_key: str = "ollama"  # kept for config compatibility
    reasoning_effort: ReasoningEffortValue = "medium"
    database_path: Path | None = None
    mcp_config_path: Path | None = None
    mem0_settings: Mem0Settings = field(default_factory=Mem0Settings)
    rag_settings: RAGSettings = field(default_factory=RAGSettings)
    skills_dirs: tuple[str, ...] = ()

    _mcp_servers: list[RunningMCPServer] = field(default_factory=list, init=False)
    _instructions: str = field(init=False, default="")
    _session_manager: SessionManager = field(init=False)
    _memory_manager: MemoryManager = field(init=False)
    _rag_manager: RAGManager = field(init=False)
    _initialized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(self.reasoning_effort)
        self._instructions = load_instructions()
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

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self.mcp_config_path:
            self._mcp_servers = await initialize_mcp_servers(
                self.mcp_config_path,
                default_model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
            )
        self._initialized = True

    async def cleanup(self) -> None:
        if self._mcp_servers:
            await cleanup_mcp_servers(self._mcp_servers)
            self._mcp_servers.clear()
        self._initialized = False

    @asynccontextmanager
    async def lifespan(self) -> AsyncGenerator["OllamaAgent", None]:
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()

    def _resolve(self, model: str | None, effort: str | None) -> tuple[str, ReasoningEffortValue]:
        return (
            model or self.model,
            validate_reasoning_effort(effort) if effort else self.reasoning_effort,
        )

    def _build_deep_agent(self, model: str, effort: ReasoningEffortValue):
        ensure_model_supports_tools(model)

        # `reasoning_effort` must work for gpt-oss models as in the main branch,
        # but the system prompt must remain exclusively user-defined.
        openai_kwargs: dict[str, Any] = {
            "model_name": model,
            "openai_api_base": self.base_url,
            "openai_api_key": self.api_key,
            "temperature": 0,
            # Always use the Responses API so reasoning/thinking tokens are
            # streamed as structured content blocks (type='reasoning').
            # The chat completions API drops Ollama's ``reasoning`` field.
            "use_responses_api": True,
            "streaming": True,
        }
        if "gpt-oss" in model.lower() and effort != "disabled":
            openai_kwargs["reasoning_effort"] = effort

        kwargs: dict[str, Any] = {
            "model": ChatOpenAI(**openai_kwargs),
            "tools": list(BUILTIN_TOOLS),
            "system_prompt": self._instructions,
            "subagents": [srv.subagent for srv in self._mcp_servers],
            # LocalShellBackend extends FilesystemBackend with an `execute` tool
            # that runs shell commands directly on the host.
            "backend": LocalShellBackend(root_dir=Path.cwd(), timeout=int(get_tool_timeout()), virtual_mode=False),
            "middleware": [stream_tool_events_mw],
        }
        if self.skills_dirs:
            kwargs["skills"] = list(self.skills_dirs)
        return create_deep_agent(**kwargs)

    async def run_async(self, prompt: object, model: str | None = None, reasoning_effort: str | None = None) -> str:
        await self.initialize()
        m, e = self._resolve(model, reasoning_effort)
        agent = self._build_deep_agent(m, e)
        history = self._session_manager.get_message_dicts()
        user_text_for_history = prompt if isinstance(prompt, str) else str(prompt)
        self._session_manager.append_message("user", user_text_for_history)
        try:
            state = await agent.ainvoke({"messages": [*history, _maybe_attach_screen_context(prompt)]})
            out = final_text_from_state(state)
            self._session_manager.append_message("assistant", out)
            return out
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            return f"Error: {exc}"

    async def run_async_streamed(
        self, prompt: object, model: str | None = None, reasoning_effort: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        await self.initialize()
        m, e = self._resolve(model, reasoning_effort)
        agent = self._build_deep_agent(m, e)

        history = self._session_manager.get_message_dicts()
        history_len = len(history)
        user_text_for_history = prompt if isinstance(prompt, str) else str(prompt)
        self._session_manager.append_message("user", user_text_for_history)

        last_state: Any | None = None
        emitted_text = ""
        emitted_from_messages = False
        try:
            async for mode, event in agent.astream(
                {"messages": [*history, _maybe_attach_screen_context(prompt)]},

                stream_mode=["messages", "custom", "values"],
            ):
                if mode == "custom" and isinstance(event, dict) and event.get("type"):
                    yield cast(dict[str, Any], event)
                    continue

                # Capture aggregate state (includes the full assistant message).
                if mode == "values" and isinstance(event, dict):
                    last_state = event
                    if not emitted_from_messages:
                        messages = event.get("messages")
                        current_messages = messages[history_len:] if isinstance(messages, list) else None
                        current = assistant_text_from_messages(current_messages) if isinstance(current_messages, list) else ""
                        if current and current != emitted_text:
                            if current.startswith(emitted_text):
                                delta = current[len(emitted_text) :]
                                emitted_text = current
                                if delta:
                                    yield {"type": "text_delta", "content": delta}
                            else:
                                emitted_text = current
                                yield {"type": "text_delta", "content": current}
                    continue

                if mode == "messages":
                    chunk = event[0] if isinstance(event, tuple) and event else event

                    chunk_type = str(getattr(chunk, "type", "") or "").lower()
                    chunk_name = getattr(chunk, "__class__", type(chunk)).__name__.lower()
                    if chunk_type == "tool" or "tool" in chunk_name:
                        continue

                    content = getattr(chunk, "content", None)

                    # Check for reasoning/thinking tokens first (chain-of-thought).
                    reasoning = streaming_reasoning(content)
                    if reasoning:
                        yield {"type": "reasoning_delta", "content": reasoning}
                        continue

                    # Extract text from streaming chunk without stripping whitespace
                    # (each token may carry leading/trailing spaces that are significant).
                    text = streaming_text(content)
                    if text:
                        emitted_from_messages = True
                        yield {"type": "text_delta", "content": text}
                    continue

            final = emitted_text
            if last_state is not None and isinstance(last_state, dict):
                messages = last_state.get("messages")
                current_messages = messages[history_len:] if isinstance(messages, list) else None
                if isinstance(current_messages, list) and current_messages:
                    final = assistant_text_from_messages(current_messages) or final
                if not final:
                    final = final_text_from_state(last_state)
            if final:
                self._session_manager.append_message("assistant", final)
        except Exception as exc:
            logger.error("Streamed agent error: %s", exc)
            yield {"type": "error", "content": str(exc)}
