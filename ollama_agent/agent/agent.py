"""AI agent runtime using LangChain DeepAgents and Ollama."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, cast

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ShellToolMiddleware
from langchain.agents.middleware import HostExecutionPolicy
from langchain.agents.middleware import wrap_tool_call
from langchain_ollama import ChatOllama

from ..core import ReasoningEffortValue, validate_reasoning_effort
from ..core import ensure_model_supports_tools
from ..memory import Mem0Settings, MemoryManager
from ..rag import RAGManager, RAGSettings
from ..settings import RunningMCPServer, cleanup_mcp_servers, initialize_mcp_servers, load_instructions
from .builtin_tools import BUILTIN_TOOLS, get_tool_timeout, set_memory_manager, set_rag_manager
from .session_manager import SessionManager
from ..vision import build_multimodal_responses_input, capture_display_as_base64, extract_display_tokens

logger = logging.getLogger(__name__)


def _deepagents_backend_factory(_: Any) -> FilesystemBackend:
    # DeepAgents uses StateBackend by default (ephemeral, starts empty), which makes
    # tools like `ls/read_file/...` operate on an empty virtual filesystem.
    # Use the real workspace on disk instead, and restrict paths to that root.
    return FilesystemBackend(root_dir=Path.cwd(), virtual_mode=True)


def _effort_guidance(model: str, effort: ReasoningEffortValue) -> str:
    if effort == "disabled":
        return ""
    if "gpt-oss" not in model.lower():
        return ""
    match effort:
        case "low":
            return """\
Reasoning effort: LOW.
- Keep reasoning brief.
- Prefer direct answers.
"""
        case "medium":
            return """\
Reasoning effort: MEDIUM.
- Think step-by-step internally.
- Keep the final answer concise.
"""
        case "high":
            return """\
Reasoning effort: HIGH.
- Be thorough and careful.
- Double-check assumptions and edge cases.
"""
        case _:
            return ""


def _final_text_from_state(state: Any) -> str:
    try:
        messages = state.get("messages") if isinstance(state, dict) else None
        if messages:
            last = messages[-1]
            return str(getattr(last, "content", last) or "")
    except Exception:
        pass
    return str(state)


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


async def _stream_tool_events(request, handler):
    """Emit tool_call/tool_output events so the existing renderers keep working."""
    runtime = getattr(request, "runtime", None)
    tool_name = (
        getattr(request, "name", None)
        or getattr(request, "tool_name", None)
        or getattr(getattr(request, "tool", None), "name", None)
        or getattr(getattr(request, "tool", None), "__name__", None)
        or "unknown"
    )
    if runtime is not None:
        try:
            runtime.stream_writer({"type": "tool_call", "name": tool_name})
        except Exception:
            pass

    result = await handler(request)

    if runtime is not None:
        try:
            content = getattr(result, "content", result)
            runtime.stream_writer({"type": "tool_output", "output": str(content)})
        except Exception:
            pass
    return result


_stream_tool_events_mw = cast(Any, wrap_tool_call)(_stream_tool_events)


def _shell_policy_from_timeout(timeout_s: int):
    try:
        return HostExecutionPolicy(command_timeout=float(timeout_s))
    except TypeError:
        return HostExecutionPolicy()


@dataclass(slots=True)
class OllamaAgent:
    """AI agent backed by Ollama with tool support."""

    model: str
    base_url: str = "http://localhost:11434/v1/"  # kept for config compatibility (not used by ChatOllama)
    api_key: str = "ollama"  # kept for config compatibility
    reasoning_effort: ReasoningEffortValue = "medium"
    database_path: Path | None = None
    mcp_config_path: Path | None = None
    mem0_settings: Mem0Settings = field(default_factory=Mem0Settings)
    rag_settings: RAGSettings = field(default_factory=RAGSettings)

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
            self._mcp_servers = await initialize_mcp_servers(self.mcp_config_path, default_model=self.model)
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

    def _get_tools(self) -> list[Any]:
        return [
            *BUILTIN_TOOLS,
            *(srv.delegate_tool for srv in self._mcp_servers if getattr(srv, "delegate_tool", None)),
        ]

    def _build_system_prompt(self, model: str, effort: ReasoningEffortValue) -> str:
        extra = _effort_guidance(model, effort)
        shell_note = (
            "\n\nShell tool notes:\n"
            "- Use shell to run shell commands inside the workspace root.\n"
            "- Do not assume shell state persists across separate runs.\n"
        )
        return "\n\n".join(p for p in (self._instructions, extra, shell_note) if p)

    def _build_deep_agent(self, model: str, effort: ReasoningEffortValue):
        ensure_model_supports_tools(model)

        llm = ChatOllama(model=model, temperature=0)
        shell_mw = ShellToolMiddleware(
            workspace_root=Path.cwd(),
            execution_policy=_shell_policy_from_timeout(get_tool_timeout()),
        )

        return create_deep_agent(
            model=llm,
            tools=self._get_tools(),
            system_prompt=self._build_system_prompt(model, effort),
            backend=_deepagents_backend_factory,
            middleware=[cast(Any, shell_mw), _stream_tool_events_mw],
        )

    def _build_messages(self, prompt: object) -> list[dict[str, Any]]:
        history = self._session_manager.get_message_dicts()
        user_msg = _maybe_attach_screen_context(prompt)
        return [*history, user_msg]

    def _build_messages_with_history(self, history: list[dict[str, Any]], prompt: object) -> list[dict[str, Any]]:
        return [*history, _maybe_attach_screen_context(prompt)]

    async def run_async(self, prompt: object, model: str | None = None, reasoning_effort: str | None = None) -> str:
        await self.initialize()
        m, e = self._resolve(model, reasoning_effort)
        agent = self._build_deep_agent(m, e)
        history = self._session_manager.get_message_dicts()
        user_text_for_history = prompt if isinstance(prompt, str) else str(prompt)
        self._session_manager.append_message("user", user_text_for_history)
        try:
            state = await agent.ainvoke({"messages": self._build_messages_with_history(history, prompt)})
            out = _final_text_from_state(state)
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
        user_text_for_history = prompt if isinstance(prompt, str) else str(prompt)
        self._session_manager.append_message("user", user_text_for_history)

        full_text: list[str] = []
        try:
            async for mode, event in agent.astream(
                {"messages": self._build_messages_with_history(history, prompt)},
                stream_mode=["messages", "custom"],
            ):
                if mode == "custom" and isinstance(event, dict) and event.get("type"):
                    yield cast(dict[str, Any], event)
                    continue

                # Best-effort token streaming from message chunks.
                if mode == "messages":
                    chunk = None
                    if isinstance(event, tuple) and event:
                        chunk = event[0]
                    else:
                        chunk = event
                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and content:
                        full_text.append(content)
                        yield {"type": "text_delta", "content": content}

            final = "".join(full_text)
            if final:
                self._session_manager.append_message("assistant", final)
        except Exception as exc:
            logger.error("Streamed agent error: %s", exc)
            yield {"type": "error", "content": str(exc)}
