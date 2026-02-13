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
from langchain_openai import ChatOpenAI

from ..core import ReasoningEffortValue, assistant_text_from_messages, final_text_from_state
from ..core import validate_reasoning_effort
from ..core import ensure_model_supports_tools
from ..memory import Mem0Settings, MemoryManager
from ..rag import RAGManager, RAGSettings
from ..settings import RunningMCPServer, cleanup_mcp_servers, initialize_mcp_servers, load_instructions
from .builtin_tools import BUILTIN_TOOLS, get_tool_timeout, set_memory_manager, set_rag_manager
from .session_manager import SessionManager
from ..vision import build_multimodal_responses_input, capture_display_as_base64, extract_display_tokens

logger = logging.getLogger(__name__)


def _streaming_text(content: Any) -> str:
    """Extract text from a streaming chunk without altering whitespace."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "") if content.get("type") == "text" else ""
    if isinstance(content, list):
        return "".join(
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    return ""


def _deepagents_backend_factory(_: Any) -> FilesystemBackend:
    # DeepAgents uses StateBackend by default (ephemeral, starts empty), which makes
    # tools like `ls/read_file/...` operate on an empty virtual filesystem.
    # Use the real workspace on disk instead, and restrict paths to that root.
    return FilesystemBackend(root_dir=Path.cwd(), virtual_mode=True)


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
    agent_name: str | None = None
    tool_name = next(
        (n for attr in ("name", "tool_name")
         if (n := getattr(request, attr, None)))
        or (n for obj in (getattr(request, "tool", None),)
            if obj for attr in ("name", "__name__")
            if (n := getattr(obj, attr, None))),
        "",
    )

    tool_call = getattr(request, "tool_call", None)
    tool_args: dict[str, Any] | None = None
    if isinstance(tool_call, dict):
        if not tool_name:
            tool_name = str(tool_call.get("name") or "")
        maybe_args = tool_call.get("args")
        if isinstance(maybe_args, dict):
            tool_args = maybe_args
        maybe_meta = tool_call.get("metadata")
        if isinstance(maybe_meta, dict):
            maybe_agent = maybe_meta.get("lc_agent_name")
            if isinstance(maybe_agent, str) and maybe_agent:
                agent_name = maybe_agent
    elif tool_call is not None:
        maybe_name = getattr(tool_call, "name", None)
        if not tool_name and isinstance(maybe_name, str) and maybe_name:
            tool_name = maybe_name
        maybe_args = getattr(tool_call, "args", None)
        if isinstance(maybe_args, dict):
            tool_args = maybe_args

    if not tool_name:
        tool_name = "unknown"

    if (not agent_name) and tool_name == "task" and isinstance(tool_args, dict):
        maybe_task_agent = tool_args.get("name")
        if isinstance(maybe_task_agent, str) and maybe_task_agent:
            agent_name = maybe_task_agent

    if runtime is not None:
        try:
            event: dict[str, Any] = {"type": "tool_call", "name": tool_name}
            if isinstance(agent_name, str) and agent_name:
                event["agent_name"] = agent_name
            runtime.stream_writer(event)
        except Exception:
            pass

    result = await handler(request)

    if runtime is not None:
        try:
            content = getattr(result, "content", result)
            content_str = str(content)
            if not agent_name:
                for attr in ("response_metadata", "additional_kwargs", "metadata"):
                    maybe_meta = getattr(result, attr, None)
                    if isinstance(maybe_meta, dict):
                        maybe_agent_name = maybe_meta.get("lc_agent_name")
                        if isinstance(maybe_agent_name, str) and maybe_agent_name:
                            agent_name = maybe_agent_name
                            break
            # Tool outputs can be large; they are intended for the model, not the UI.
            # Emit only small metadata so CLI/REPL can show progress without dumping
            # the full payload.
            event = {
                "type": "tool_output",
                "output": "",
                "output_len": len(content_str),
            }
            if isinstance(agent_name, str) and agent_name:
                event["agent_name"] = agent_name
            runtime.stream_writer(event)
        except Exception:
            pass
    return result


_stream_tool_events_mw = cast(Any, wrap_tool_call)(_stream_tool_events)


def _shell_policy_from_timeout(timeout_s: int):
    try:
        return HostExecutionPolicy(command_timeout=float(timeout_s))
    except TypeError:
        return HostExecutionPolicy()


def _is_gpt_oss(model: str) -> bool:
    return "gpt-oss" in (model or "").lower()


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

    def _get_tools(self) -> list[Any]:
        return [*BUILTIN_TOOLS]

    def _get_subagents(self) -> list[dict[str, Any]]:
        return [srv.subagent for srv in self._mcp_servers if isinstance(getattr(srv, "subagent", None), dict)]

    def _build_deep_agent(self, model: str, effort: ReasoningEffortValue):
        ensure_model_supports_tools(model)
        subagents = self._get_subagents()

        # `reasoning_effort` must work for gpt-oss models as in the main branch,
        # but the system prompt must remain exclusively user-defined.
        openai_kwargs: dict[str, Any] = {
            "model_name": model,
            "openai_api_base": self.base_url,
            "openai_api_key": self.api_key,
            "temperature": 0,
            # Deep Agents subagent task() currently produces tool outputs that
            # Ollama's /v1/responses endpoint rejects when they are non-string.
            # Use chat completions for compatibility when subagents are enabled.
            "use_responses_api": False if subagents else True,
            "streaming": True,
        }
        if _is_gpt_oss(model) and effort != "disabled":
            openai_kwargs["reasoning_effort"] = effort

        llm = ChatOpenAI(**openai_kwargs)
        shell_mw = ShellToolMiddleware(
            workspace_root=Path.cwd(),
            execution_policy=_shell_policy_from_timeout(get_tool_timeout()),
        )

        return create_deep_agent(
            model=llm,
            tools=self._get_tools(),
            system_prompt=self._instructions,
            subagents=subagents,
            backend=_deepagents_backend_factory,
            middleware=[cast(Any, shell_mw), _stream_tool_events_mw],
        )

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
        user_text_for_history = prompt if isinstance(prompt, str) else str(prompt)
        self._session_manager.append_message("user", user_text_for_history)

        last_state: Any | None = None
        emitted_text = ""
        emitted_from_messages = False
        try:
            async for mode, event in agent.astream(
                {"messages": self._build_messages_with_history(history, prompt)},
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
                        current = assistant_text_from_messages(messages) if isinstance(messages, list) else ""
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

                    # Extract text from streaming chunk without stripping whitespace
                    # (each token may carry leading/trailing spaces that are significant).
                    content = getattr(chunk, "content", None)
                    text = _streaming_text(content)
                    if text:
                        emitted_from_messages = True
                        yield {"type": "text_delta", "content": text}
                    continue

            final = final_text_from_state(last_state) if last_state is not None else emitted_text
            if final:
                self._session_manager.append_message("assistant", final)
        except Exception as exc:
            logger.error("Streamed agent error: %s", exc)
            yield {"type": "error", "content": str(exc)}
