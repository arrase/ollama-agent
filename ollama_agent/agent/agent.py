"""Agent runtime boundary — CUD-inspired architecture."""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Self, cast

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..core import (
    ReasoningEffortValue,
    assistant_text_from_messages,
    create_ollama_chat_model,
    ensure_model_supports_tools,
    final_text_from_state,
    resolve_ollama_reasoning,
    validate_reasoning_effort,
)
from ..mcp import RunningMCPServer
from ..rag import RAGManager
from ..settings import (
    HISTORY_DB_PATH,
    MEMORY_PATH,
    MCP_SERVERS_PATH,
    SKILLS_DIR,
    Settings,
    ensure_memory_file,
    load_instructions,
    load_settings,
    save_settings,
)
from ..skills import SkillManager
from ..streaming.parsers import streaming_reasoning, streaming_text
from ..vision import (
    build_multimodal_responses_input,
    capture_display_as_base64,
    extract_display_tokens,
)
from .builtin_tools import BUILTIN_TOOLS, get_tool_timeout, set_rag_manager
from .middleware import stream_tool_events_mw
from .subagents import build_subagents

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vision helper
# ---------------------------------------------------------------------------


def _maybe_attach_screen_context(prompt: object) -> dict[str, Any]:
    """Convert @dpN tokens to a multimodal user message (LangChain content blocks)."""
    text = prompt if isinstance(prompt, str) else str(prompt)
    if "@dp" not in text:
        return {"role": "user", "content": text}

    cleaned, displays = extract_display_tokens(text)
    if not displays:
        return {"role": "user", "content": text}

    images = [capture_display_as_base64(i) for i in displays]
    return build_multimodal_responses_input(cleaned, images)[0]


# ---------------------------------------------------------------------------
# Runtime response
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuntimeResponse:
    content: str
    raw: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Agent Runtime (CUD-inspired)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentRuntime:
    """Stateful agent runtime with DeepAgents graph.

    The runtime rebuilds its graph on each ``reload()`` from settings, MCP
    servers, and instruction files.  All async resources are owned by an
    internal :class:`~contextlib.AsyncExitStack` and cleaned up automatically
    when the runtime is closed.
    """

    settings: Settings = field(default_factory=Settings)
    thread_id: str = "default"
    graph: Any = field(default=None, init=False, repr=False)
    _instructions: str = field(default="", init=False)
    _rag_manager: RAGManager | None = field(default=None, init=False)
    _exit_stack: contextlib.AsyncExitStack = field(
        default_factory=contextlib.AsyncExitStack, init=False, repr=False
    )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @property
    def rag_manager(self) -> RAGManager | None:
        return self._rag_manager

    async def reload(self) -> None:
        """Tear down existing resources and rebuild the agent graph."""
        await self._exit_stack.aclose()
        self._exit_stack = contextlib.AsyncExitStack()
        self._instructions = load_instructions()
        ensure_memory_file(MEMORY_PATH)
        self.graph = await self._build_graph()

    async def _build_graph(self) -> Any:
        ms = self.settings.model
        ensure_model_supports_tools(ms.name, ms.base_url)

        warnings: list[str] = []
        model = create_ollama_chat_model(
            model=ms.name,
            base_url=ms.base_url,
            api_key=None,
            context_window=ms.context_window,
            reasoning_effort=validate_reasoning_effort(ms.reasoning_effort),
            temperature=ms.temperature,
            warn_callback=warnings.append,
        )
        for w in warnings:
            _log.warning(w)

        # Backend: CWD for shell + APP_DIR for agent files (memory, etc.)
        timeout = int(get_tool_timeout())
        default_backend = LocalShellBackend(
            root_dir=Path.cwd(),
            timeout=timeout,
            virtual_mode=not self.settings.runtime.allow_traversal,
        )
        agent_backend = FilesystemBackend(
            root_dir=MEMORY_PATH.parent,
            virtual_mode=True,
        )
        skills_backend = FilesystemBackend(
            root_dir=SKILLS_DIR,
            virtual_mode=True,
        )
        backend = CompositeBackend(
            default=default_backend,
            routes={
                "/agent/": agent_backend,
                "/skills/": skills_backend,
            },
        )

        # MCP flat tools (for main agent, from mcp_servers.json)
        mcp_tools = await self._load_mcp_tools()

        # Custom subagents (from settings.yaml)
        subagents = await build_subagents(
            self.settings.subagents,
            model_settings=self.settings.model,
            exit_stack=self._exit_stack,
        )

        kwargs: dict[str, Any] = dict(
            model=model,
            tools=[*BUILTIN_TOOLS, *mcp_tools],
            system_prompt=self._instructions,
            backend=backend,
            memory=["/agent/MEMORY.md"],
            skills=["/skills/"],
            checkpointer=await self._sqlite_checkpointer(),
            middleware=[
                create_summarization_tool_middleware(model, backend),
                stream_tool_events_mw,
            ],
            name="ollama-agent",
        )
        if subagents:
            kwargs["subagents"] = subagents

        return create_deep_agent(**kwargs)

    async def _sqlite_checkpointer(self) -> Any:
        saver = AsyncSqliteSaver.from_conn_string(str(HISTORY_DB_PATH))
        return await self._exit_stack.enter_async_context(saver)

    async def _load_mcp_tools(self) -> list[Any]:
        """Load flat MCP tools for the main agent from mcp_servers.json."""
        if not MCP_SERVERS_PATH.exists():
            return []

        try:
            data = json.loads(MCP_SERVERS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            _log.error("Failed to load MCP config %s: %s", MCP_SERVERS_PATH, exc)
            return []

        servers_cfg = data.get("mcpServers", data.get("servers", {}))
        if not isinstance(servers_cfg, dict) or not servers_cfg:
            return []

        # Build MultiServerMCPClient connection dict
        connections: dict[str, dict[str, Any]] = {}
        for name, cfg in servers_cfg.items():
            if not isinstance(cfg, dict):
                continue
            conn = _build_mcp_connection(cfg)
            if conn:
                connections[name] = conn
            else:
                _log.warning("Skipping MCP server '%s': could not determine transport", name)

        if not connections:
            return []

        try:
            client = MultiServerMCPClient(connections)  # type: ignore[arg-type]
            tools = await client.get_tools()

            async def _cleanup() -> None:
                try:
                    if hasattr(client, "close"):
                        await client.close()
                except Exception:
                    _log.warning("MCP client cleanup failed", exc_info=True)

            self._exit_stack.push_async_callback(_cleanup)
            _log.info(
                "Loaded %d MCP tools from %d servers",
                len(tools),
                len(connections),
            )
            return tools
        except Exception as exc:
            _log.error("Failed to initialize MCP tools: %s", exc)
            return []

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def invoke(
        self, message: str, *, thread_id: str | None = None
    ) -> RuntimeResponse:
        """Send a message and return the agent response."""
        thread = thread_id or self.thread_id
        if self.graph is None:
            await self.reload()
        if self.graph is None:
            return RuntimeResponse("Runtime failed to initialize.")

        config = {"configurable": {"thread_id": thread}}
        user_msg = _maybe_attach_screen_context(message)
        raw = await self.graph.ainvoke({"messages": [user_msg]}, config)
        return _response_from_raw(raw)

    async def run_streamed(
        self,
        prompt: object,
        *,
        thread_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream agent events for the given prompt."""
        thread = thread_id or self.thread_id
        if self.graph is None:
            await self.reload()
        if self.graph is None:
            yield {"type": "error", "content": "Runtime failed to initialize."}
            return

        config = {"configurable": {"thread_id": thread}}
        user_msg = _maybe_attach_screen_context(prompt)

        # Get initial message count to avoid emitting text from previous turns
        initial_messages_len = 0
        try:
            state = await self.graph.aget_state(config)
            if state and state.values and "messages" in state.values:
                initial_messages_len = len(state.values["messages"])
        except Exception:
            pass

        last_state: Any | None = None
        emitted_text = ""
        emitted_from_messages = False
        try:
            async for mode, event in self.graph.astream(
                {"messages": [user_msg]},
                config,
                stream_mode=["messages", "custom", "values"],
            ):
                if mode == "custom" and isinstance(event, dict) and event.get("type"):
                    yield cast(dict[str, Any], event)
                    continue

                if mode == "values" and isinstance(event, dict):
                    last_state = event
                    if not emitted_from_messages:
                        messages = event.get("messages", [])
                        # Only emit if a new message beyond the user input was added
                        if len(messages) > initial_messages_len + 1:
                            emitted_text, delta = _process_value_event(
                                event, emitted_text
                            )
                            if delta:
                                yield {"type": "text_delta", "content": delta}
                    continue

                if mode == "messages":
                    chunk = event[0] if isinstance(event, tuple) and event else event
                    result = _process_message_chunk(chunk)
                    if result:
                        if result["type"] == "text_delta":
                            emitted_from_messages = True
                        yield result
        except Exception as exc:
            _log.error("Streamed agent error: %s", exc)
            yield {"type": "error", "content": str(exc)}

    async def clear_history(self) -> str:
        """Clear conversation history by deleting the history database."""
        if HISTORY_DB_PATH.exists():
            try:
                HISTORY_DB_PATH.unlink()
            except OSError as exc:
                return f"Failed to clear history: {exc}"
        await self.reload()
        return "History cleared."

    async def view_memory(self) -> str:
        return (
            MEMORY_PATH.read_text(encoding="utf-8")
            if MEMORY_PATH.exists()
            else "Memory is empty."
        )

    async def clear_memory(self) -> str:
        MEMORY_PATH.write_text(
            "# Long-Term Memory\n\nNo persistent memories yet.\n", encoding="utf-8"
        )
        await self.reload()
        return "Memory cleared."

    async def set_model(self, model_name: str) -> str:
        self.settings.model.name = model_name
        save_settings(self.settings)
        await self.reload()
        return f"Model set to {model_name}."

    async def aclose(self) -> None:
        await self._exit_stack.aclose()


# ---------------------------------------------------------------------------
# MCP connection helpers
# ---------------------------------------------------------------------------


def _build_mcp_connection(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build a MultiServerMCPClient connection dict from a server config."""
    if cfg.get("command"):
        out: dict[str, Any] = {
            "transport": "stdio",
            "command": cfg["command"],
        }
        if "args" in cfg:
            out["args"] = cfg["args"]
        if "env" in cfg:
            out["env"] = cfg["env"]
        return out

    url = cfg.get("url") or cfg.get("httpUrl")
    if url:
        out = {"transport": "http", "url": url}
        for k in ("headers", "timeout", "sse_read_timeout"):
            if k in cfg:
                out[k] = cfg[k]
        return out

    return None


# ---------------------------------------------------------------------------
# Pure helpers (no state)
# ---------------------------------------------------------------------------


def _extract_content(raw: Any) -> str:
    if isinstance(raw, dict):
        messages = raw.get("messages")
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
            return str(getattr(last, "content", ""))
        return str(raw.get("content", raw))
    return str(raw)


def _response_from_raw(raw: Any) -> RuntimeResponse:
    content = _extract_content(raw).strip()
    return RuntimeResponse(
        content=content or "The agent finished without text output.", raw=raw
    )


def _process_value_event(
    event: dict[str, Any], emitted_text: str
) -> tuple[str, str | None]:
    """Process 'values' event to extract text deltas."""
    messages = event.get("messages")
    if not isinstance(messages, list) or not messages:
        return emitted_text, None
    current = assistant_text_from_messages(messages) or ""
    if current and current != emitted_text:
        if current.startswith(emitted_text):
            delta = current[len(emitted_text) :]
            return current, delta if delta else None
        return current, current
    return emitted_text, None


def _process_message_chunk(chunk: Any) -> dict[str, Any] | None:
    """Process 'messages' chunk to extract reasoning or text deltas."""
    chunk_type = str(getattr(chunk, "type", "") or "").lower()
    chunk_name = getattr(chunk, "__class__", type(chunk)).__name__.lower()
    if chunk_type == "tool" or "tool" in chunk_name:
        return None

    content = getattr(chunk, "content", None)
    additional_kwargs = getattr(chunk, "additional_kwargs", None)

    reasoning = streaming_reasoning(content, additional_kwargs)
    if reasoning:
        return {"type": "reasoning_delta", "content": reasoning}

    text = streaming_text(content)
    if text:
        return {"type": "text_delta", "content": text}
    return None
