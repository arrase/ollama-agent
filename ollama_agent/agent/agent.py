"""Agent runtime boundary — CUD-inspired architecture."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Self, cast

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..core import (
    create_ollama_chat_model,
    ensure_model_supports_tools,
    validate_reasoning_effort,
)
from ..mcp import load_main_mcp_tools
from ..settings import (
    HISTORY_DB_PATH,
    MEMORY_PATH,
    SKILLS_DIR,
    Settings,
    ensure_memory_file,
    load_instructions,
    save_settings,
)
from ..streaming.parsers import streaming_reasoning, streaming_text
from ..vision import (
    build_multimodal_responses_input,
    capture_display_as_base64,
    extract_display_tokens,
)
from .builtin_tools import BUILTIN_TOOLS, get_tool_timeout
from .middleware import stream_tool_events_mw
from .subagents import build_subagents

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vision helper
# ---------------------------------------------------------------------------


def _maybe_attach_screen_context(prompt: str) -> dict[str, Any]:
    """Convert @dpN tokens to a multimodal user message (LangChain content blocks)."""
    if "@dp" not in prompt:
        return {"role": "user", "content": prompt}

    cleaned, displays = extract_display_tokens(prompt)
    if not displays:
        return {"role": "user", "content": prompt}

    images = [capture_display_as_base64(i) for i in displays]
    return build_multimodal_responses_input(cleaned, images)[0]


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
    _exit_stack: contextlib.AsyncExitStack = field(
        default_factory=contextlib.AsyncExitStack, init=False, repr=False
    )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def reload(self) -> None:
        """Tear down existing resources and rebuild the agent graph."""
        await self._exit_stack.aclose()
        self._exit_stack = contextlib.AsyncExitStack()
        self._instructions = await asyncio.to_thread(load_instructions)
        await asyncio.to_thread(ensure_memory_file, MEMORY_PATH)
        self.graph = await self._build_graph()

    async def _build_graph(self) -> Any:
        ms = self.settings.model
        await ensure_model_supports_tools(ms.name, ms.base_url)

        warnings: list[str] = []
        model = await create_ollama_chat_model(
            model=ms.name,
            base_url=ms.base_url,
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
        mcp_tools = await load_main_mcp_tools(self._exit_stack)

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

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def run_streamed(
        self,
        prompt: str,
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

        hide_reasoning = self.settings.model.reasoning_effort in ("hide", "disabled")

        try:
            async for mode, event in self.graph.astream(
                {"messages": [user_msg]},
                config,
                stream_mode=["messages", "custom"],
            ):
                if mode == "custom" and isinstance(event, dict) and event.get("type"):
                    yield cast(dict[str, Any], event)
                    continue

                if mode == "messages":
                    chunk = event[0] if isinstance(event, tuple) and event else event
                    result = _process_message_chunk(
                        chunk, hide_reasoning=hide_reasoning
                    )
                    if result:
                        yield result
        except Exception as exc:
            _log.error("Streamed agent error: %s", exc)
            yield {"type": "error", "content": str(exc)}

    async def set_model(self, model_name: str) -> str:
        self.settings.model.name = model_name
        await asyncio.to_thread(save_settings, self.settings)
        await self.reload()
        return f"Model set to {model_name}."

    async def aclose(self) -> None:
        await self._exit_stack.aclose()


# ---------------------------------------------------------------------------
# Pure helpers (no state)
# ---------------------------------------------------------------------------


def _process_message_chunk(
    chunk: Any,
    hide_reasoning: bool = False,
) -> dict[str, Any] | None:
    """Process 'messages' chunk to extract reasoning or text deltas."""
    chunk_type = str(getattr(chunk, "type", "") or "").lower()
    chunk_name = getattr(chunk, "__class__", type(chunk)).__name__.lower()
    if chunk_type == "tool" or "tool" in chunk_name:
        return None

    content = getattr(chunk, "content", None)
    additional_kwargs = getattr(chunk, "additional_kwargs", None)

    reasoning = streaming_reasoning(content, additional_kwargs)
    if reasoning:
        if hide_reasoning:
            return None
        return {"type": "reasoning_delta", "content": reasoning}

    text = streaming_text(content)
    if text:
        return {"type": "text_delta", "content": text}
    return None
