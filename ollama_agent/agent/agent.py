"""Agent runtime boundary — CUD-inspired architecture."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import platform
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Self, cast

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from ..core import (
    PromptProcessingError,
    create_ollama_chat_model,
    ensure_model_supports_tools,
    process_prompt_mentions,
    validate_reasoning_effort,
)
from ..mcp import load_main_mcp_tools
from ..settings import (
    AGENTS_PATH,
    HISTORY_DB_PATH,
    MEMORY_PATH,
    SKILLS_DIR,
    Settings,
    ensure_agents_file,
    ensure_memory_file,
    ensure_prompt_files,
    find_agents_file,
    load_instructions,
    load_fs_policy_traversal,
    load_fs_policy_sandboxed,
    load_rag_policy,
    save_settings,
)
from ..streaming.parsers import streaming_reasoning, streaming_text
from .builtin_tools import BUILTIN_TOOLS, get_rag_manager, get_tool_timeout, rag_search
from .middleware import stream_tool_events_mw
from .subagents import build_subagents

_log = logging.getLogger(__name__)


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
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    yolo_mode: bool = field(default=False)
    auto_approved_tools: set[str] = field(default_factory=set)
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
        await asyncio.to_thread(ensure_prompt_files)
        base_instructions = await asyncio.to_thread(load_instructions)
        if self.settings.runtime.allow_traversal:
            fs_policy = await asyncio.to_thread(load_fs_policy_traversal)
        else:
            fs_policy = await asyncio.to_thread(load_fs_policy_sandboxed)
        
        if "{FILESYSTEM_POLICY}" in base_instructions:
            instructions = base_instructions.replace("{FILESYSTEM_POLICY}", fs_policy)
        else:
            instructions = f"{base_instructions}\n\n{fs_policy}"

        rag_mgr = get_rag_manager()
        rag_active = rag_mgr is not None and rag_mgr.current_database is not None
        if rag_active:
            rag_policy = await asyncio.to_thread(load_rag_policy)
            if "{RAG_POLICY}" in instructions:
                instructions = instructions.replace("{RAG_POLICY}", rag_policy)
            else:
                instructions = f"{instructions}\n\n{rag_policy}"
        else:
            if "{RAG_POLICY}" in instructions:
                instructions = instructions.replace("{RAG_POLICY}", "")

        os_info = f"\n\n# ENVIRONMENT\nOperating System: {platform.system()} ({platform.release()})\n"
        self._instructions = instructions + os_info

        await asyncio.to_thread(ensure_memory_file, MEMORY_PATH)
        await asyncio.to_thread(ensure_agents_file, AGENTS_PATH)
        self.graph = await self._build_graph()

    async def _build_graph(self) -> Any:
        ms = self.settings.model
        await ensure_model_supports_tools(ms.name, ms.base_url)

        model = await create_ollama_chat_model(
            model=ms.name,
            base_url=ms.base_url,
            context_window=ms.context_window,
            reasoning_effort=validate_reasoning_effort(ms.reasoning_effort),
            temperature=ms.temperature,
            warn_callback=_log.warning,
        )

        # Backend: CWD for shell + APP_DIR for agent files (memory, etc.)
        timeout = int(get_tool_timeout())
        default_backend = LocalShellBackend(
            root_dir=Path.cwd(),
            timeout=timeout,
            virtual_mode=not self.settings.runtime.allow_traversal,
            inherit_env=self.settings.runtime.inherit_env,
        )
        agent_backend = FilesystemBackend(
            root_dir=MEMORY_PATH.parent,
            virtual_mode=True,
        )
        skills_backend = FilesystemBackend(
            root_dir=SKILLS_DIR,
            virtual_mode=True,
        )
        routes: dict[str, Any] = {
            "/agent/": agent_backend,
            "/skills/": skills_backend,
        }

        # Memory sources: global user memory and AGENTS.md (project / global)
        memory_sources: list[str] = ["/agent/MEMORY.md"]
        if (MEMORY_PATH.parent / "AGENTS.md").is_file():
            memory_sources.append("/agent/AGENTS.md")

        project_agents = find_agents_file(Path.cwd())
        if project_agents is not None:
            if project_agents.parent == Path.cwd().resolve():
                memory_sources.append(f"/{project_agents.name}")
            else:
                routes["/project/"] = FilesystemBackend(
                    root_dir=project_agents.parent,
                    virtual_mode=True,
                )
                memory_sources.append(f"/project/{project_agents.name}")
        else:
            memory_sources.append("/AGENTS.md")

        backend = CompositeBackend(
            default=default_backend,
            routes=routes,
        )

        # MCP flat tools (for main agent, from mcp_servers.json)
        mcp_tools = await load_main_mcp_tools(self._exit_stack)

        # Custom subagents (from settings.yaml)
        subagents = await build_subagents(
            self.settings.subagents,
            model_settings=self.settings.model,
            exit_stack=self._exit_stack,
        )

        def should_interrupt_tool(request: Any) -> bool:
            return not self.yolo_mode and request.tool_call["name"] not in self.auto_approved_tools

        interrupt_on = {
            "execute": {
                "allowed_decisions": ["approve", "reject"],
                "when": should_interrupt_tool,
            },
            "write_file": {
                "allowed_decisions": ["approve", "reject"],
                "when": should_interrupt_tool,
            },
            "edit_file": {
                "allowed_decisions": ["approve", "reject"],
                "when": should_interrupt_tool,
            },
        }

        tools: list[Any] = [*BUILTIN_TOOLS, *mcp_tools]
        rag_mgr = get_rag_manager()
        if rag_mgr is not None and rag_mgr.current_database is not None:
            tools.append(rag_search)

        kwargs: dict[str, Any] = {
            "model": model,
            "tools": tools,
            "system_prompt": self._instructions,
            "backend": backend,
            "memory": memory_sources,
            "skills": ["/skills/"],
            "checkpointer": await self._sqlite_checkpointer(),
            "middleware": [
                create_summarization_tool_middleware(model, backend),
                stream_tool_events_mw,
            ],
            "name": "ollama-agent",
            "interrupt_on": interrupt_on,
        }
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
        prompt: str | Command,
        *,
        thread_id: str = "",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream agent events for the given prompt."""
        thread = thread_id or self.thread_id
        if self.graph is None:
            await self.reload()
        assert self.graph is not None

        config = {"configurable": {"thread_id": thread}}
        hide_reasoning = self.settings.model.reasoning_effort in ("hide", "disabled")

        inputs: dict[str, Any] | Command
        if isinstance(prompt, Command):
            inputs = prompt
        else:
            # 1. Process prompt mentions
            try:
                mentions_cfg = self.settings.mentions
                processed_prompt, attachments = process_prompt_mentions(
                    prompt,
                    max_file_size=mentions_cfg.max_file_size,
                    max_files=mentions_cfg.max_files,
                    max_total_size=mentions_cfg.max_total_size,
                )
            except PromptProcessingError as exc:
                yield {"type": "error", "content": str(exc)}
                return

            # 2. Construct user message (multimodal vs text-only)
            if attachments:
                user_msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": processed_prompt}] + attachments,
                }
            else:
                user_msg = {"role": "user", "content": processed_prompt}

            inputs = {"messages": [user_msg]}

        async for mode, event in self.graph.astream(
            inputs,
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

        # Check if we were interrupted
        state = await self.graph.aget_state(config)
        if state.interrupts:
            yield {"type": "interrupt", "interrupts": state.interrupts, "config": config}


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
    if getattr(chunk, "type", None) == "tool":
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

