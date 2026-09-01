"""Agent runtime boundary — CUD-inspired architecture."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Self, cast

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    create_summarization_tool_middleware,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from ..core import (
    PromptProcessingError,
    create_ollama_chat_model,
    ensure_model_supports_tools,
    process_prompt_mentions,
    validate_reasoning_effort,
)
from ..i18n import _
from ..mcp import load_main_mcp_tools
from ..settings import (
    AGENTS_PATH,
    BUILTIN_SKILLS_DIR,
    HISTORY_DB_PATH,
    MEMORY_PATH,
    SKILLS_DIR,
    TASKS_DIR,
    Settings,
    ensure_memory_file,
    ensure_prompt_files,
    find_agents_file,
    load_instructions,
    load_fs_policy_traversal,
    load_fs_policy_sandboxed,
    load_rag_policy,
    save_settings,
)
from ..streaming.parsers import ThinkTagParser
from .builtin_tools import (
    BUILTIN_TOOLS,
    get_tool_timeout,
    set_active_thread_id,
)
from .compaction import (
    HISTORY_PATH_PREFIX,
    KEEP_RECENT_MESSAGES,
    SUMMARIZATION_SESSION_ID_KEY,
    SUMMARIZATION_STATE_KEY,
    apply_summarization_event,
    build_summary_message,
    compute_state_cutoff,
    find_safe_cutoff,
    generate_summary,
    new_session_id,
    offload_history,
)
from .environment import SKILL_ROOTS, environment_block
from .middleware import stream_tool_events_mw
from .subagents import build_subagents

_log = logging.getLogger(__name__)


def _prepare_instructions(settings: Settings) -> str:
    ensure_prompt_files()
    base_instructions = load_instructions()
    fs_policy = load_fs_policy_traversal() if settings.runtime.allow_traversal else load_fs_policy_sandboxed()
    rag_policy = load_rag_policy()
    instructions = (
        base_instructions
        .replace("{FILESYSTEM_POLICY}", fs_policy)
        .replace("{RAG_POLICY}", rag_policy)
    )
    os_info = environment_block(include_cwd=True)
    ensure_memory_file(MEMORY_PATH)
    return instructions + os_info


@dataclass(slots=True)
class AgentRuntime:
    """Stateful agent runtime wrapping DeepAgents create_deep_agent."""

    settings: Settings = field(default_factory=Settings)
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    yolo_mode: bool = field(default=False)
    stealth_mode: bool = field(default=False)
    auto_approved_tools: set[str] = field(default_factory=set)
    last_context_tokens: int = field(default=0, init=False)
    effective_context_window: int = field(default=0, init=False)
    effective_model_params: dict[str, tuple[Any, str]] = field(default_factory=dict, init=False)
    graph: Any = field(default=None, init=False, repr=False)
    _backend: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _summarization_engine: SummarizationMiddleware | None = field(default=None, init=False, repr=False)
    _instructions: str = field(default="", init=False)
    _checkpointer: Any = field(default=None, init=False, repr=False)
    _memory_checkpointer: Any = field(default=None, init=False, repr=False)
    _checkpointer_stack: contextlib.AsyncExitStack = field(
        default_factory=contextlib.AsyncExitStack, init=False, repr=False
    )
    _exit_stack: contextlib.AsyncExitStack = field(
        default_factory=contextlib.AsyncExitStack, init=False, repr=False
    )
    _init_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def reload(self) -> None:
        """Tear down existing resources and rebuild the agent graph."""
        await self._exit_stack.aclose()
        self._exit_stack = contextlib.AsyncExitStack()
        self._instructions = await asyncio.to_thread(_prepare_instructions, self.settings)
        self.graph = await self._build_graph()

    async def _build_graph(self) -> Any:
        ms = self.settings.model

        model_coro = create_ollama_chat_model(
            model=ms.name,
            base_url=ms.base_url,
            context_window=ms.context_window,
            reasoning_effort=validate_reasoning_effort(ms.reasoning_effort),
            temperature=ms.temperature,
            top_p=ms.top_p,
            top_k=ms.top_k,
            min_p=ms.min_p,
            presence_penalty=ms.presence_penalty,
            repeat_penalty=ms.repeat_penalty,
            warn_callback=_log.warning,
        )
        mcp_coro = load_main_mcp_tools()
        subagents_coro = build_subagents(
            self.settings.subagents,
            model_settings=self.settings.model,
        )
        checkpointer_coro = (
            asyncio.sleep(0, result=self._get_memory_checkpointer())
            if self.stealth_mode
            else self._sqlite_checkpointer()
        )

        model, mcp_tools, subagents, checkpointer = await asyncio.gather(
            model_coro,
            mcp_coro,
            subagents_coro,
            checkpointer_coro,
        )
        await ensure_model_supports_tools(
            ms.name,
            ms.base_url,
            show_info=getattr(model, "show_info", None),
        )

        self.effective_context_window = model.num_ctx
        self.effective_model_params = model.effective_params

        # Backend: CWD for shell + APP_DIR for agent files (memory, etc.)
        timeout = get_tool_timeout()
        default_backend = LocalShellBackend(
            root_dir=Path.cwd().resolve(),
            timeout=timeout,
            virtual_mode=not self.settings.runtime.allow_traversal,
            inherit_env=self.settings.runtime.inherit_env,
        )
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        TASKS_DIR.mkdir(parents=True, exist_ok=True)
        BUILTIN_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        agent_backend = FilesystemBackend(
            root_dir=MEMORY_PATH.parent,
            virtual_mode=True,
        )
        system_skills_backend = FilesystemBackend(
            root_dir=BUILTIN_SKILLS_DIR,
            virtual_mode=True,
        )
        skills_backend = FilesystemBackend(
            root_dir=SKILLS_DIR,
            virtual_mode=True,
        )
        tasks_backend = FilesystemBackend(
            root_dir=TASKS_DIR,
            virtual_mode=True,
        )
        routes: dict[str, Any] = {
            "/agent/": agent_backend,
            "/system_skills/": system_skills_backend,
            "/skills/": skills_backend,
            "/tasks/": tasks_backend,
        }

        # Memory sources: global user memory and AGENTS.md (project / global)
        memory_sources: list[str] = ["/agent/MEMORY.md"]
        if AGENTS_PATH.is_file():
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

        def should_interrupt_tool(request: Any) -> bool:
            return not self.yolo_mode and request.tool_call["name"] not in self.auto_approved_tools

        interrupt_on = {
            tool_name: {
                "allowed_decisions": ["approve", "reject"],
                "when": should_interrupt_tool,
            }
            for tool_name in ("execute", "write_file", "edit_file")
        }

        tools: list[Any] = [*BUILTIN_TOOLS, *mcp_tools]

        summarization_tool_mw = create_summarization_tool_middleware(model, backend)
        self._backend = backend
        self._model = model
        # Live deepagents engine used by manual compaction (see compaction.py).
        self._summarization_engine = summarization_tool_mw._summarization

        kwargs: dict[str, Any] = {
            "model": model,
            "tools": tools,
            "system_prompt": self._instructions,
            "backend": backend,
            "memory": memory_sources,
            "skills": SKILL_ROOTS,
            "checkpointer": checkpointer,
            "middleware": [
                summarization_tool_mw,
                stream_tool_events_mw,
            ],
            "name": "ollama-agent",
            "interrupt_on": interrupt_on,
        }
        if subagents:
            kwargs["subagents"] = subagents

        return create_deep_agent(**kwargs)

    def _get_memory_checkpointer(self) -> Any:
        if self._memory_checkpointer is None:
            self._memory_checkpointer = MemorySaver()
        return self._memory_checkpointer

    async def _sqlite_checkpointer(self) -> Any:
        if self._checkpointer is None:
            saver = AsyncSqliteSaver.from_conn_string(str(HISTORY_DB_PATH))
            self._checkpointer = await self._checkpointer_stack.enter_async_context(saver)
            await self._checkpointer.setup()
        return self._checkpointer

    async def _ensure_graph(self, thread_id: str = "") -> tuple[Any, str, dict[str, Any]]:
        """Resolve the thread, lazily build the graph, and return (graph, thread, config)."""
        thread = thread_id or self.thread_id
        async with self._init_lock:
            if self.graph is None:
                await self.reload()
            if self.graph is None:
                raise RuntimeError(_("Agent graph is not initialized"))
        config: dict[str, Any] = {"configurable": {"thread_id": thread}}
        return self.graph, thread, config

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
        graph, thread, config = await self._ensure_graph(thread_id)
        set_active_thread_id(thread)
        hide_reasoning = self.settings.model.reasoning_effort in ("hide", "disabled")

        inputs: dict[str, Any] | Command
        if isinstance(prompt, Command):
            inputs = prompt
        else:
            # 1. Process prompt mentions
            try:
                mentions_cfg = self.settings.mentions
                processed_prompt, attachments, mention_warnings = process_prompt_mentions(
                    prompt,
                    max_file_size=mentions_cfg.max_file_size,
                    max_files=mentions_cfg.max_files,
                    max_total_size=mentions_cfg.max_total_size,
                )
            except PromptProcessingError as exc:
                yield {"type": "error", "content": str(exc)}
                return

            for warning in mention_warnings:
                _log.warning(warning)

            # 2. Construct user message (multimodal vs text-only)
            if attachments:
                user_msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": processed_prompt}] + attachments,
                }
            else:
                user_msg = {"role": "user", "content": processed_prompt}

            inputs = {"messages": [user_msg]}

        parser = ThinkTagParser()
        async for mode, event in graph.astream(
            inputs,
            config,
            stream_mode=["messages", "custom"],
        ):
            if mode == "custom" and isinstance(event, dict) and event.get("type"):
                yield cast(dict[str, Any], event)
                continue

            if mode == "messages":
                chunk = event[0] if isinstance(event, tuple) and event else event
                meta = getattr(chunk, "response_metadata", None)
                # Usage metadata is the only trusted token source; when the
                # host does not report it, last_context_tokens stays 0 (= unknown).
                if isinstance(meta, dict) and "prompt_eval_count" in meta:
                    self.last_context_tokens = int(meta.get("eval_count", 0)) + int(meta["prompt_eval_count"])
                for result in parser.process_chunk(
                    chunk, hide_reasoning=hide_reasoning
                ):
                    yield result

        for result in parser.flush(hide_reasoning=hide_reasoning):
            yield result

        # Surface pending interrupts (e.g. awaiting tool approval) after the stream ends.
        state = await graph.aget_state(config)
        if state.interrupts:
            yield {"type": "interrupt", "interrupts": state.interrupts, "config": config}

    async def compact_context(self, thread_id: str = "") -> dict[str, Any]:
        """Compact conversation history for the specified thread into a summary."""
        graph, _thread, config = await self._ensure_graph(thread_id)
        if self._model is None:
            raise RuntimeError(_("Agent model is not initialized"))

        state = await graph.aget_state(config)
        values: dict[str, Any] = state.values if state and state.values else {}
        raw_messages = list(values.get("messages") or [])
        if not raw_messages:
            return {"success": False, "message": _("No messages in session to compact.")}

        prior_event = values.get(SUMMARIZATION_STATE_KEY)
        effective = apply_summarization_event(self._summarization_engine, raw_messages, prior_event)

        cutoff = find_safe_cutoff(effective, KEEP_RECENT_MESSAGES)
        if cutoff <= 0:
            return {
                "success": False,
                "message": _("Not enough messages in session to compact (at least 2 messages required)."),
            }
        to_summarize, preserved = effective[:cutoff], effective[cutoff:]

        session_id = new_session_id(values)
        history_path = f"{HISTORY_PATH_PREFIX}/{session_id}.md"
        summary = await generate_summary(self._model, to_summarize)
        file_path = await offload_history(self._backend, to_summarize, history_path)

        new_event: dict[str, Any] = {
            "cutoff_index": compute_state_cutoff(self._summarization_engine, prior_event, cutoff),
            "summary_message": build_summary_message(self._summarization_engine, summary, file_path),
            "file_path": file_path,
        }

        await graph.aupdate_state(
            config,
            {
                SUMMARIZATION_STATE_KEY: new_event,
                SUMMARIZATION_SESSION_ID_KEY: session_id,
            },
        )

        self.last_context_tokens = count_tokens_approximately(
            [new_event["summary_message"], *preserved]
        )

        return {
            "success": True,
            "messages_summarized": len(to_summarize),
            "messages_preserved": len(preserved),
            "file_path": file_path,
            "summary": summary,
        }

    async def get_thread_messages(self, thread_id: str = "") -> list[Any]:
        """Return the raw stored messages for a thread (empty when unknown)."""
        graph, _thread, config = await self._ensure_graph(thread_id)
        state = await graph.aget_state(config)
        values: dict[str, Any] = state.values if state and state.values else {}
        return list(values.get("messages") or [])

    async def count_effective_tokens(self, thread_id: str = "") -> int:
        """Count tokens of the effective context for a thread (after compaction)."""
        graph, _thread, config = await self._ensure_graph(thread_id)
        state = await graph.aget_state(config)
        values: dict[str, Any] = state.values if state and state.values else {}
        effective = apply_summarization_event(
            self._summarization_engine,
            list(values.get("messages") or []),
            values.get(SUMMARIZATION_STATE_KEY),
        )
        return count_tokens_approximately(effective)

    async def set_model(self, model_name: str) -> str:
        self.settings.model.name = model_name
        await asyncio.to_thread(save_settings, self.settings)
        await self.reload()
        return _("Model set to {model_name}.", model_name=model_name)

    async def set_reasoning_effort(self, effort: str) -> str:
        validated = validate_reasoning_effort(effort)
        self.settings.model.reasoning_effort = validated
        await asyncio.to_thread(save_settings, self.settings)
        await self.reload()
        return _("Reasoning effort set to {validated}.", validated=validated)

    async def set_context_window(self, context_window: int | str) -> str:
        self.settings.model.context_window = context_window
        await asyncio.to_thread(save_settings, self.settings)
        await self.reload()
        return _("Context window set to {context_window}.", context_window=context_window)

    async def aclose(self) -> None:
        await self._exit_stack.aclose()
        await self._checkpointer_stack.aclose()
        self._checkpointer = None
        self._memory_checkpointer = None


