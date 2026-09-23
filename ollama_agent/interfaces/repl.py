from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ollama
from jinja2.exceptions import TemplateError
from langgraph.types import Command
from rich.console import Console
from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager, set_tool_timeout
from ..agent.episodic_memory import HistoryError
from ..core.common import extract_text
from ..core.models import get_model_thinking_config
from ..i18n import _
from ..rag import RAGContext, RAGManager, load_rag_database
from ..skills import SkillsContext
from ..streaming import StreamingRenderer, extract_action_requests, stream_agent_events
from ..tasks import (
    TaskError,
    TasksContext,
    apply_task_settings,
    parse_var_assignments,
)
from .clipboard import ClipboardError, copy_to_system_clipboard, get_system_clipboard
from .dispatch import REPLHandler, build_repl_handlers, safe_call
from .model_commands import (
    set_context_window,
    set_effort,
    set_model,
)
from .session_commands import (
    export_session,
    get_available_sessions,
    new_session,
    resume_session,
)
from .tui_components import (
    AgentFooter,
    AgentHeader,
    AgentResponse,
    PromptQueueWidget,
    ReplInput,
    SystemOutputWidget,
    ToolApprovalWidget,
    UserMessage,
)

# @-mention regex: matches @"quoted", @'quoted', or @bare at word boundaries.
_AT_MENTION_RE = re.compile(r"""(?:^|[\s\(\[\{<])@(?:"([^"]*)|'([^']*)|([^\s"'\(\[\{<>,;]*))$""")

CMD_MODEL = "/model"
CMD_EFFORT = "/effort"
CMD_CONTEXT = "/context"
CMD_PARAMS = "/params"
CMD_SESSION = "/session"
CMD_TASK = "/task"
CMD_SKILL = "/skill"
CMD_RAG = "/rag"
CMD_MCP = "/mcp"
CMD_AGENTS = "/agents"
CMD_QUEUE = "/queue"
CMD_YOLO = "/yolo"
CMD_STEALTH = "/stealth"
CMD_NEW = "/new"
CMD_CLEAR = "/clear"
CMD_EXIT = "/exit"
CMD_QUIT = "/quit"
_CHAT_SCROLL_ID = "#chat-scroll"
_AUTOCOMPLETE_LIST_ID = "#autocomplete-list"


def _list_models_sync(base_url: str) -> list[Any]:
    """Fetch the list of available Ollama models synchronously."""
    client = ollama.Client(host=base_url)
    response = client.list()
    return list(response.models)


class MainScreen(Screen):
    """Default screen with a guard against Textual crashes when text selection
    ends over a widget detached mid-drag (``parent`` is ``None``)."""

    def _forward_event(self, event: events.Event) -> None:
        try:
            super()._forward_event(event)
        except AttributeError as err:
            if "'NoneType' object has no attribute 'region'" in str(err):
                # Known upstream Textual crash: text selection released over a
                # widget detached mid-drag. Keep observable via logging.
                logging.warning("Worked around Textual detached-widget selection crash", exc_info=err)
                self._select_state = None
                return
            raise


def _get_root_commands() -> list[tuple[str, str]]:
    return [
        (CMD_MODEL, _("Manage models")),
        (CMD_EFFORT, _("Show or set reasoning/thinking effort")),
        (CMD_CONTEXT, _("Show or set context window size (num_ctx)")),
        (CMD_PARAMS, _("Manage model sampling parameters")),
        (CMD_SESSION, _("Manage chat sessions")),
        (CMD_TASK, _("Manage saved tasks")),
        (CMD_SKILL, _("Manage skills")),
        (CMD_RAG, _("Manage RAG databases")),
        (CMD_MCP, _("Manage and check MCP servers")),
        (CMD_AGENTS, _("Manage configured subagents")),
        (CMD_QUEUE, _("Show or clear the prompt queue")),
        (CMD_YOLO, _("Toggle YOLO mode or set it explicitly (on/off)")),
        (CMD_STEALTH, _("Toggle stealth mode or set it explicitly (on/off)")),
        (CMD_NEW, _("Start a new chat session and clear the screen")),
        (CMD_CLEAR, _("Start a new chat session and clear the screen (alias for /new)")),
        (CMD_EXIT, _("Exit the REPL")),
        (CMD_QUIT, _("Exit the REPL (alias for /exit)")),
    ]


def _get_subcommands() -> dict[str, list[tuple[str, str]]]:
    queue_rm_desc = _("Remove a prompt from the queue")
    return {
        CMD_MODEL: [
            ("list", _("List available Ollama models")),
            ("set", _("Switch to a different model")),
        ],
        CMD_CONTEXT: [
            ("set", _("Set context window size (e.g. 8192, 16384, max)")),
        ],
        CMD_PARAMS: [
            ("list", _("Show active model parameters and resolution sources")),
            ("set", _("Set a parameter value (e.g. /params set temperature 0.7)")),
        ],
        CMD_SESSION: [
            ("list", _("List all past sessions")),
            ("search", _("Search past sessions by keyword")),
            ("resume", _("Resume a previous session")),
            ("switch", _("Switch to a previous session (alias for resume)")),
            ("new", _("Start a new session")),
            ("export", _("Export session to Markdown")),
            ("delete", _("Delete a session from history")),
        ],
        CMD_TASK: [
            ("list", _("List all saved tasks")),
            ("create", _("Create a task with agent guidance")),
            ("run", _("Run a saved task prompt")),
            ("delete", _("Delete a saved task")),
        ],
        CMD_SKILL: [
            ("list", _("List all available skills")),
            ("show", _("Show skill details and instructions")),
            ("create", _("Create a skill with agent guidance")),
            ("delete", _("Delete a skill")),
        ],
        CMD_RAG: [
            ("status", _("Show current RAG database status")),
            ("list", _("List all RAG databases")),
            ("create", _("Create a new RAG database")),
            ("delete", _("Delete a RAG database")),
            ("load", _("Load a RAG database")),
            ("unload", _("Unload active RAG database")),
            ("add", _("Add file or directory to RAG")),
        ],
        CMD_MCP: [
            ("list", _("List configured MCP servers and their status")),
            ("reload", _("Reload MCP servers and rebuild tool graph")),
        ],
        CMD_AGENTS: [
            ("list", _("List configured subagents and their properties")),
        ],
        CMD_QUEUE: [
            ("clear", _("Clear all queued prompts")),
            ("rm", queue_rm_desc),
            ("remove", queue_rm_desc),
            ("delete", queue_rm_desc),
        ],
        CMD_YOLO: [
            ("on", _("Enable YOLO mode (bypasses confirmations)")),
            ("off", _("Disable YOLO mode")),
        ],
        CMD_STEALTH: [
            ("on", _("Enable stealth mode (no SQLite history)")),
            ("off", _("Disable stealth mode")),
        ],
    }


IMMEDIATE_COMMANDS: frozenset[tuple[str, str]] = frozenset({
    (CMD_EXIT, "*"), (CMD_QUIT, "*"), (CMD_QUEUE, "*"), (CMD_YOLO, "*"), (CMD_STEALTH, "*"),
    (CMD_MODEL, ""), (CMD_MODEL, "list"),
    (CMD_EFFORT, ""),
    (CMD_CONTEXT, ""),
    (CMD_PARAMS, ""), (CMD_PARAMS, "list"),
    (CMD_SESSION, ""), (CMD_SESSION, "list"), (CMD_SESSION, "search"), (CMD_SESSION, "export"), (CMD_SESSION, "delete"),
    (CMD_TASK, ""), (CMD_TASK, "list"), (CMD_TASK, "delete"),
    (CMD_SKILL, ""), (CMD_SKILL, "list"), (CMD_SKILL, "show"), (CMD_SKILL, "delete"),
    (CMD_RAG, ""), (CMD_RAG, "status"), (CMD_RAG, "list"), (CMD_RAG, "create"), (CMD_RAG, "delete"), (CMD_RAG, "load"), (CMD_RAG, "unload"),
    (CMD_MCP, ""), (CMD_MCP, "list"), (CMD_MCP, "status"),
    (CMD_AGENTS, ""), (CMD_AGENTS, "list"),
})


def _is_immediate_command(val: str) -> bool:
    parts = val.split()
    if not parts:
        return False
    cmd = parts[0].lower()
    sub = parts[1].lower() if len(parts) > 1 else ""
    return (cmd, "*") in IMMEDIATE_COMMANDS or (cmd, sub) in IMMEDIATE_COMMANDS


class _TUIStreamingRenderer(StreamingRenderer):
    def __init__(self, app: OllamaAgentApp, scroll: Any, widget: AgentResponse):
        self.app = app
        self.scroll = scroll
        self.widget = widget
        self._auto_scroll = True
        self._last_scroll_y = scroll.scroll_y
        self._last_max_scroll_y = scroll.max_scroll_y
        self._timer = app.set_interval(0.1, self._do_scroll)

    def _do_scroll(self) -> None:
        if self._auto_scroll:
            self.scroll.scroll_end(animate=False)
            self._last_scroll_y = self.scroll.scroll_y
            self._last_max_scroll_y = self.scroll.max_scroll_y

    def _scroll(self) -> None:
        # If scroll_y decreased but max_scroll_y didn't drop, the user scrolled up.
        if self.scroll.scroll_y < self._last_scroll_y and self.scroll.max_scroll_y >= self._last_max_scroll_y:
            self._auto_scroll = False
        elif self.scroll.scroll_y >= self.scroll.max_scroll_y - 4:
            self._auto_scroll = True
        self._last_scroll_y = self.scroll.scroll_y
        self._last_max_scroll_y = self.scroll.max_scroll_y

    def on_text_delta(self, event: dict[str, Any]) -> None:
        self.widget.append_text(event["content"])
        self._scroll()

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        self.widget.append_thinking(event["content"])
        self._scroll()

    def on_tool_call(self, event: dict[str, Any]) -> None:
        self.widget.add_tool_call(
            name=event["name"],
            agent=event.get("agent_name"),
        )
        self._scroll()

    def on_tool_output(self, event: dict[str, Any]) -> None:
        self.widget.add_tool_output(
            agent=event.get("agent_name"),
            output_len=event.get("output_len"),
        )
        self._scroll()

    def on_error(self, event: dict[str, Any]) -> None:
        self.widget.add_error(event["content"])
        self._scroll()

    def on_warning(self, event: dict[str, Any]) -> None:
        self.widget.add_warning(event["content"])
        self._scroll()

    def close(self) -> None:
        self._timer.stop()
        self.widget.finish_generation()
        self._do_scroll()


def _complete_root_commands(token: str) -> list[tuple[str, Text]]:
    return [
        (
            cmd,
            Text.from_markup(f"[bold #38bdf8]{cmd:<12}[/bold #38bdf8] [dim #8b949e]{desc}[/dim #8b949e]"),
        )
        for cmd, desc in _get_root_commands()
        if cmd.startswith(token)
    ]


def _format_thinking_value(v: Any) -> str:
    if v is False:
        return "false"
    if v is True:
        return "true"
    return str(v)


def _complete_effort(sub_token: str, runtime_model: Any) -> list[tuple[str, Text]]:
    sub_token_lower = sub_token.lower()
    thinking_cfg = (
        get_model_thinking_config(runtime_model.show_info)
        if runtime_model
        else None
    )
    results = []
    if "default".startswith(sub_token_lower):
        results.append(
            (
                f"{CMD_EFFORT} default",
                Text.from_markup(f"[bold #38bdf8]default     [/bold #38bdf8] [dim #8b949e]{_('(model default)')}[/dim #8b949e]"),
            )
        )
    if not (thinking_cfg and "values" in thinking_cfg):
        return results

    default = thinking_cfg.get("default")
    for v in thinking_cfg["values"]:
        v_str = _format_thinking_value(v)
        if v_str.lower().startswith(sub_token_lower):
            def_label = _("(default)") if v == default else ""
            results.append(
                (
                    f"{CMD_EFFORT} {v_str}",
                    Text.from_markup(f"[bold #38bdf8]{v_str:<12}[/bold #38bdf8] [dim #8b949e]{def_label}[/dim #8b949e]"),
                )
            )
    return results


def _complete_subcommands(root_cmd: str, sub_token: str) -> list[tuple[str, Text]]:
    subcommands = _get_subcommands()
    if root_cmd not in subcommands:
        return []
    return [
        (
            f"{root_cmd} {sub}",
            Text.from_markup(f"[bold #38bdf8]{sub:<12}[/bold #38bdf8] [dim #8b949e]{desc}[/dim #8b949e]"),
        )
        for sub, desc in subcommands[root_cmd]
        if sub.startswith(sub_token)
    ]


def _collect_dir_candidates(
    dirs: list[str],
    root_path: Path,
    cwd: Path,
    prefix_lower: str,
    show_hidden: bool,
) -> list[tuple[str, str]]:
    candidate_dirs = []
    for dirname in sorted(dirs):
        if not show_hidden and dirname.startswith("."):
            continue
        rel = (root_path / dirname).relative_to(cwd).as_posix() + "/"
        rel_lower = rel.lower()
        if prefix_lower.startswith(rel_lower) or rel_lower.startswith(prefix_lower):
            candidate_dirs.append((dirname, rel))
    return candidate_dirs


def _iter_dir_completions(candidate_dirs: list[tuple[str, str]], prefix_lower: str) -> Iterator[tuple[str, str]]:
    for _dirname, rel in candidate_dirs:
        rel_lower = rel.lower()
        if rel_lower != prefix_lower and rel_lower.startswith(prefix_lower):
            yield rel, _("dir")


def _filter_file_candidate(
    filename: str,
    root_path: Path,
    cwd: Path,
    prefix_lower: str,
    show_hidden: bool,
) -> tuple[str, str] | None:
    if not show_hidden and filename.startswith("."):
        return None
    rel = (root_path / filename).relative_to(cwd).as_posix()
    if not rel.lower().startswith(prefix_lower):
        return None
    try:
        size_kb = (root_path / filename).stat().st_size / 1024
        meta = f"{size_kb:.1f} KB"
    except OSError:
        meta = _("file")
    return rel, meta


def _iter_file_completions(
    files: list[str],
    root_path: Path,
    cwd: Path,
    prefix_lower: str,
    show_hidden: bool,
) -> Iterator[tuple[str, str]]:
    for filename in sorted(files):
        item = _filter_file_candidate(filename, root_path, cwd, prefix_lower, show_hidden)
        if item is not None:
            yield item


# ─── Main TUI App ────────────────────────────────────────────────────────────
class OllamaAgentApp(App):
    """Main Textual Application representing the Agent's interactive TUI."""

    BINDINGS = [
        ("escape", "cancel_generation", _("Interrupt")),
        ("ctrl+c", "cancel_or_quit", _("Interrupt/Quit")),
        ("super+c", "copy_selection", _("Copy")),
        ("ctrl+shift+c", "copy_selection", _("Copy")),
        ("ctrl+insert", "copy_selection", _("Copy")),
    ]

    def action_cancel_generation(self) -> None:
        sys_out = self.query_one(SystemOutputWidget)
        if sys_out.display:
            self.clear_system_output()
            return

        if self._prompt_queue:
            self._prompt_queue.clear()
            self._update_queue_ui()
            self.show_system_notice(f"[bold #f87171]🛑 {_('Prompt queue cleared.')}[/bold #f87171]")
        if self._is_generating and self._current_worker is not None:
            self._current_worker.cancel()
        elif self._is_approval_pending:
            self._is_approval_pending = False
            footer = self.query_one(AgentFooter)
            footer.set_approval(False)
            self.show_system_notice(f"[bold #f87171]🛑 {_('Approval cancelled.')}[/bold #f87171]")

    def action_cancel_or_quit(self) -> None:
        if self._is_generating or self._is_approval_pending or self._prompt_queue:
            self.action_cancel_generation()
        else:
            self.exit()

    def action_copy_selection(self) -> None:
        selected_text = self.screen.get_selected_text()
        if selected_text:
            self.copy_to_clipboard(selected_text)

    def copy_to_clipboard(self, text: str) -> None:
        super().copy_to_clipboard(text)
        try:
            copy_to_system_clipboard(text)
        except ClipboardError as exc:
            self.notify(_("Failed to copy to system clipboard: {exc}", exc=exc), severity="warning")

    @property
    def clipboard(self) -> str:
        try:
            sys_clip = get_system_clipboard()
        except ClipboardError:
            return super().clipboard
        if sys_clip:
            return sys_clip
        return super().clipboard

    def on_text_selected(self, event: events.TextSelected) -> None:
        self.action_copy_selection()

    def get_default_screen(self) -> Screen:
        return MainScreen(id="_default")

    CSS_PATH = "repl.tcss"

    def __init__(self, repl: OllamaREPL) -> None:
        super().__init__()
        self.repl = repl
        self.repl.app = self
        self._is_generating = False
        self._is_approval_pending = False
        self._prompt_queue: deque[str] = deque()
        self._current_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield AgentHeader(self.repl)
        yield ScrollableContainer(id=_CHAT_SCROLL_ID.lstrip("#"))
        yield OptionList(id=_AUTOCOMPLETE_LIST_ID.lstrip("#"))
        yield PromptQueueWidget(id="prompt-queue")
        yield SystemOutputWidget(id="system-output")
        with Container(id="input-container"), Horizontal(id="input-bar"):
            yield Static("❯ ", id="prompt-char")
            yield ReplInput(id="repl-input")
        yield AgentFooter()

    def show_system_output(self, content: str | Text, title: str | None = None) -> None:
        self.query_one(SystemOutputWidget).show_output(content, title=title)

    def show_system_notice(self, notice: str | Text) -> None:
        self.query_one(SystemOutputWidget).show_notice(notice)

    def clear_system_output(self) -> None:
        self.query_one(SystemOutputWidget).clear_output()

    def on_mount(self) -> None:
        self.query_one(ReplInput).focus()
        self.update_mode_ui()
        self.run_worker(self._warmup_agent(), group="warmup")

    async def _warmup_agent(self) -> None:
        res = self.repl.runtime._ensure_graph()
        if inspect.isawaitable(res):
            await res
        self.query_one(AgentHeader).update_header()

    def update_mode_ui(self) -> None:
        prompt_char = self.query_one("#prompt-char")
        input_container = self.query_one("#input-container")
        yolo = self.repl.runtime.yolo_mode
        stealth = self.repl.runtime.stealth_mode

        input_container.set_class(yolo, "yolo-mode")
        input_container.set_class(stealth, "stealth-mode")

        if yolo and stealth:
            prompt_char.styles.color = "#fbbf24"  # Amber / Dual mode
        elif yolo:
            prompt_char.styles.color = "#f87171"  # Red / Coral
        elif stealth:
            prompt_char.styles.color = "#c084fc"  # Purple / Violet
        else:
            prompt_char.styles.color = "#38bdf8"  # Sky Blue

        header = self.query_one(AgentHeader)
        header.update_header()

    def _update_queue_ui(self) -> None:
        footer = self.query_one(AgentFooter)
        footer.set_queued_count(len(self._prompt_queue))
        self.query_one(PromptQueueWidget).update_queue(self._prompt_queue)

    # ── Input events ──────────────────────────────────────────────────────

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "repl-input":
            self.update_autocomplete(event.text_area.text)

    def on_repl_input_submitted(self, event: ReplInput.Submitted) -> None:
        if event.input.id != "repl-input":
            return
        val = event.value.strip()
        if not val:
            return
        self.clear_system_output()
        event.input.text = ""
        event.input.add_history_entry(val)

        if _is_immediate_command(val):
            worker = self.run_worker(self._run_slash_command(val))
            if not self._is_generating:
                self._current_worker = worker
            return

        if self._is_generating or self._is_approval_pending:
            self._prompt_queue.append(val)
            self._update_queue_ui()
            return

        if val.startswith("/"):
            self._current_worker = self.run_worker(self._run_slash_command(val))
        else:
            scroll = self.query_one(_CHAT_SCROLL_ID)
            scroll.mount(UserMessage(val))
            agent_msg = AgentResponse()
            scroll.mount(agent_msg)
            self._deferred_scroll()
            self._current_worker = self.run_worker(self._run_stream(val, scroll, agent_msg))

    def _process_next_in_queue(self) -> None:
        if not self._prompt_queue or self._is_generating or self._is_approval_pending:
            return
        item = self._prompt_queue.popleft()
        self._update_queue_ui()
        if item.startswith("/"):
            self._current_worker = self.run_worker(self._run_slash_command(item))
        else:
            scroll = self.query_one(_CHAT_SCROLL_ID)
            scroll.mount(UserMessage(item))
            agent_msg = AgentResponse()
            scroll.mount(agent_msg)
            self._deferred_scroll()
            self._current_worker = self.run_worker(self._run_stream(item, scroll, agent_msg))

    # ── Autocomplete ──────────────────────────────────────────────────────

    def hide_autocomplete(self) -> None:
        autolist = self.query_one(_AUTOCOMPLETE_LIST_ID, OptionList)
        autolist.clear_options()
        autolist.display = False
        autolist.highlighted = None

    def _slash_completions(self, text: str) -> list[tuple[str, Text]]:
        parts = text.split(" ")
        num_parts = len(parts)

        # Level 0: Root commands (e.g., "/" or "/mo")
        if num_parts == 1:
            return _complete_root_commands(parts[0])

        root_cmd = parts[0]
        if root_cmd == CMD_EFFORT and num_parts == 2:
            return _complete_effort(parts[1], self.repl.runtime.model)

        # Level 1: Subcommands (e.g., "/task " or "/task r")
        if num_parts == 2:
            return _complete_subcommands(root_cmd, parts[1])

        # Level 2: Arguments / Dynamic entities (e.g., "/task run ")
        if num_parts == 3:
            return self._complete_arguments(root_cmd, parts[1], parts[2])

        return []

    def _complete_model_args(self, root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
        try:
            models = _list_models_sync(self.repl.runtime.settings.model.base_url)
        except (ollama.ResponseError, OSError):
            return []
        results = []
        for m in models:
            if not (m.model and m.model.startswith(arg_token)):
                continue
            size_str = f"{(m.size / (1024**3)):.1f}GB" if m.size else ""
            markup = f"[bold #e6edf3]{m.model:<30}[/bold #e6edf3] [dim #8b949e]{size_str}[/dim #8b949e]"
            results.append((f"{root_cmd} {sub_cmd} {m.model}", Text.from_markup(markup)))
        return results

    def _complete_context_args(self, root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
        presets = ["4096", "8192", "16384", "32768", "65536", "131072", "max"]
        tok_label = _("tokens")
        return [
            (
                f"{root_cmd} {sub_cmd} {p}",
                Text.from_markup(f"[bold #e6edf3]{p:<10}[/bold #e6edf3] [dim #8b949e]{tok_label}[/dim #8b949e]"),
            )
            for p in presets
            if p.startswith(arg_token)
        ]

    def _complete_task_args(self, root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
        tasks = self.repl._task_ctx.task_manager.list_all()
        return [
            (
                f"{root_cmd} {sub_cmd} {tid}",
                Text.from_markup(f"[bold #e6edf3]{tid:<20}[/bold #e6edf3] [dim #8b949e]{t.title}[/dim #8b949e]"),
            )
            for tid, t in tasks
            if tid.startswith(arg_token)
        ]

    def _complete_skill_args(self, root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
        skills = self.repl._skills_ctx.skill_manager.list_all()
        return [
            (
                f"{root_cmd} {sub_cmd} {sid}",
                Text.from_markup(f"[bold #e6edf3]{sid:<20}[/bold #e6edf3] [dim #8b949e]{s.name}[/dim #8b949e]"),
            )
            for sid, s in skills
            if sid.startswith(arg_token)
        ]

    def _complete_session_args(self, root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
        try:
            sessions = get_available_sessions()
        except HistoryError as exc:
            logging.warning("Session autocomplete unavailable: %s", exc)
            return []
        steps_label = _("steps")
        return [
            (
                f"{root_cmd} {sub_cmd} {s['thread_id']}",
                Text.from_markup(
                    f"[bold #e6edf3]{s['thread_id'][:8]:<10}[/bold #e6edf3] "
                    f"[dim #8b949e]{s['steps']} {steps_label}[/dim #8b949e]"
                ),
            )
            for s in sessions
            if s["thread_id"].startswith(arg_token)
        ]

    def _complete_rag_args(self, root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
        dbs = self.repl._get_rag_ctx().rag_manager.list_databases()
        chunks_label = _("chunks")
        return [
            (
                f"{root_cmd} {sub_cmd} {d['name']}",
                Text.from_markup(
                    f"[bold #e6edf3]{d['name']:<20}[/bold #e6edf3] "
                    f"[dim #8b949e]{d['chunks'] if d['chunks'] is not None else 0} {chunks_label}[/dim #8b949e]"
                ),
            )
            for d in dbs
            if d["name"].startswith(arg_token)
        ]

    def _complete_queue_args(self, root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
        clean_arg = arg_token.lstrip("#")
        items = []
        for idx, item in enumerate(self._prompt_queue, 1):
            if clean_arg and not str(idx).startswith(clean_arg):
                continue
            text = item.replace("\n", " ")
            preview = escape(text[:57] + "..." if len(text) > 60 else text)
            items.append(
                (
                    f"{root_cmd} {sub_cmd} {idx}",
                    Text.from_markup(f"[bold #e6edf3]#{idx}[/bold #e6edf3] [dim #8b949e]{preview}[/dim #8b949e]"),
                )
            )
        return items

    def _complete_arguments(self, root_cmd: str, sub_cmd: str, arg_token: str) -> list[tuple[str, Text]]:
        if root_cmd == CMD_MODEL and sub_cmd == "set":
            return self._complete_model_args(root_cmd, sub_cmd, arg_token)
        if root_cmd == CMD_CONTEXT and sub_cmd == "set":
            return self._complete_context_args(root_cmd, sub_cmd, arg_token)
        if root_cmd == CMD_TASK and sub_cmd in ("run", "delete"):
            return self._complete_task_args(root_cmd, sub_cmd, arg_token)
        if root_cmd == CMD_SKILL and sub_cmd in ("show", "delete"):
            return self._complete_skill_args(root_cmd, sub_cmd, arg_token)
        if root_cmd == CMD_SESSION and sub_cmd in ("resume", "switch", "delete"):
            return self._complete_session_args(root_cmd, sub_cmd, arg_token)
        if root_cmd == CMD_RAG and sub_cmd in ("load", "delete"):
            return self._complete_rag_args(root_cmd, sub_cmd, arg_token)
        if root_cmd == CMD_QUEUE and sub_cmd in ("rm", "remove", "delete"):
            return self._complete_queue_args(root_cmd, sub_cmd, arg_token)
        return []

    def update_autocomplete(self, value: str) -> None:
        autolist = self.query_one(_AUTOCOMPLETE_LIST_ID, OptionList)

        # 1. File @-mention candidates
        match = _AT_MENTION_RE.search(value)
        if match:
            prefix = match.group(1) or match.group(2) or match.group(3) or ""
            completions = list(self._file_completions(prefix))
            if completions:
                autolist.clear_options()
                for rel_path, meta in completions[:20]:
                    markup = f"[bold #e6edf3]{rel_path:<40}[/bold #e6edf3] [dim #8b949e]{meta}[/dim #8b949e]"
                    autolist.add_option(
                        Option(
                            prompt=Text.from_markup(markup),
                            id=rel_path,
                        )
                    )
                autolist.highlighted = 0
                autolist.display = True
                return

        # 2. Slash-command candidates
        text = value.lstrip()
        if text.startswith("/"):
            slash_candidates = self._slash_completions(text)
            if slash_candidates:
                autolist.clear_options()
                for item_id, prompt_text in slash_candidates:
                    autolist.add_option(
                        Option(
                            prompt=prompt_text,
                            id=item_id,
                        )
                    )
                autolist.highlighted = 0
                autolist.display = True
                return

        self.hide_autocomplete()

    def accept_completion(self, option_index: int) -> None:
        if option_index < 0:
            return

        autolist = self.query_one(_AUTOCOMPLETE_LIST_ID, OptionList)
        option = autolist.get_option_at_index(option_index)
        if option.id is None:
            return
        completed_text = option.id

        inp = self.query_one(ReplInput)
        val = inp.text

        if completed_text.startswith("/"):
            new_val = completed_text + " "
        else:
            at_idx = val.rfind("@")
            if at_idx == -1:
                new_val = val
            else:
                needs_quote = any(c in completed_text for c in " '\"()[]{},;")
                if needs_quote:
                    completed_text = f'"{completed_text}"'
                suffix = "" if completed_text.endswith("/") else " "
                new_val = val[:at_idx] + "@" + completed_text + suffix

        inp.text = new_val
        inp.action_cursor_line_end()
        self.hide_autocomplete()
        inp.focus()

    # ── File tree walk ────────────────────────────────────────────────────

    def _file_completions(self, prefix: str) -> Iterator[tuple[str, str]]:
        cwd = Path.cwd()
        show_hidden = prefix.startswith(".")
        count = 0
        max_completions = 100
        max_dirs_visited = 50
        dirs_visited = 0

        for root, dirs, files in os.walk(cwd):
            dirs_visited += 1
            if dirs_visited > max_dirs_visited:
                return

            root_path = Path(root)
            prefix_lower = prefix.lower()
            candidate_dirs = _collect_dir_candidates(dirs, root_path, cwd, prefix_lower, show_hidden)
            dirs[:] = [d for d, _rel in candidate_dirs]

            for item in _iter_dir_completions(candidate_dirs, prefix_lower):
                if count >= max_completions:
                    return
                count += 1
                yield item

            for item in _iter_file_completions(files, root_path, cwd, prefix_lower, show_hidden):
                if count >= max_completions:
                    return
                count += 1
                yield item

    # ── Deferred scroll helper ────────────────────────────────────────────

    def _deferred_scroll(self) -> None:
        """Schedule a scroll-to-end after the next layout refresh."""
        scroll = self.query_one(_CHAT_SCROLL_ID)
        self.call_after_refresh(scroll.scroll_end, animate=False)

    # ── Slash command dispatch ────────────────────────────────────

    async def _resume_slash_session(self, args: list[str], scroll: Any) -> None:
        if len(args) < 2:
            usage_msg = _("Usage: /session resume <session_id>")
            self.show_system_notice(f"[bold #f87171]✕ {usage_msg}[/bold #f87171]")
            return
        try:
            resolved = resume_session(self.repl.console, args[1])
        except HistoryError as exc:
            self.show_system_notice(f"[red]{exc}[/red]")
            return
        if resolved:
            self.repl.runtime.thread_id = resolved
            await scroll.remove_children()
            messages = await self.repl.runtime.get_thread_messages(resolved)
            self.repl.runtime.last_context_tokens = await self.repl.runtime.count_effective_tokens(resolved)
            for msg in messages:
                role = msg.type
                content = extract_text(msg.content)
                if not content:
                    continue
                if role in ("human", "user"):
                    scroll.mount(UserMessage(content))
                elif role in ("ai", "assistant"):
                    scroll.mount(AgentResponse(initial_text=content))
            resumed_label = f"{resolved[:8]} ({resolved})"
            notice = _("Resumed session: {session_id}", session_id=resumed_label)
            self.show_system_notice(f"[bold #38bdf8]✓ {notice}[/bold #38bdf8]")
            self.query_one(AgentHeader).update_header()
            self._deferred_scroll()
        else:
            not_found = _("Session not found: {session_id}", session_id=args[1])
            self.show_system_notice(f"[bold #f87171]✕ {not_found}[/bold #f87171]")

    async def _export_slash_session(self, args: list[str]) -> None:
        try:
            out_file = await export_session(
                self.repl.console,
                self.repl.runtime,
                self.repl.runtime.thread_id,
                output_path=args[1] if len(args) > 1 else None,
            )
        except HistoryError as exc:
            self.show_system_notice(f"[red]{exc}[/red]")
            return
        if out_file:
            export_msg = _("Session exported to: {path}", path=out_file)
            self.show_system_notice(f"[bold #38bdf8]✓ {export_msg}[/bold #38bdf8]")
        else:
            self.show_system_notice(f"[bold #f87171]✕ {_('Failed to export session.')}[/bold #f87171]")

    async def _handle_slash_session(self, cmd: str, args: list[str], scroll: Any) -> bool:
        if cmd in (CMD_CLEAR, CMD_NEW) or (cmd == CMD_SESSION and args and args[0] == "new"):
            self.repl._handle_new_session()
            await scroll.remove_children()
            notice = _("New session started: {session_id}", session_id=self.repl.runtime.thread_id[:8])
            self.show_system_notice(f"[bold #38bdf8]✓ {notice}[/bold #38bdf8]")
            self.query_one(AgentHeader).update_header()
            return True

        if cmd != CMD_SESSION or not args:
            return False

        if args[0] in ("resume", "switch"):
            await self._resume_slash_session(args, scroll)
            return True

        if args[0] == "export":
            await self._export_slash_session(args)
            return True

        return False

    async def _handle_slash_task(self, cmd_line: str, args: list[str], scroll: Any) -> bool:
        cmd = cmd_line.split()[0].lower()
        if cmd != CMD_TASK or not args:
            return False

        if args[0] == "create":
            sub_args = args[1:]
            task_info = " ".join(sub_args)
            if task_info:
                prompt_text = (
                    f"[System Instruction: The user executed '/task create {task_info}'. "
                    f"Use your 'task-creator' instructions to guide the user or draft the task, "
                    f"generate a clear and self-contained YAML task file, and save it in /tasks/<task_id>.yaml.]"
                )
            else:
                prompt_text = (
                    "[System Instruction: The user executed '/task create'. "
                    "Use your 'task-creator' instructions to ask what repeatable workflow or prompt "
                    "they want to save as a task, and guide them through creating it in /tasks/<task_id>.yaml.]"
                )
            scroll.mount(UserMessage(cmd_line))
            agent_msg = AgentResponse()
            scroll.mount(agent_msg)
            self._deferred_scroll()
            await self._run_stream(prompt_text, scroll, agent_msg)
            return True

        if args[0] == "run":
            sub_args = args[1:]
            positional = [a for a in sub_args if not a.startswith("-")]
            if not positional:
                self.show_system_notice(f"[bold #f87171]✕ {_('Usage: /task run <id> [-y]')}[/bold #f87171]")
                return True
            target_id = positional[0]
            var_args = positional[1:]
            try:
                _, t = self.repl._task_ctx.resolve_task(target_id)
                variables = parse_var_assignments(var_args)
                rendered_prompt = t.render(variables)
            except (TaskError, ValueError, TemplateError) as exc:
                self.show_system_notice(f"[red]{exc}[/red]")
                return True

            scroll.mount(UserMessage(cmd_line))
            agent_msg = AgentResponse()
            scroll.mount(agent_msg)
            self._deferred_scroll()

            settings = self.repl.runtime.settings
            prev_model = settings.model.name
            prev_effort = settings.model.reasoning_effort
            prev_yolo = self.repl.runtime.yolo_mode
            try:
                apply_task_settings(settings, t)
                if "-y" in sub_args or "--yolo" in sub_args:
                    self.repl.runtime.yolo_mode = True
                await self.repl.runtime.reload()
                await self._run_stream(rendered_prompt, scroll, agent_msg)
            finally:
                settings.model.name = prev_model
                settings.model.reasoning_effort = prev_effort
                self.repl.runtime.yolo_mode = prev_yolo
                await self.repl.runtime.reload()
                self.update_mode_ui()
            return True

        return False

    async def _handle_slash_skill(self, cmd_line: str, args: list[str], scroll: Any) -> bool:
        cmd = cmd_line.split()[0].lower()
        if cmd != CMD_SKILL or not args or args[0] != "create":
            return False

        sub_args = args[1:]
        skill_info = " ".join(sub_args)
        if skill_info:
            prompt_text = (
                f"[System Instruction: The user executed '/skill create {skill_info}'. "
                f"Use your 'skill-creator' instructions to guide the user, gather requirements, "
                f"evaluate whether helper scripts in scripts/ are needed, write the SKILL.md and any scripts "
                f"to /skills/<skill_id>/, and confirm when created.]"
            )
        else:
            prompt_text = (
                "[System Instruction: The user executed '/skill create'. "
                "Use your 'skill-creator' instructions to ask what capability or workflow they want to teach "
                "the agent, evaluate whether helper scripts are needed, and guide them step-by-step through "
                "creating the skill in /skills/<skill_id>/.]"
            )
        scroll.mount(UserMessage(cmd_line))
        agent_msg = AgentResponse()
        scroll.mount(agent_msg)
        self._deferred_scroll()
        await self._run_stream(prompt_text, scroll, agent_msg)
        return True

    async def _run_slash_command(self, cmd_line: str) -> None:
        try:
            parts = cmd_line.split()
            cmd = parts[0].lower()
            args = parts[1:]
            scroll = self.query_one(_CHAT_SCROLL_ID)

            if cmd in (CMD_EXIT, CMD_QUIT):
                self.exit()
                return

            if await self._handle_slash_session(cmd, args, scroll):
                return

            if await self._handle_slash_task(cmd_line, args, scroll):
                return

            if await self._handle_slash_skill(cmd_line, args, scroll):
                return

            commands = self.repl._get_commands()
            if cmd not in commands:
                self.show_system_notice(f"[bold #f87171]✕ {_('Unknown command: {cmd}', cmd=cmd)}[/bold #f87171]")
                return

            handler = commands[cmd]
            scroll_w = scroll.size.width if scroll.size.width > 10 else self.size.width
            self.repl.console.width = max(40, scroll_w - 6)
            self.repl.console.height = 25
            with self.repl.console.capture() as capture:
                await safe_call(handler, args, console=self.repl.console)
            output = capture.get()
            if output:
                self.show_system_output(Text.from_ansi(output), title=cmd_line)

            if cmd in (CMD_YOLO, CMD_STEALTH):
                self.update_mode_ui()
            elif cmd == CMD_QUEUE:
                self._update_queue_ui()
            elif cmd in (CMD_MODEL, CMD_EFFORT, CMD_CONTEXT, CMD_RAG):
                self.query_one(AgentHeader).update_header()
        finally:
            self._process_next_in_queue()

    # ── Streaming chat ────────────────────────────────────────────────────

    async def _handle_approval_decision(
        self, decisions: list[dict[str, Any]], scroll: Any, agent_msg: AgentResponse
    ) -> None:
        self._is_approval_pending = False
        command: Command[Any] = Command(resume={"decisions": decisions})
        await self._run_stream(command, scroll, agent_msg)

    async def _run_stream(self, prompt: str | Command[Any], scroll: Any, agent_msg: AgentResponse) -> None:
        self._is_generating = True
        footer = self.query_one(AgentFooter)
        footer.set_generating(True)

        try:
            try:
                await stream_agent_events(
                    self.repl.runtime,
                    prompt,
                    _TUIStreamingRenderer(self, scroll, agent_msg),
                )
            except asyncio.CancelledError:
                self.show_system_notice(f"[bold #f87171]🛑 {_('Execution interrupted by user.')}[/bold #f87171]")
                if self._prompt_queue:
                    self._prompt_queue.clear()
                    self._update_queue_ui()
                raise
            inp = self.query_one(ReplInput)
            inp.disabled = False

            # Check if the execution got interrupted
            config = {"configurable": {"thread_id": self.repl.runtime.thread_id}}
            state = await self.repl.runtime.graph.aget_state(config)
            if state.interrupts:
                action_requests = extract_action_requests({"interrupts": state.interrupts})
                self._is_approval_pending = True
                footer.set_approval(True)
                approval_widget = ToolApprovalWidget(
                    action_requests=action_requests,
                    app_ref=self,
                    scroll=scroll,
                    agent_msg=agent_msg,
                )
                agent_msg.mount(approval_widget)
                self._deferred_scroll()
            else:
                inp.focus()
        finally:
            self._is_generating = False
            footer.set_generating(False)
            self.query_one(AgentHeader).update_header()
            self._process_next_in_queue()


# ─── OllamaREPL entry-point (unchanged public API) ───────────────────────────


class OllamaREPL:
    """Read-Eval-Print Loop for interacting with the Ollama Agent."""

    def __init__(
        self,
        runtime: AgentRuntime,
        rag_database: str | None = None,
    ):
        self.runtime = runtime
        self.console = Console(force_terminal=True, color_system="truecolor")
        self._task_ctx = TasksContext(console=self.console, settings=self.runtime.settings)
        self._skills_ctx = SkillsContext(console=self.console)
        self._initial_rag_database = rag_database
        self._rag_ctx: RAGContext | None = None
        self._commands: dict[str, REPLHandler] | None = None
        self.app: OllamaAgentApp | None = None

    def _get_rag_ctx(self) -> RAGContext:
        if self._rag_ctx is None:
            mgr = RAGManager(self.runtime.settings.rag)
            self._rag_ctx = RAGContext(console=self.console, rag_manager=mgr)
            set_rag_manager(mgr)
        return self._rag_ctx

    def _get_commands(self) -> dict[str, REPLHandler]:
        """Lazily build and cache REPL command handlers."""
        if self._commands is None:
            self._commands = build_repl_handlers(
                task_ctx=self._task_ctx,
                skills_ctx=self._skills_ctx,
                get_rag_ctx=self._get_rag_ctx,
                console=self.console,
                current_model=lambda: self.runtime.settings.model.name,
                base_url=lambda: self.runtime.settings.model.base_url,
                switch_model=self._switch_model,
                handle_yolo=self._handle_yolo_cmd,
                handle_stealth=self._handle_stealth_cmd,
                handle_queue=self._handle_queue_cmd,
                get_runtime=lambda: self.runtime,
                current_thread_id=lambda: self.runtime.thread_id,
                switch_effort=self._switch_effort,
                switch_context_window=self._switch_context_window,
            )
        return self._commands

    async def cleanup(self) -> None:
        if self._rag_ctx:
            self._rag_ctx.rag_manager.unload()
        await self.runtime.aclose()

    async def run(self) -> None:
        if self._initial_rag_database:
            rag_ctx = self._get_rag_ctx()
            load_rag_database(rag_ctx, self._initial_rag_database)

        set_tool_timeout(self.runtime.settings.runtime.builtin_tool_timeout)

        app = OllamaAgentApp(self)
        try:
            await app.run_async()
        except KeyboardInterrupt:
            pass
        finally:
            await self.cleanup()

    async def _switch_model(self, model_name: str) -> None:
        await set_model(self.console, model_name, runtime=self.runtime)

    async def _switch_effort(self, effort: str) -> None:
        await set_effort(self.console, effort, runtime=self.runtime)

    async def _switch_context_window(self, context_window: str) -> None:
        await set_context_window(self.console, context_window, runtime=self.runtime)

    def _handle_new_session(self) -> None:
        self.runtime.thread_id = new_session(self.console)
        self.runtime.last_context_tokens = 0

    def _handle_yolo_cmd(self, args: list[str]) -> None:
        if args:
            val = args[0].lower()
            if val in ("on", "true", "yes", "1"):
                self.runtime.yolo_mode = True
            elif val in ("off", "false", "no", "0"):
                self.runtime.yolo_mode = False
            else:
                self.console.print(f"[red]{escape(_('Usage: /yolo [on|off]'))}[/red]")
                return
        else:
            self.runtime.yolo_mode = not self.runtime.yolo_mode

        status = _("on") if self.runtime.yolo_mode else _("off")
        color = "red" if self.runtime.yolo_mode else "green"
        self.console.print(f"[bold {color}]{_('YOLO mode is now {status}', status=status)}[/bold {color}]")

    async def _handle_stealth_cmd(self, args: list[str]) -> None:
        if args:
            val = args[0].lower()
            if val in ("on", "true", "yes", "1"):
                self.runtime.stealth_mode = True
            elif val in ("off", "false", "no", "0"):
                self.runtime.stealth_mode = False
            else:
                self.console.print(f"[red]{escape(_('Usage: /stealth [on|off]'))}[/red]")
                return
        else:
            self.runtime.stealth_mode = not self.runtime.stealth_mode

        await self.runtime.reload()

        status = _("on") if self.runtime.stealth_mode else _("off")
        color = "#c084fc" if self.runtime.stealth_mode else "green"
        desc = (
            _("chat history will not be saved to SQLite")
            if self.runtime.stealth_mode
            else _("chat history will be saved to SQLite")
        )
        self.console.print(
            f"[bold {color}]{_('Stealth mode is now {status} ({desc})', status=status, desc=desc)}[/bold {color}]"
        )

    def _handle_queue_rm(self, queue: deque, pos_str: str) -> None:
        if not queue:
            self.console.print(f"[dim]{_('Prompt queue is empty.')}[/dim]")
            return
        raw_pos = pos_str.lstrip("#")
        if not raw_pos.isdigit():
            msg = _("Invalid queue position '{pos}'. Usage: /queue rm <position>", pos=raw_pos)
            self.console.print(f"[red]{msg}[/red]")
            return
        pos = int(raw_pos)
        if pos < 1 or pos > len(queue):
            err = _(
                "Queue position {pos} out of range (queue has {count} items).",
                pos=pos,
                count=len(queue),
            )
            self.console.print(f"[red]{err}[/red]")
            return
        item = queue[pos - 1]
        del queue[pos - 1]
        if self.app is not None:
            self.app._update_queue_ui()
        truncated_text = item.replace("\n", " ")
        if len(truncated_text) > 60:
            truncated_text = truncated_text[:57] + "..."
        msg = _("Removed #{pos} from prompt queue: {text}", pos=pos, text=truncated_text)
        self.console.print(f"[bold #34d399]✓ {msg}[/bold #34d399]")

    def _handle_queue_cmd(self, args: list[str]) -> None:
        queue = self.app._prompt_queue if self.app is not None else deque()
        if not args or args[0] == "list":
            if not queue:
                self.console.print(f"[dim]{_('Prompt queue is empty.')}[/dim]")
                return
            self.console.print(f"[bold #38bdf8]{_('Queued prompts ({count}):', count=len(queue))}[/bold #38bdf8]")
            for i, item in enumerate(queue, 1):
                self.console.print(f"  [dim]#{i}[/dim] {item}")
            return
        if args[0] == "clear":
            count = len(queue)
            queue.clear()
            if self.app is not None:
                self.app._update_queue_ui()
            msg = _("Prompt queue cleared ({count} removed).", count=count)
            self.console.print(f"[bold #34d399]✓ {msg}[/bold #34d399]")
            return
        if args[0] in ("rm", "remove", "delete"):
            if len(args) < 2:
                self.console.print(f"[red]{_('Usage: /queue rm <position>')}[/red]")
                return
            self._handle_queue_rm(queue, args[1])
            return
        err_msg = _("Unknown queue subcommand '{sub}'. Usage: /queue [clear | rm <position>]", sub=args[0])
        self.console.print(f"[red]{err_msg}[/red]")
