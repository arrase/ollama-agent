from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ollama
from rich.console import Console
from rich.text import Text

from textual import events
from textual.app import App, ComposeResult
from textual.worker import Worker
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command

from .clipboard import copy_to_system_clipboard, get_system_clipboard
from .tui_components import (
    AgentFooter,
    AgentHeader,
    AgentResponse,
    ReplInput,
    SystemMessage,
    ToolApprovalWidget,
    UserMessage,
)

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager, set_tool_timeout
from ..core.common import extract_text
from ..i18n import _
from ..rag import RAGContext, RAGManager, load_rag_database
from ..skills import SkillsContext
from ..tasks.commands import CLIContext, TaskError
from .dispatch import REPLCommand, build_repl_handlers, safe_call
from .model_commands import _list_models_sync, set_effort, set_model
from .session_commands import (
    compact_session,
    export_session,
    get_available_sessions,
    new_session,
    resume_session,
)
from ..streaming import StreamingRenderer, stream_agent_events

# @-mention regex: matches @"quoted", @'quoted', or @bare at word boundaries.
_AT_MENTION_RE = re.compile(
    r"""(?:^|[\s\(\[\{<])@(?:"([^"]*)|'([^']*)|([^\s"'\(\[\{<>,;]*))$"""
)


def _get_root_commands() -> list[tuple[str, str]]:
    return [
        ("/model", _("Manage models")),
        ("/effort", _("Show or set reasoning/thinking effort")),
        ("/params", _("Manage model sampling parameters")),
        ("/session", _("Manage chat sessions")),
        ("/compact", _("Compact conversation history into a summary")),
        ("/task", _("Manage saved tasks")),
        ("/skill", _("Manage skills")),
        ("/rag", _("Manage RAG databases")),
        ("/mcp", _("Manage and check MCP servers")),
        ("/yolo", _("Toggle YOLO mode or set it explicitly (on/off)")),
        ("/new", _("Start a new chat session and clear the screen")),
        ("/clear", _("Start a new chat session and clear the screen (alias for /new)")),
        ("/exit", _("Exit the REPL")),
    ]


def _get_subcommands() -> dict[str, list[tuple[str, str]]]:
    return {
        "/model": [
            ("list", _("List available Ollama models")),
            ("set", _("Switch to a different model")),
        ],
        "/params": [
            ("list", _("Show active model parameters and resolution sources")),
            ("set", _("Set a parameter value (e.g. /params set temperature 0.7)")),
        ],
        "/session": [
            ("list", _("List all past sessions")),
            ("search", _("Search past sessions by keyword")),
            ("resume", _("Resume a previous session")),
            ("new", _("Start a new session")),
            ("export", _("Export session to Markdown")),
            ("delete", _("Delete a session from history")),
        ],
        "/task": [
            ("list", _("List all saved tasks")),
            ("create", _("Create a task with agent guidance")),
            ("run", _("Run a saved task prompt")),
            ("delete", _("Delete a saved task")),
        ],
        "/skill": [
            ("list", _("List all available skills")),
            ("show", _("Show skill details and instructions")),
            ("create", _("Create a skill with agent guidance")),
            ("delete", _("Delete a skill")),
        ],
        "/rag": [
            ("status", _("Show current RAG database status")),
            ("list", _("List all RAG databases")),
            ("create", _("Create a new RAG database")),
            ("delete", _("Delete a RAG database")),
            ("load", _("Load a RAG database")),
            ("unload", _("Unload active RAG database")),
            ("add", _("Add file or directory to RAG")),
        ],
        "/mcp": [
            ("list", _("List configured MCP servers and their status")),
        ],
        "/yolo": [
            ("on", _("Enable YOLO mode (bypasses confirmations)")),
            ("off", _("Disable YOLO mode")),
        ],
    }


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
        if self._is_generating and self._current_worker is not None:
            self._current_worker.cancel()

    def action_cancel_or_quit(self) -> None:
        if self._is_generating:
            if self._current_worker is not None:
                self._current_worker.cancel()
        else:
            self.exit()

    def action_copy_selection(self) -> None:
        selected_text = self.screen.get_selected_text()
        if selected_text:
            self.copy_to_clipboard(selected_text)

    def copy_to_clipboard(self, text: str) -> None:
        super().copy_to_clipboard(text)
        copy_to_system_clipboard(text)

    @property
    def clipboard(self) -> str:
        sys_clip = get_system_clipboard()
        if sys_clip:
            return sys_clip
        return super().clipboard

    def on_text_selected(self, event: events.TextSelected) -> None:
        selected_text = self.screen.get_selected_text()
        if selected_text:
            self.copy_to_clipboard(selected_text)

    CSS_PATH = "repl.css"

    def __init__(self, repl: OllamaREPL) -> None:
        super().__init__()
        self.repl = repl
        self._is_generating = False
        self._current_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield AgentHeader(self.repl)
        yield ScrollableContainer(id="chat-scroll")
        yield OptionList(id="autocomplete-list")
        with Container(id="input-container"):
            with Horizontal(id="input-bar"):
                yield Static("❯ ", id="prompt-char")
                yield ReplInput(id="repl-input")
        yield AgentFooter()

    def on_mount(self) -> None:
        self.query_one(ReplInput).focus()
        self.update_yolo_ui()

    def update_yolo_ui(self) -> None:
        prompt_char = self.query_one("#prompt-char")
        input_container = self.query_one("#input-container")
        input_container.set_class(self.repl.runtime.yolo_mode, "yolo-mode")
        if self.repl.runtime.yolo_mode:
            prompt_char.styles.color = "#f87171"  # Red / Coral
        else:
            prompt_char.styles.color = "#38bdf8"  # Sky Blue

        # Update the header immediately
        header = self.query_one(AgentHeader)
        header.update_header()

    # ── Input events ──────────────────────────────────────────────────────

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "repl-input":
            self.update_autocomplete(event.text_area.text)

    def on_repl_input_submitted(self, event: ReplInput.Submitted) -> None:
        if event.input.id != "repl-input":
            return
        if self._is_generating:
            return
        val = event.value.strip()
        if not val:
            return
        event.input.text = ""
        event.input.add_history_entry(val)

        if val.startswith("/"):
            self._current_worker = self.run_worker(self._run_slash_command(val))
        else:
            scroll = self.query_one("#chat-scroll")
            scroll.mount(UserMessage(val))
            agent_msg = AgentResponse()
            scroll.mount(agent_msg)
            self._deferred_scroll()
            self._current_worker = self.run_worker(self._stream_chat(val, scroll, agent_msg))

    # ── Autocomplete ──────────────────────────────────────────────────────

    def hide_autocomplete(self) -> None:
        autolist = self.query_one("#autocomplete-list", OptionList)
        autolist.clear_options()
        autolist.display = False
        autolist.highlighted = None

    def _slash_completions(self, text: str) -> list[tuple[str, Text]]:
        parts = text.split(" ")
        num_parts = len(parts)
        root_commands = _get_root_commands()
        subcommands = _get_subcommands()

        # Level 0: Root commands (e.g., "/" or "/mo")
        if num_parts == 1:
            token = parts[0]
            return [
                (
                    cmd,
                    Text.from_markup(f"[bold #38bdf8]{cmd:<12}[/bold #38bdf8] [dim #8b949e]{desc}[/dim #8b949e]"),
                )
                for cmd, desc in root_commands
                if cmd.startswith(token)
            ]

        root_cmd = parts[0]
        if root_cmd not in subcommands:
            return []

        # Level 1: Subcommands (e.g., "/task " or "/task r")
        if num_parts == 2:
            sub_token = parts[1]
            return [
                (
                    f"{root_cmd} {sub}",
                    Text.from_markup(f"[bold #38bdf8]{sub:<12}[/bold #38bdf8] [dim #8b949e]{desc}[/dim #8b949e]"),
                )
                for sub, desc in subcommands[root_cmd]
                if sub.startswith(sub_token)
            ]

        # Level 2: Arguments / Dynamic entities (e.g., "/task run ")
        if num_parts == 3:
            sub_cmd = parts[1]
            arg_token = parts[2]

            if root_cmd == "/model" and sub_cmd == "set":
                try:
                    models = _list_models_sync(self.repl.runtime.settings.model.base_url)
                except (ollama.ResponseError, OSError):
                    return []
                return [
                    (
                        f"{root_cmd} {sub_cmd} {m.model}",
                        Text.from_markup(
                            f"[bold #e6edf3]{m.model:<30}[/bold #e6edf3] [dim #8b949e]{f'{(m.size / (1024**3)):.1f}GB' if m.size else ''}[/dim #8b949e]"
                        ),
                    )
                    for m in models
                    if m.model and m.model.startswith(arg_token)
                ]

            if root_cmd == "/task" and sub_cmd in ("run", "delete"):
                tasks = self.repl._task_ctx.task_manager.list_all()
                return [
                    (
                        f"{root_cmd} {sub_cmd} {tid}",
                        Text.from_markup(f"[bold #e6edf3]{tid:<20}[/bold #e6edf3] [dim #8b949e]{t.title}[/dim #8b949e]"),
                    )
                    for tid, t in tasks
                    if tid.startswith(arg_token)
                ]

            if root_cmd == "/skill" and sub_cmd in ("show", "delete"):
                skills = self.repl._skills_ctx.skill_manager.list_all()
                return [
                    (
                        f"{root_cmd} {sub_cmd} {sid}",
                        Text.from_markup(f"[bold #e6edf3]{sid:<20}[/bold #e6edf3] [dim #8b949e]{s.name}[/dim #8b949e]"),
                    )
                    for sid, s in skills
                    if sid.startswith(arg_token)
                ]

            if root_cmd == "/session" and sub_cmd in ("resume", "switch", "delete"):
                sessions = get_available_sessions()
                return [
                    (
                        f"{root_cmd} {sub_cmd} {s['thread_id']}",
                        Text.from_markup(f"[bold #e6edf3]{s['thread_id'][:8]:<10}[/bold #e6edf3] [dim #8b949e]{s['steps']} {_('steps')}[/dim #8b949e]"),
                    )
                    for s in sessions
                    if s["thread_id"].startswith(arg_token)
                ]

            if root_cmd == "/rag" and sub_cmd in ("load", "delete"):
                dbs = self.repl._get_rag_ctx().rag_manager.list_databases()
                return [
                    (
                        f"{root_cmd} {sub_cmd} {d['name']}",
                        Text.from_markup(f"[bold #e6edf3]{d['name']:<20}[/bold #e6edf3] [dim #8b949e]{d['chunks'] if d['chunks'] is not None else 0} {_('chunks')}[/dim #8b949e]"),
                    )
                    for d in dbs
                    if d["name"].startswith(arg_token)
                ]

        return []

    def update_autocomplete(self, value: str) -> None:
        autolist = self.query_one("#autocomplete-list", OptionList)

        # 1. File @-mention candidates
        match = _AT_MENTION_RE.search(value)
        if match:
            prefix = match.group(1) or match.group(2) or match.group(3) or ""
            completions = list(self._file_completions(prefix))
            if completions:
                autolist.clear_options()
                for rel_path, meta in completions[:20]:
                    autolist.add_option(
                        Option(
                            prompt=Text.from_markup(f"[bold #e6edf3]{rel_path:<40}[/bold #e6edf3] [dim #8b949e]{meta}[/dim #8b949e]"),
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

        autolist = self.query_one("#autocomplete-list", OptionList)
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
        tree = os.walk(cwd)
        max_dirs_visited = 50
        dirs_visited = 0

        for root, dirs, files in tree:
            dirs_visited += 1
            if dirs_visited > max_dirs_visited:
                return

            root_path = Path(root)
            prefix_lower = prefix.lower()
            candidate_dirs = []
            for dirname in sorted(dirs):
                if not show_hidden and dirname.startswith("."):
                    continue
                try:
                    rel = (root_path / dirname).relative_to(cwd).as_posix() + "/"
                except ValueError:
                    continue
                rel_lower = rel.lower()
                if prefix_lower.startswith(rel_lower) or rel_lower.startswith(prefix_lower):
                    candidate_dirs.append((dirname, rel))

            dirs[:] = [d for d, _rel in candidate_dirs]

            for _dirname, rel in candidate_dirs:
                if count >= max_completions:
                    return
                rel_lower = rel.lower()
                if rel_lower == prefix_lower or not rel_lower.startswith(prefix_lower):
                    continue
                count += 1
                yield rel, _("dir")

            for filename in sorted(files):
                if count >= max_completions:
                    return
                if not show_hidden and filename.startswith("."):
                    continue
                try:
                    rel = (root_path / filename).relative_to(cwd).as_posix()
                except ValueError:
                    continue
                if not rel.lower().startswith(prefix_lower):
                    continue
                meta = _("file")
                try:
                    size_kb = (root_path / filename).stat().st_size / 1024
                    meta = f"{size_kb:.1f} KB"
                except OSError:
                    pass
                count += 1
                yield rel, meta

    # ── Deferred scroll helper ────────────────────────────────────────────

    def _deferred_scroll(self) -> None:
        """Schedule a scroll-to-end after the next layout refresh."""
        scroll = self.query_one("#chat-scroll")
        self.call_after_refresh(scroll.scroll_end, animate=False)

    # ── Slash command dispatch ────────────────────────────────────

    async def _run_slash_command(self, cmd_line: str) -> None:
        parts = cmd_line.split()
        cmd = parts[0].lower()
        args = parts[1:]
        scroll = self.query_one("#chat-scroll")

        if cmd in ("/exit", "/quit"):
            self.exit()
            return

        if cmd in ("/clear", "/new") or (cmd == "/session" and args and args[0] == "new"):
            await self.repl._handle_new_session([])
            await scroll.remove_children()
            scroll.mount(SystemMessage(f"[bold #38bdf8]✓ {_('New session started: {session_id}', session_id=self.repl.runtime.thread_id[:8])}[/bold #38bdf8]"))
            self.query_one(AgentHeader).update_header()
            self._deferred_scroll()
            return

        if cmd in ("/compact", "/compress"):
            scroll.mount(SystemMessage(f"[dim]⚡ {_('Compacting conversation context...')}[/dim]"))
            self._deferred_scroll()
            res = await self.repl.runtime.compact_context()
            if res["success"]:
                msg_text = (
                    f"[bold #38bdf8]✓ {_('Context compacted successfully:')}[/bold #38bdf8]\n"
                    f"  • [dim]{_('Messages summarized:')}[/dim] {res['messages_summarized']}\n"
                    f"  • [dim]{_('Recent messages preserved:')}[/dim] {res['messages_preserved']}"
                )
                if res.get("file_path"):
                    msg_text += f"\n  • [dim]{_('History offloaded to:')}[/dim] [cyan]{res['file_path']}[/cyan]"
                scroll.mount(SystemMessage(msg_text))
                self.query_one(AgentHeader).update_header()
            else:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ {_('Compaction skipped:')}[/bold #f87171] {res.get('message', _('Failed to compact.'))}"))
            self._deferred_scroll()
            return

        if cmd == "/session" and args and args[0] in ("resume", "switch"):
            if len(args) < 2:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ {_('Usage: /session resume <session_id>')}[/bold #f87171]"))
                self._deferred_scroll()
                return
            resolved = resume_session(self.repl.console, args[1])
            if resolved:
                self.repl.runtime.thread_id = resolved
                await scroll.remove_children()
                if self.repl.runtime.graph is not None:
                    config = {"configurable": {"thread_id": resolved}}
                    state = await self.repl.runtime.graph.aget_state(config)
                    if state and state.values and "messages" in state.values:
                        messages = state.values["messages"]
                        event = state.values.get("_summarization_event")
                        effective = (
                            self.repl.runtime._summarization_mw._apply_event_to_messages(
                                messages, event
                            )
                            if self.repl.runtime._summarization_mw
                            else messages
                        )
                        self.repl.runtime.last_context_tokens = (
                            count_tokens_approximately(effective)
                        )
                        for msg in messages:
                            role = getattr(msg, "type", "unknown")
                            content = extract_text(getattr(msg, "content", ""))
                            if not content:
                                continue
                            if role in ("human", "user"):
                                scroll.mount(UserMessage(content))
                            elif role in ("ai", "assistant"):
                                scroll.mount(AgentResponse(initial_text=content))
                scroll.mount(SystemMessage(f"[bold #38bdf8]✓ {_('Resumed session: {session_id}', session_id=f'{resolved[:8]} ({resolved})')}[/bold #38bdf8]"))
                self.query_one(AgentHeader).update_header()
                self._deferred_scroll()
            else:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ {_('Session not found: {session_id}', session_id=args[1])}[/bold #f87171]"))
                self._deferred_scroll()
            return

        if cmd == "/session" and args and args[0] == "export":
            out_file = await export_session(
                self.repl.console,
                self.repl.runtime,
                self.repl.runtime.thread_id,
                output_path=args[1] if len(args) > 1 else None,
            )
            if out_file:
                scroll.mount(SystemMessage(f"[bold #38bdf8]✓ {_('Session exported to: {path}', path=out_file)}[/bold #38bdf8]"))
            else:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ {_('Failed to export session.')}[/bold #f87171]"))
            self._deferred_scroll()
            return

        if cmd == "/task" and args and args[0] == "create":
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
            self._current_worker = self.run_worker(self._stream_chat(prompt_text, scroll, agent_msg))
            return

        if cmd == "/skill" and args and args[0] == "create":
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
            self._current_worker = self.run_worker(self._stream_chat(prompt_text, scroll, agent_msg))
            return

        if cmd == "/task" and args and args[0] == "run":
            sub_args = args[1:]
            target_id = next((a for a in sub_args if not a.startswith("-")), "")
            if not target_id:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ {_('Usage: /task run <id> [-y]')}[/bold #f87171]"))
                self._deferred_scroll()
                return
            try:
                tid, t = self.repl._task_ctx._find_or_exit(target_id)
            except (TaskError, SystemExit) as exc:
                scroll.mount(SystemMessage(f"[red]{exc}[/red]"))
                self._deferred_scroll()
                return

            scroll.mount(SystemMessage(Text.from_markup(
                f"[bold #38bdf8]▶ {_('Executing Task: {title} ({task_id})', title=t.title, task_id=tid)}[/bold #38bdf8]\n"
                f"  [dim]{_('Model:')}[/dim] {t.model} [dim]·[/dim] [dim]{_('Effort:')}[/dim] {t.reasoning_effort}"
            )))
            agent_msg = AgentResponse()
            scroll.mount(agent_msg)
            self._deferred_scroll()

            self.repl.runtime.settings.model.name = t.model
            self.repl.runtime.settings.model.reasoning_effort = t.reasoning_effort
            if "-y" in sub_args or "--yolo" in sub_args:
                self.repl.runtime.yolo_mode = True
            await self.repl.runtime.reload()
            await self._stream_chat(t.prompt, scroll, agent_msg)
            return

        commands = self.repl._get_commands()
        if cmd not in commands:
            scroll.mount(SystemMessage(f"[bold #f87171]✕ {_('Unknown command: {cmd}', cmd=cmd)}[/bold #f87171]"))
            self._deferred_scroll()
            return

        spec = commands[cmd]
        scroll_w = scroll.size.width if scroll.size.width > 10 else self.size.width
        self.repl.console._width = max(40, scroll_w - 6)
        self.repl.console._height = 25
        with self.repl.console.capture() as capture:
            await safe_call(spec.handler, args)
        output = capture.get()
        if output:
            scroll.mount(SystemMessage(Text.from_ansi(output)))
            self._deferred_scroll()

        if cmd == "/yolo":
            self.update_yolo_ui()
        elif cmd == "/rag" and args and args[0] in ("load", "unload", "delete"):
            await self.repl.runtime.reload()
        elif cmd in ("/model", "/effort"):
            self.query_one(AgentHeader).update_header()

    # ── Streaming chat ────────────────────────────────────────────────────
 
    async def _stream_chat(self, prompt: str, scroll: Any, agent_msg: AgentResponse) -> None:
        await self._run_stream(prompt, scroll, agent_msg)

    async def _handle_approval_decision(self, decisions: list[dict[str, Any]], scroll: Any, agent_msg: AgentResponse) -> None:
        command: Command[Any] = Command(resume={"decisions": decisions})
        await self._run_stream(command, scroll, agent_msg)

    async def _run_stream(self, prompt: str | Command[Any], scroll: Any, agent_msg: AgentResponse) -> None:
        self._is_generating = True
        footer = self.query_one(AgentFooter)
        footer.set_generating(True)

        try:
            try:
                await stream_agent_events(self.repl.runtime, prompt, _TUIStreamingRenderer(self, scroll, agent_msg), auto_close=True)
            except asyncio.CancelledError:
                scroll.mount(SystemMessage(f"[bold #f87171]🛑 {_('Execution interrupted by user.')}[/bold #f87171]"))
                self._deferred_scroll()
                inp = self.query_one(ReplInput)
                inp.disabled = False
                inp.focus()
                footer.set_approval(False)
                raise
            except Exception as e:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ {_('Error:')}[/bold #f87171] [red]{e}[/red]"))
                self._deferred_scroll()
                inp = self.query_one(ReplInput)
                inp.disabled = False
                inp.focus()
                footer.set_approval(False)
                return

            # Check if the execution got interrupted
            config = {"configurable": {"thread_id": self.repl.runtime.thread_id}}
            state = await self.repl.runtime.graph.aget_state(config)
            if state.interrupts:
                interrupt_val = state.interrupts[0].value
                action_requests = interrupt_val.get("action_requests", [])
                if action_requests:
                    inp = self.query_one(ReplInput)
                    inp.disabled = True
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
                inp = self.query_one(ReplInput)
                inp.disabled = False
                inp.focus()
        finally:
            self._is_generating = False
            footer.set_generating(False)



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
        self._task_ctx = CLIContext(console=self.console)
        self._skills_ctx = SkillsContext(console=self.console)
        self._initial_rag_database = rag_database
        self._rag_ctx: RAGContext | None = None
        self._commands: dict[str, REPLCommand] | None = None

    def _get_rag_ctx(self) -> RAGContext:
        if self._rag_ctx is None:
            mgr = RAGManager(self.runtime.settings.rag)
            self._rag_ctx = RAGContext(console=self.console, rag_manager=mgr)
            set_rag_manager(mgr)
        return self._rag_ctx

    def _get_commands(self) -> dict[str, REPLCommand]:
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
                handle_exit=lambda _args: None,
                handle_new=self._handle_new_session,
                handle_task_create=lambda _args: None,
                handle_skill_create=lambda _args: None,
                handle_yolo=self._handle_yolo_cmd,
                current_thread_id=lambda: self.runtime.thread_id,
                handle_session_resume=self._handle_session_resume,
                handle_session_export=self._handle_session_export,
                handle_compact=self._handle_compact,
                get_runtime=lambda: self.runtime,
                current_effort=lambda: self.runtime.settings.model.reasoning_effort,
                switch_effort=self._switch_effort,
            )
        return self._commands

    async def _handle_compact(self, args: list[str]) -> None:
        target_id = args[0] if args else self.runtime.thread_id
        await compact_session(self.console, self.runtime, target_id)

    async def _handle_session_resume(self, session_id: str) -> None:
        resolved = resume_session(self.console, session_id)
        if resolved:
            self.runtime.thread_id = resolved

    async def _handle_session_export(self, args: list[str]) -> None:
        out_path = args[0] if args else None
        await export_session(self.console, self.runtime, self.runtime.thread_id, output_path=out_path)

    async def cleanup(self) -> None:
        if self._rag_ctx:
            self._rag_ctx.rag_manager.unload()
        await self.runtime.aclose()

    async def run(self) -> None:
        rag_ctx = self._get_rag_ctx()
        if self._initial_rag_database:
            try:
                load_rag_database(rag_ctx, self._initial_rag_database)
            except SystemExit:
                pass

        set_tool_timeout(self.runtime.settings.runtime.builtin_tool_timeout)
        await self.runtime.reload()

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

    async def _handle_new_session(self, args: list[str]) -> None:
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
                self.console.print(f"[red]{_('Usage: /yolo [on|off]')}[/red]")
                return
        else:
            self.runtime.yolo_mode = not self.runtime.yolo_mode

        status = _("on") if self.runtime.yolo_mode else _("off")
        color = "red" if self.runtime.yolo_mode else "green"
        self.console.print(f"[bold {color}]{_('YOLO mode is now {status}', status=status)}[/bold {color}]")
