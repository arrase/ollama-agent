from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from textual import events
from textual.app import App, ComposeResult
from textual.worker import Worker
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import Static, OptionList, TextArea
from textual.widgets.option_list import Option
from deepagents.middleware.summarization import count_tokens_approximately
from langgraph.types import Command

from .clipboard import copy_to_system_clipboard, get_system_clipboard
from .tui_components import (
    AgentFooter,
    AgentHeader,
    ReplInput,
    UserMessage,
    AgentResponse,
    ToolCallMessage,
    ToolOutputMessage,
    SystemMessage,
    TaskCreateModal,
    SkillCreateModal,
    ToolApprovalWidget,
)

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager, set_tool_timeout
from ..rag import RAGContext, RAGManager, load_rag_database
from ..skills import SkillsContext, create_skill
from ..core.common import extract_text
from ..tasks.commands import CLIContext, TaskError, create_task
from .dispatch import REPLCommand, build_repl_handlers, render_repl_help, safe_call
from .model_commands import set_model
from .session_commands import (
    compact_session,
    delete_session,
    export_session,
    get_available_sessions,
    list_sessions,
    new_session,
    resume_session,
)
from ..streaming import stream_agent_events, StreamingRenderer

# @-mention regex: matches @"quoted", @'quoted', or @bare at word boundaries.
_AT_MENTION_RE = re.compile(
    r"""(?:^|[\s\(\[\{<])@(?:"([^"]*)|'([^']*)|([^\s"'\(\[\{<>,;]*))$"""
)

_ROOT_COMMANDS: list[tuple[str, str]] = [
    ("/model", "Manage models (list, set)"),
    ("/session", "Manage chat sessions (list, resume, new, export, delete)"),
    ("/compact", "Compact conversation history into a summary"),
    ("/task", "Manage saved tasks (list, create, run, delete)"),
    ("/skill", "Manage skills (list, show, create, delete)"),
    ("/rag", "Manage RAG databases (status, list, create, delete, load, unload, add)"),
    ("/yolo", "Toggle YOLO mode (on/off)"),
    ("/new", "Start a new chat session"),
    ("/clear", "Clear the screen"),
    ("/help", "Show help message"),
    ("/exit", "Exit the REPL"),
]

_SUBCOMMANDS: dict[str, list[tuple[str, str]]] = {
    "/model": [
        ("list", "List available Ollama models"),
        ("set", "Switch to a different model"),
    ],
    "/session": [
        ("list", "List all past sessions"),
        ("resume", "Resume a previous session"),
        ("new", "Start a new session"),
        ("export", "Export session to Markdown"),
        ("delete", "Delete a session from history"),
    ],
    "/task": [
        ("list", "List all saved tasks"),
        ("create", "Create a task using interactive modal"),
        ("run", "Run a saved task prompt"),
        ("delete", "Delete a saved task"),
    ],
    "/skill": [
        ("list", "List all available skills"),
        ("show", "Show skill details and instructions"),
        ("create", "Create a skill using interactive modal"),
        ("delete", "Delete a skill"),
    ],
    "/rag": [
        ("status", "Show current RAG database status"),
        ("list", "List all RAG databases"),
        ("create", "Create a new RAG database"),
        ("delete", "Delete a RAG database"),
        ("load", "Load a RAG database"),
        ("unload", "Unload active RAG database"),
        ("add", "Add file or directory to RAG"),
    ],
    "/yolo": [
        ("on", "Enable YOLO mode (bypasses confirmations)"),
        ("off", "Disable YOLO mode"),
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
        ("escape", "cancel_generation", "Interrupt"),
        ("ctrl+c", "cancel_or_quit", "Interrupt/Quit"),
        ("super+c", "copy_selection", "Copy"),
        ("ctrl+shift+c", "copy_selection", "Copy"),
        ("ctrl+insert", "copy_selection", "Copy"),
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

        # Level 0: Root commands (e.g., "/" or "/mo")
        if num_parts == 1:
            token = parts[0]
            return [
                (
                    cmd,
                    Text.from_markup(f"[bold #38bdf8]{cmd:<12}[/bold #38bdf8] [dim #8b949e]{desc}[/dim #8b949e]"),
                )
                for cmd, desc in _ROOT_COMMANDS
                if cmd.startswith(token)
            ]

        root_cmd = parts[0]
        if root_cmd not in _SUBCOMMANDS:
            return []

        # Level 1: Subcommands (e.g., "/task " or "/task r")
        if num_parts == 2:
            sub_token = parts[1]
            return [
                (
                    f"{root_cmd} {sub}",
                    Text.from_markup(f"[bold #38bdf8]{sub:<12}[/bold #38bdf8] [dim #8b949e]{desc}[/dim #8b949e]"),
                )
                for sub, desc in _SUBCOMMANDS[root_cmd]
                if sub.startswith(sub_token)
            ]

        # Level 2: Arguments / Dynamic entities (e.g., "/task run ")
        if num_parts == 3:
            sub_cmd = parts[1]
            arg_token = parts[2]

            if root_cmd == "/task" and sub_cmd in ("run", "delete"):
                tasks = self.repl._task_ctx.task_manager.list()
                return [
                    (
                        f"{root_cmd} {sub_cmd} {tid}",
                        Text.from_markup(f"[bold #e6edf3]{tid:<20}[/bold #e6edf3] [dim #8b949e]{t.title}[/dim #8b949e]"),
                    )
                    for tid, t in tasks.items()
                    if tid.startswith(arg_token)
                ]

            if root_cmd == "/skill" and sub_cmd in ("show", "delete"):
                skills = self.repl._skills_ctx.skill_manager.list()
                return [
                    (
                        f"{root_cmd} {sub_cmd} {sid}",
                        Text.from_markup(f"[bold #e6edf3]{sid:<20}[/bold #e6edf3] [dim #8b949e]{s.name}[/dim #8b949e]"),
                    )
                    for sid, s in skills.items()
                    if sid.startswith(arg_token)
                ]

            if root_cmd == "/session" and sub_cmd in ("resume", "switch", "delete"):
                sessions = get_available_sessions()
                return [
                    (
                        f"{root_cmd} {sub_cmd} {s['thread_id']}",
                        Text.from_markup(f"[bold #e6edf3]{s['thread_id'][:8]:<10}[/bold #e6edf3] [dim #8b949e]{s['steps']} steps[/dim #8b949e]"),
                    )
                    for s in sessions
                    if s["thread_id"].startswith(arg_token)
                ]

            if root_cmd == "/rag" and sub_cmd in ("load", "delete"):
                dbs = self.repl._get_rag_ctx().rag_manager.list_databases()
                return [
                    (
                        f"{root_cmd} {sub_cmd} {d['name']}",
                        Text.from_markup(f"[bold #e6edf3]{d['name']:<20}[/bold #e6edf3] [dim #8b949e]{d.get('doc_count', 0)} docs[/dim #8b949e]"),
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

            dirs[:] = [d for d, _ in candidate_dirs]

            for _, rel in candidate_dirs:
                if count >= max_completions:
                    return
                rel_lower = rel.lower()
                if rel_lower == prefix_lower or not rel_lower.startswith(prefix_lower):
                    continue
                count += 1
                yield rel, "dir"

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
                meta = "file"
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

        if cmd == "/clear":
            await scroll.remove_children()
            return

        if cmd in ("/compact", "/compress"):
            scroll.mount(SystemMessage("[dim]⚡ Compacting conversation context...[/dim]"))
            self._deferred_scroll()
            res = await self.repl.runtime.compact_context()
            if res["success"]:
                msg_text = (
                    f"[bold #38bdf8]✓ Context compacted successfully:[/bold #38bdf8]\n"
                    f"  • [dim]Messages summarized:[/dim] {res['messages_summarized']}\n"
                    f"  • [dim]Recent messages preserved:[/dim] {res['messages_preserved']}"
                )
                if res.get("file_path"):
                    msg_text += f"\n  • [dim]History offloaded to:[/dim] [cyan]{res['file_path']}[/cyan]"
                scroll.mount(SystemMessage(msg_text))
                self.query_one(AgentHeader).update_header()
            else:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ Compaction skipped:[/bold #f87171] {res.get('message', 'Failed to compact.')}"))
            self._deferred_scroll()
            return

        if cmd == "/session" and args and args[0] in ("resume", "switch"):
            if len(args) < 2:
                scroll.mount(SystemMessage("[bold #f87171]✕ Usage:[/bold #f87171] /session resume <session_id>"))
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
                            role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
                            content = extract_text(getattr(msg, "content", ""))
                            if not content:
                                continue
                            if role in ("human", "user"):
                                scroll.mount(UserMessage(content))
                            elif role in ("ai", "assistant"):
                                scroll.mount(AgentResponse(initial_text=content))
                scroll.mount(SystemMessage(f"[bold #38bdf8]✓ Resumed session:[/bold #38bdf8] [bold #e6edf3]{resolved[:8]}[/bold #e6edf3] [dim]({resolved})[/dim]"))
                self.query_one(AgentHeader).update_header()
                self._deferred_scroll()
            else:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ Session not found:[/bold #f87171] {args[1]}"))
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
                scroll.mount(SystemMessage(f"[bold #38bdf8]✓ Session exported to:[/bold #38bdf8] [bold #e6edf3]{out_file}[/bold #e6edf3]"))
            else:
                scroll.mount(SystemMessage("[bold #f87171]✕ Failed to export session.[/bold #f87171]"))
            self._deferred_scroll()
            return

        if cmd == "/session" and args and args[0] == "new":
            self.repl.runtime.thread_id = new_session(self.repl.console)
            await scroll.remove_children()
            scroll.mount(SystemMessage(f"[bold #38bdf8]✓ New session started:[/bold #38bdf8] [bold #e6edf3]{self.repl.runtime.thread_id[:8]}[/bold #e6edf3]"))
            self.query_one(AgentHeader).update_header()
            self._deferred_scroll()
            return

        if cmd == "/task" and args and args[0] == "create":
            sub_args = args[1:]
            task_id = sub_args[0] if sub_args and not sub_args[0].startswith("-") else ""
            self._push_task_modal(task_id, "--force" in sub_args or "-f" in sub_args)
            return

        if cmd == "/skill" and args and args[0] == "create":
            sub_args = args[1:]
            skill_id = sub_args[0] if sub_args and not sub_args[0].startswith("-") else ""
            self._push_skill_modal(skill_id, "--force" in sub_args or "-f" in sub_args)
            return

        if cmd == "/task" and args and args[0] == "run":
            sub_args = args[1:]
            target_id = next((a for a in sub_args if not a.startswith("-")), "")
            if not target_id:
                scroll.mount(SystemMessage("[bold #f87171]✕ Usage:[/bold #f87171] /task run <id> [-y]"))
                self._deferred_scroll()
                return
            try:
                tid, t = self.repl._task_ctx._find_or_exit(target_id)
            except (TaskError, SystemExit) as exc:
                scroll.mount(SystemMessage(f"[red]{exc}[/red]"))
                self._deferred_scroll()
                return

            scroll.mount(SystemMessage(Text.from_markup(
                f"[bold #38bdf8]▶ Executing Task:[/bold #38bdf8] [bold #e6edf3]{t.title}[/bold #e6edf3] [dim]({tid})[/dim]\n"
                f"  [dim]model:[/dim] {t.model} [dim]·[/dim] [dim]effort:[/dim] {t.reasoning_effort}"
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
            scroll.mount(SystemMessage(f"[bold #f87171]✕ Unknown command:[/bold #f87171] {cmd}"))
            self._deferred_scroll()
            return

        spec = commands[cmd]
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

    # ── Modal helpers ─────────────────────────────────────────────────────

    def _push_task_modal(self, task_id: str, force: bool) -> None:
        def on_dismiss(result: tuple[str, str, str, str, str] | None) -> None:
            if result:
                tid, title, model, effort, prompt = result
                self.run_worker(self._do_create_task(tid, title, model, effort, prompt, force))
        self.push_screen(TaskCreateModal(self, task_id, force), on_dismiss)

    async def _do_create_task(self, task_id: str, title: str, model: str, effort: str, prompt: str, force: bool) -> None:
        scroll = self.query_one("#chat-scroll")
        with self.repl.console.capture() as capture:
            await safe_call(
                create_task, self.repl._task_ctx, task_id,
                title=title, prompt=prompt, model=model,
                reasoning_effort=effort, force=force,
            )
        output = capture.get()
        if output:
            scroll.mount(SystemMessage(Text.from_ansi(output)))
            self._deferred_scroll()

    def _push_skill_modal(self, skill_id: str, force: bool) -> None:
        def on_dismiss(result: tuple[str, str, str, str] | None) -> None:
            if result:
                sid, name, description, instructions = result
                self.run_worker(self._do_create_skill(sid, name, description, instructions, force))
        self.push_screen(SkillCreateModal(self, skill_id, force), on_dismiss)

    async def _do_create_skill(self, skill_id: str, name: str, description: str, instructions: str, force: bool) -> None:
        scroll = self.query_one("#chat-scroll")
        with self.repl.console.capture() as capture:
            await safe_call(
                create_skill, self.repl._skills_ctx, skill_id,
                name=name, description=description,
                instructions=instructions, force=force,
            )
        output = capture.get()
        if output:
            scroll.mount(SystemMessage(Text.from_ansi(output)))
            self._deferred_scroll()

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
                scroll.mount(SystemMessage("[bold #f87171]🛑 Execution interrupted by user.[/bold #f87171]"))
                self._deferred_scroll()
                self.query_one(ReplInput).focus()
                raise
            except Exception as e:
                scroll.mount(SystemMessage(f"[bold #f87171]✕ Error:[/bold #f87171] [red]{e}[/red]"))
                self._deferred_scroll()
                return

            # Check if the execution got interrupted
            config = {"configurable": {"thread_id": self.repl.runtime.thread_id}}
            state = await self.repl.runtime.graph.aget_state(config)
            if state.interrupts:
                interrupt_val = state.interrupts[0].value
                action_requests = interrupt_val.get("action_requests", [])
                if action_requests:
                    approval_widget = ToolApprovalWidget(
                        action_requests=action_requests,
                        app_ref=self,
                        scroll=scroll,
                        agent_msg=agent_msg,
                    )
                    agent_msg.mount(approval_widget)
                    self._deferred_scroll()
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
                handle_exit=lambda _: None,
                handle_clear=lambda _: None,
                handle_new=self._handle_new_session,
                handle_task_create=lambda _: None,
                handle_skill_create=lambda _: None,
                handle_yolo=self._handle_yolo_cmd,
                current_thread_id=lambda: self.runtime.thread_id,
                handle_session_resume=self._handle_session_resume,
                handle_session_export=self._handle_session_export,
                handle_compact=self._handle_compact,
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

    async def _handle_new_session(self, args: list[str]) -> None:
        self.runtime.thread_id = new_session(self.console)

    def _handle_yolo_cmd(self, args: list[str]) -> None:
        if args:
            val = args[0].lower()
            if val in ("on", "true", "yes", "1"):
                self.runtime.yolo_mode = True
            elif val in ("off", "false", "no", "0"):
                self.runtime.yolo_mode = False
            else:
                self.console.print("[red]Usage: /yolo [on|off][/red]")
                return
        else:
            self.runtime.yolo_mode = not self.runtime.yolo_mode

        status = "ON" if self.runtime.yolo_mode else "OFF"
        color = "red" if self.runtime.yolo_mode else "green"
        self.console.print(f"[bold {color}]YOLO mode is now {status}[/bold {color}]")
