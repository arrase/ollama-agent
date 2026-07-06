"""REPL interface for Ollama Agent using Textual TUI."""

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from textual.app import App, ComposeResult
from textual import events
from textual.message import Message
from textual.worker import Worker
from textual.containers import Grid, Container, ScrollableContainer, Horizontal
from textual.widgets import Static, OptionList, TextArea
from textual.widgets.option_list import Option
from textual.screen import ModalScreen
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .tui_components import (
    AgentHeader,
    ReplInput,
    UserMessage,
    AgentResponse,
    ToolCallMessage,
    ToolOutputMessage,
    SystemMessage,
    Label,
    TaskCreateModal,
    SkillCreateModal,
    ToolApprovalWidget,
)

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager, set_tool_timeout
from ..rag import RAGContext, RAGManager, load_rag_database
from ..settings.paths import APP_DIR, HISTORY_DB_PATH
from ..skills import SkillsContext, create_skill
from ..tasks.commands import CLIContext, create_task
from .dispatch import build_repl_handlers, render_repl_help
from .model_commands import set_model
from .session_commands import new_session
from .repl_wizards import safe_call
from ..streaming import stream_agent_events, StreamingRenderer

# @-mention regex: matches @"quoted", @'quoted', or @bare at word boundaries.
_AT_MENTION_RE = re.compile(
    r"""(?:^|[\s\(\[\{<])@(?:"([^"]*)|'([^']*)|([^\s"'\(\[\{<>,;]*))$"""
)


# ─── Header ──────────────────────────────────────────────────────────────────


# ─── Custom Widgets and Dialogs imported from .tui_components ─────────────────


# ─── Main TUI App ────────────────────────────────────────────────────────────
class OllamaAgentApp(App):
    """Main Textual Application representing the Agent's interactive TUI."""

    BINDINGS = [
        ("escape", "cancel_generation", "Interrumpir"),
        ("ctrl+c", "cancel_or_quit", "Interrumpir/Salir"),
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

    CSS_PATH = "repl.css"

    def __init__(self, repl: "OllamaREPL"):
        super().__init__()
        self.repl = repl
        self._is_generating = False
        self._current_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield AgentHeader(self.repl)
        yield ScrollableContainer(id="chat-scroll")
        with Horizontal(id="input-bar"):
            yield Static("❯❯ ", id="prompt-char")
            yield ReplInput(id="repl-input")
        yield OptionList(id="autocomplete-list")

    def on_mount(self) -> None:
        self.query_one(ReplInput).focus()
        self.update_yolo_ui()

    def update_yolo_ui(self) -> None:
        prompt_char = self.query_one("#prompt-char")
        if self.repl.runtime.yolo_mode:
            prompt_char.styles.color = "#f38ba8"  # Red
        else:
            prompt_char.styles.color = "#89b4fa"  # Blue

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
        if getattr(self, "_is_generating", False):
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
        autolist = self.query_one("#autocomplete-list")
        autolist.display = False
        autolist.highlighted = None

    def update_autocomplete(self, value: str) -> None:
        autolist = self.query_one("#autocomplete-list")
        text = value.lstrip()

        # 1. Slash-command candidates
        if text.startswith("/") and " " not in text:
            candidates = [
                (name, spec.summary)
                for name, spec in self.repl._get_commands().items()
                if name.startswith(text)
            ]
            if candidates:
                autolist.clear_options()
                for name, summary in candidates:
                    autolist.add_option(Option(prompt=f"{name:<16}{summary}", id=name))
                autolist.highlighted = 0
                autolist.display = True
                return

        # 2. File @-mention candidates
        match = _AT_MENTION_RE.search(value)
        if match:
            prefix = match.group(1) if match.group(1) is not None else (
                match.group(2) if match.group(2) is not None else match.group(3)
            )
            completions = list(self._file_completions(prefix))
            if completions:
                autolist.clear_options()
                for rel_path, meta in completions[:20]:
                    autolist.add_option(Option(prompt=f"{rel_path:<42}{meta}", id=rel_path))
                autolist.highlighted = 0
                autolist.display = True
                return

        self.hide_autocomplete()

    def accept_completion(self, option_index: int) -> None:
        if option_index is None or option_index < 0:
            return

        autolist = self.query_one("#autocomplete-list")
        option = autolist.get_option_at_index(option_index)
        completed_text = option.id

        inp = self.query_one(ReplInput)
        val = inp.text

        if val.lstrip().startswith("/"):
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

    def _file_completions(self, prefix: str):
        cwd = Path.cwd()
        show_hidden = prefix.startswith(".")
        count = 0
        max_completions = 100

        try:
            tree = os.walk(cwd)
        except OSError:
            return

        for root, dirs, files in tree:
            root_path = Path(root)
            candidate_dirs = []
            for dirname in sorted(dirs):
                if not show_hidden and dirname.startswith("."):
                    continue
                try:
                    rel = str((root_path / dirname).relative_to(cwd)) + "/"
                except ValueError:
                    continue
                if prefix.startswith(rel) or rel.startswith(prefix):
                    candidate_dirs.append((dirname, rel))

            dirs[:] = [d for d, _ in candidate_dirs]

            for _, rel in candidate_dirs:
                if count >= max_completions:
                    return
                if rel == prefix or not rel.startswith(prefix):
                    continue
                count += 1
                yield rel, "dir"

            for filename in sorted(files):
                if count >= max_completions:
                    return
                if not show_hidden and filename.startswith("."):
                    continue
                try:
                    rel = str((root_path / filename).relative_to(cwd))
                except ValueError:
                    continue
                if not rel.startswith(prefix):
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

    # ── Slash command dispatch ────────────────────────────────────────────

    async def _run_slash_command(self, cmd_line: str):
        parts = cmd_line.split()
        cmd = parts[0].lower()
        args = parts[1:]
        scroll = self.query_one("#chat-scroll")

        if cmd == "/yolo":
            if args:
                val = args[0].lower()
                if val in ("on", "true", "yes", "1"):
                    self.repl.runtime.yolo_mode = True
                elif val in ("off", "false", "no", "0"):
                    self.repl.runtime.yolo_mode = False
                else:
                    scroll.mount(SystemMessage("[red]Usage: /yolo [on|off][/red]"))
                    self._deferred_scroll()
                    return
            else:
                self.repl.runtime.yolo_mode = not self.repl.runtime.yolo_mode

            status = "ON" if self.repl.runtime.yolo_mode else "OFF"
            color = "red" if self.repl.runtime.yolo_mode else "green"
            scroll.mount(SystemMessage(f"[bold {color}]YOLO mode is now {status}[/bold {color}]"))
            self._deferred_scroll()
            self.update_yolo_ui()
            return

        if cmd in ("/exit", "/quit"):
            self.exit()
            return

        if cmd == "/clear":
            scroll.query("*").remove()
            return

        if cmd == "/help":
            commands = self.repl._get_commands()
            with self.repl.console.capture() as capture:
                render_repl_help(self.repl.console, commands)
            output = capture.get()
            if output:
                scroll.mount(SystemMessage(Text.from_ansi(output)))
                self._deferred_scroll()
            return

        if cmd == "/new":
            scroll.query("*").remove()
            commands = self.repl._get_commands()
            spec = commands.get("/new")
            if spec:
                with self.repl.console.capture() as capture:
                    await safe_call(spec.handler, args)
                output = capture.get()
                if output:
                    scroll.mount(SystemMessage(Text.from_ansi(output)))
                    self._deferred_scroll()
            return

        if cmd == "/task-create":
            self._push_task_modal(args[0] if args else "", "--force" in args)
            return

        if cmd == "/skill-create":
            self._push_skill_modal(args[0] if args else "", "--force" in args)
            return

        if cmd == "/task-run":
            if not args:
                scroll.mount(SystemMessage("[red]Usage: /task-run <id>[/red]"))
                self._deferred_scroll()
                return
            try:
                tid, t = self.repl._task_ctx._find_or_exit(args[0])
            except SystemExit:
                return

            scroll.mount(SystemMessage(Text.from_ansi(
                f"[bold cyan]▶ Executing:[/bold cyan] {t.title} ({tid})\n"
                f"  Model: {t.model} │ Effort: {t.reasoning_effort}"
            )))
            agent_msg = AgentResponse()
            scroll.mount(agent_msg)
            self._deferred_scroll()

            self.repl.runtime.settings.model.name = t.model
            self.repl.runtime.settings.model.reasoning_effort = t.reasoning_effort
            await self.repl.runtime.reload()
            await self._stream_chat(t.prompt, scroll, agent_msg)
            return

        commands = self.repl._get_commands()
        if cmd not in commands:
            scroll.mount(SystemMessage(f"[red]Unknown command: {cmd}[/red]"))
            self._deferred_scroll()
            return

        spec = commands[cmd]
        if spec.usage and not args:
            scroll.mount(SystemMessage(f"[red]Usage: {spec.usage}[/red]"))
            self._deferred_scroll()
            return

        with self.repl.console.capture() as capture:
            await safe_call(spec.handler, args)
        output = capture.get()
        if output:
            scroll.mount(SystemMessage(Text.from_ansi(output)))
            self._deferred_scroll()

    # ── Modal helpers ─────────────────────────────────────────────────────

    def _push_task_modal(self, task_id: str, force: bool):
        def on_dismiss(result):
            if result:
                tid, title, model, effort, prompt = result
                self.run_worker(self._do_create_task(tid, title, model, effort, prompt, force))
        self.push_screen(TaskCreateModal(self, task_id, force), on_dismiss)

    async def _do_create_task(self, task_id, title, model, effort, prompt, force):
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

    def _push_skill_modal(self, skill_id: str, force: bool):
        def on_dismiss(result):
            if result:
                sid, name, description, instructions = result
                self.run_worker(self._do_create_skill(sid, name, description, instructions, force))
        self.push_screen(SkillCreateModal(self, skill_id, force), on_dismiss)

    async def _do_create_skill(self, skill_id, name, description, instructions, force):
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
 
    async def _stream_chat(self, prompt: str, scroll, agent_msg: AgentResponse):
        await self._run_stream(prompt, scroll, agent_msg)

    async def _handle_approval_decision(self, decisions: list[dict], scroll, agent_msg: AgentResponse):
        command = Command(resume={"decisions": decisions})
        await self._run_stream(command, scroll, agent_msg)

    async def _run_stream(self, prompt: str | Command, scroll, agent_msg: AgentResponse):
        self._is_generating = True
        app = self

        class _Renderer(StreamingRenderer):
            def __init__(self, widget: AgentResponse):
                self.widget = widget
                self._auto_scroll = True
                self._last_scroll_y = scroll.scroll_y
                self._last_max_scroll_y = scroll.max_scroll_y
                self._timer = app.set_interval(0.1, self._do_scroll)

            def _do_scroll(self) -> None:
                if self._auto_scroll:
                    scroll.scroll_end(animate=False)
                    self._last_scroll_y = scroll.scroll_y
                    self._last_max_scroll_y = scroll.max_scroll_y

            def _scroll(self) -> None:
                # If scroll_y decreased but max_scroll_y didn't drop, the user scrolled up.
                if scroll.scroll_y < self._last_scroll_y and scroll.max_scroll_y >= self._last_max_scroll_y:
                    self._auto_scroll = False
                elif scroll.scroll_y >= scroll.max_scroll_y - 4:
                    self._auto_scroll = True
                self._last_scroll_y = scroll.scroll_y
                self._last_max_scroll_y = scroll.max_scroll_y

            def on_text_delta(self, event: dict[str, Any]) -> None:
                self.widget.append_text(event.get("content", ""))
                self._scroll()

            def on_reasoning_delta(self, event: dict[str, Any]) -> None:
                self.widget.append_thinking(event.get("content", ""))
                self._scroll()

            def on_tool_call(self, event: dict[str, Any]) -> None:
                self.widget.add_tool_call(
                    name=event.get("name", "unknown"),
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
                self.widget.add_error(event.get("content", "Unknown error"))
                self._scroll()

            def on_warning(self, event: dict[str, Any]) -> None:
                self.widget.add_warning(event.get("content", "Unknown warning"))
                self._scroll()

            def close(self) -> None:
                self._timer.stop()
                self.widget.flush_text()
                self._do_scroll()

        try:
            try:
                await stream_agent_events(self.repl.runtime, prompt, _Renderer(agent_msg), auto_close=True)
            except asyncio.CancelledError:
                scroll.mount(SystemMessage("[red]🛑 Execution interrupted by user.[/red]"))
                self._deferred_scroll()
                self.query_one(ReplInput).focus()
                raise
            except Exception as e:
                scroll.mount(SystemMessage(f"[red]Error: {e}[/red]"))
                self._deferred_scroll()
                return

            # Check if the execution got interrupted
            config = {"configurable": {"thread_id": self.repl.runtime.thread_id}}
            try:
                state = await self.repl.runtime.graph.aget_state(config)
                if state.interrupts:
                    interrupt_val = state.interrupts[0].value
                    action_requests = interrupt_val.get("action_requests", [])

                    # Mount approval widget
                    approval_widget = ToolApprovalWidget(
                        action_requests=action_requests,
                        app_ref=self,
                        scroll=scroll,
                        agent_msg=agent_msg,
                    )
                    agent_msg.mount(approval_widget)
                    self._deferred_scroll()
            except Exception as e:
                scroll.mount(SystemMessage(f"[red]Error checking state: {e}[/red]"))
                self._deferred_scroll()
        finally:
            self._is_generating = False


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
        self._commands: dict | None = None

    def _get_rag_ctx(self) -> RAGContext:
        if self._rag_ctx is None:
            mgr = RAGManager(self.runtime.settings.rag)
            self._rag_ctx = RAGContext(console=self.console, rag_manager=mgr)
            set_rag_manager(mgr)
        return self._rag_ctx

    def _get_commands(self) -> dict:
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
            )
        return self._commands

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
        self.runtime.preload()

        app = OllamaAgentApp(self)
        try:
            await app.run_async()
        except KeyboardInterrupt:
            pass

    async def _switch_model(self, model_name: str) -> None:
        await set_model(self.console, model_name, runtime=self.runtime)
        self.runtime.preload()

    async def _handle_new_session(self, args: list[str]) -> None:
        self.runtime.thread_id = new_session(self.console)
