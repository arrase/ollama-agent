from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from textual.app import App, ComposeResult
from textual.worker import Worker
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import Static, OptionList, TextArea
from textual.widgets.option_list import Option
from langgraph.types import Command

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
from ..tasks.commands import CLIContext, create_task
from .dispatch import REPLCommand, build_repl_handlers, render_repl_help, safe_call
from .model_commands import set_model
from .session_commands import new_session
from ..streaming import stream_agent_events, StreamingRenderer

# @-mention regex: matches @"quoted", @'quoted', or @bare at word boundaries.
_AT_MENTION_RE = re.compile(
    r"""(?:^|[\s\(\[\{<])@(?:"([^"]*)|'([^']*)|([^\s"'\(\[\{<>,;]*))$"""
)

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
        self.widget.add_error(event["content"])
        self._scroll()

    def on_warning(self, event: dict[str, Any]) -> None:
        self.widget.add_warning(event["content"])
        self._scroll()

    def close(self) -> None:
        self._timer.stop()
        self.widget.finish_generation()
        self._do_scroll()


# ─── Header ──────────────────────────────────────────────────────────────────


# ─── Custom Widgets and Dialogs imported from .tui_components ─────────────────


# ─── Main TUI App ────────────────────────────────────────────────────────────
class OllamaAgentApp(App):
    """Main Textual Application representing the Agent's interactive TUI."""

    BINDINGS = [
        ("escape", "cancel_generation", "Interrupt"),
        ("ctrl+c", "cancel_or_quit", "Interrupt/Quit"),
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
        with Container(id="input-container"):
            with Horizontal(id="input-bar"):
                yield Static("❯ ", id="prompt-char")
                yield ReplInput(id="repl-input")
        yield OptionList(id="autocomplete-list")
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
        autolist.display = False
        autolist.highlighted = None

    def update_autocomplete(self, value: str) -> None:
        autolist = self.query_one("#autocomplete-list", OptionList)
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
                    autolist.add_option(
                        Option(
                            prompt=Text.from_markup(f"[bold #38bdf8]{name:<16}[/bold #38bdf8] [dim #8b949e]{summary}[/dim #8b949e]"),
                            id=name,
                        )
                    )
                autolist.highlighted = 0
                autolist.display = True
                return

        # 2. File @-mention candidates
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
        tree = os.walk(cwd)
        max_dirs_visited = 50
        dirs_visited = 0

        for root, dirs, files in tree:
            dirs_visited += 1
            if dirs_visited > max_dirs_visited:

                return

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

        if cmd in ("/exit", "/quit"):
            self.exit()
            return

        if cmd == "/clear":
            scroll.query("*").remove()
            return

        if cmd == "/task-create":
            self._push_task_modal(args[0] if args else "", "--force" in args)
            return

        if cmd == "/skill-create":
            self._push_skill_modal(args[0] if args else "", "--force" in args)
            return

        if cmd == "/task-run":
            if not args:
                scroll.mount(SystemMessage("[bold #f87171]✕ Usage:[/bold #f87171] /task-run <id>"))
                self._deferred_scroll()
                return
            try:
                tid, t = self.repl._task_ctx._find_or_exit(args[0])
            except SystemExit:
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
            await self.repl.runtime.reload()
            await self._stream_chat(t.prompt, scroll, agent_msg)
            return

        commands = self.repl._get_commands()
        if cmd not in commands:
            scroll.mount(SystemMessage(f"[bold #f87171]✕ Unknown command:[/bold #f87171] {cmd}"))
            self._deferred_scroll()
            return

        spec = commands[cmd]
        if spec.usage and not args:
            scroll.mount(SystemMessage(f"[bold #f87171]✕ Usage:[/bold #f87171] {spec.usage}"))
            self._deferred_scroll()
            return

        with self.repl.console.capture() as capture:
            await safe_call(spec.handler, args)
        output = capture.get()
        if output:
            scroll.mount(SystemMessage(Text.from_ansi(output)))
            self._deferred_scroll()

        if cmd == "/yolo":
            self.update_yolo_ui()

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
                scroll.mount(SystemMessage(f"[bold #f87171]✕ Error checking state:[/bold #f87171] [red]{e}[/red]"))
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

        app = OllamaAgentApp(self)
        try:
            await app.run_async()
        except KeyboardInterrupt:
            pass

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
