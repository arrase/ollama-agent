"""Main Textual application representing the agent's interactive TUI."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.types import Command
from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker

from ...i18n import _
from ...streaming import extract_action_requests, stream_agent_events
from ..clipboard import ClipboardError, copy_to_system_clipboard, get_system_clipboard
from ..commands.sessions import get_available_sessions
from .commands import _is_immediate_command, run_slash_command
from .completion import (
    CMD_CLEAR,
    CMD_CONTEXT,
    CMD_EFFORT,
    CMD_EXIT,
    CMD_MODEL,
    CMD_NEW,
    CMD_QUEUE,
    CMD_QUIT,
    CMD_RAG,
    CMD_SESSION,
    CMD_SKILL,
    CMD_TASK,
    _AT_MENTION_RE,
    _AUTOCOMPLETE_LIST_ID,
    _CHAT_SCROLL_ID,
    _collect_dir_candidates,
    _complete_effort,
    _complete_root_commands,
    _complete_subcommands,
    _iter_dir_completions,
    _iter_file_completions,
    _list_models_sync,
)
from .renderer import _TUIStreamingRenderer
from .screens import MainScreen
from .widgets import (
    AgentFooter,
    AgentHeader,
    AgentResponse,
    PromptQueueWidget,
    ReplInput,
    SystemOutputWidget,
    ToolApprovalWidget,
    UserMessage,
)

if TYPE_CHECKING:
    from .repl import OllamaREPL


class OllamaAgentApp(App):
    """Main Textual Application representing the Agent's interactive TUI."""

    BINDINGS = [
        ("escape", "cancel_generation", _("Interrupt")),
        ("ctrl+c", "cancel_or_quit", _("Interrupt/Quit")),
        ("super+c", "copy_selection", _("Copy")),
        ("ctrl+shift+c", "copy_selection", _("Copy")),
        ("ctrl+insert", "copy_selection", _("Copy")),
    ]

    CSS_PATH = Path(__file__).parent / "repl.tcss"

    _is_generating: bool
    _is_approval_pending: bool

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
        except (OSError, RuntimeError):
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
        except OSError:
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

    def update_autocomplete(self, text: str) -> None:
        autolist = self.query_one(_AUTOCOMPLETE_LIST_ID, OptionList)

        # 1. File completions via @-mention
        match = _AT_MENTION_RE.search(text)
        if match:
            token = match.group(1) or match.group(2) or match.group(3) or ""
            items = list(self._file_completions(token))
            if items:
                autolist.clear_options()
                for insert_val, display_markup in items:
                    autolist.add_option(Option(prompt=display_markup, id=insert_val))
                autolist.display = True
                autolist.highlighted = 0
                return

        # 2. Slash command completions
        if text.startswith("/"):
            items = self._slash_completions(text)
            if items:
                autolist.clear_options()
                for insert_val, display_text in items:
                    autolist.add_option(Option(prompt=display_text, id=insert_val))
                autolist.display = True
                autolist.highlighted = 0
                return

        self.hide_autocomplete()

    def accept_completion(self, option_index: int) -> None:
        autolist = self.query_one(_AUTOCOMPLETE_LIST_ID, OptionList)
        opt = autolist.get_option_at_index(option_index)
        if not opt or not opt.id:
            return

        inp = self.query_one(ReplInput)
        text = inp.text

        match = _AT_MENTION_RE.search(text)
        if match:
            # File mention completion
            start = match.start()
            # Preserve character before @ if it was part of boundary match
            prefix_char = text[start] if text[start] != "@" else ""
            before = text[:start] + prefix_char + "@"
            quote = '"' if " " in opt.id else ""
            inp.text = before + quote + opt.id + quote + " "
        else:
            # Slash command completion
            trailing = "" if opt.id in (CMD_CLEAR, CMD_NEW, CMD_EXIT, CMD_QUIT) else " "
            inp.text = opt.id + trailing

        inp.action_cursor_line_end()
        self.hide_autocomplete()

    def _file_completions(self, token: str, max_completions: int = 15) -> Iterator[tuple[str, Text]]:
        cwd = Path.cwd()
        token_path = Path(token)
        if token.endswith(("/", "\\")):
            search_dir = (cwd / token_path).resolve()
            prefix = ""
        else:
            search_dir = (cwd / token_path.parent).resolve()
            prefix = token_path.name

        try:
            search_dir.relative_to(cwd)
        except ValueError:
            return

        if not search_dir.is_dir():
            return

        show_hidden = prefix.startswith(".")
        prefix_lower = prefix.lower()
        count = 0

        for root, dirs, files in os.walk(search_dir):
            root_path = Path(root)
            candidate_dirs = _collect_dir_candidates(dirs, root_path, cwd, prefix_lower, show_hidden)
            dirs[:] = [d[0] for d in candidate_dirs]

            for rel, meta in _iter_dir_completions(candidate_dirs, prefix_lower):
                item = (
                    rel,
                    Text.from_markup(f"[bold #38bdf8]{rel:<30}[/bold #38bdf8] [dim #8b949e]{meta}[/dim #8b949e]"),
                )
                if count >= max_completions:
                    return
                count += 1
                yield item

            for rel, meta in _iter_file_completions(files, root_path, cwd, prefix_lower, show_hidden):
                item = (
                    rel,
                    Text.from_markup(f"[bold #38bdf8]{rel:<30}[/bold #38bdf8] [dim #8b949e]{meta}[/dim #8b949e]"),
                )
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

    async def _run_slash_command(self, cmd_line: str) -> None:
        await run_slash_command(self, cmd_line)

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
