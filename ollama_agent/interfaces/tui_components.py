from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Button, Collapsible, Markdown, OptionList, Static, TextArea

from ..agent.episodic_memory import HistoryError, load_past_user_prompts
from ..i18n import _

if TYPE_CHECKING:
    from .repl import OllamaREPL, OllamaAgentApp, QueuedItem

_log = logging.getLogger(__name__)


# ─── Header ──────────────────────────────────────────────────────────────────

class AgentHeader(Static):
    """Dynamic TUI Header displaying agent status information."""

    def __init__(self, repl: OllamaREPL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.repl = repl

    def on_mount(self) -> None:
        self.update_header()
        self.set_interval(2.0, self.update_header)

    def update_header(self) -> None:
        ms = self.repl.runtime.settings.model
        tokens = self.repl.runtime.last_context_tokens
        eff_ctx = getattr(self.repl.runtime, "effective_context_window", None)
        num_ctx = eff_ctx if isinstance(eff_ctx, int) and eff_ctx > 0 else (ms.context_window if isinstance(ms.context_window, int) else 0)

        if num_ctx > 0 and isinstance(tokens, (int, float)):
            tokens_val = tokens
            pct = int((tokens_val / num_ctx) * 100)
            if pct > 90:
                color = "#f87171"
            elif pct > 75:
                color = "#fbbf24"
            else:
                color = "#38bdf8"
            tok_str = f"{tokens_val / 1000:.1f}k" if tokens_val >= 1000 else str(int(tokens_val))
            ctx_str = f"{num_ctx / 1000:.1f}k" if num_ctx >= 1000 else str(int(num_ctx))
            ctx_info = f"  [dim]│[/dim]  [bold #8b949e]{_('Context:')}[/bold #8b949e] [bold {color}]{tok_str}/{ctx_str} ({pct}%)[/bold {color}]"
        else:
            ctx_info = ""

        rag_ctx = self.repl._rag_ctx
        rag_db = rag_ctx.rag_manager.current_database if rag_ctx else None
        rag_info = f"  [dim]│[/dim]  [bold #8b949e]{_('RAG:')}[/bold #8b949e] [bold #a78bfa]{escape(str(rag_db))}[/bold #a78bfa]" if rag_db else ""
        yolo_status = (
            f"[bold #f87171 on #3b181e] {_('YOLO: ON')} [/bold #f87171 on #3b181e]"
            if self.repl.runtime.yolo_mode
            else f"[dim #8b949e]{_('YOLO: OFF')}[/dim #8b949e]"
        )
        self.update(
            f"[bold #38bdf8]● ollama-agent[/bold #38bdf8]  [dim]│[/dim]  "
            f"[bold #8b949e]{_('Model:')}[/bold #8b949e] [bold #e6edf3]{escape(str(ms.name))}[/bold #e6edf3]{ctx_info}  [dim]│[/dim]  "
            f"[bold #8b949e]{_('Effort:')}[/bold #8b949e] [#e6edf3]{escape(str(ms.reasoning_effort))}[/#e6edf3]{rag_info}  [dim]│[/dim]  "
            f"{yolo_status}"
        )


# ─── Footer ──────────────────────────────────────────────────────────────────

class AgentFooter(Static):
    """Dynamic TUI Footer displaying keyboard shortcuts and live status."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._is_generating = False
        self._is_approval = False
        self._queued_count = 0

    def on_mount(self) -> None:
        self.update_footer()

    def set_generating(self, is_generating: bool) -> None:
        self._is_generating = is_generating
        self.update_footer()

    def set_approval(self, is_approval: bool) -> None:
        self._is_approval = is_approval
        self.update_footer()

    def set_queued_count(self, count: int) -> None:
        self._queued_count = count
        self.update_footer()

    def update_footer(self) -> None:
        queue_info = (
            f"  [dim]│[/dim]  [bold #38bdf8]⏳ {_('{count} queued', count=self._queued_count)}[/bold #38bdf8]"
            if self._queued_count > 0
            else ""
        )
        if self._is_approval:
            self.update(
                f"[bold #fbbf24]⚠ {_('Approval required:')}[/bold #fbbf24]   "
                f"[dim]y[/dim] [bold #8b949e]{_('approve')}[/bold #8b949e]   "
                f"[dim]n[/dim] [bold #8b949e]{_('reject')}[/bold #8b949e]   "
                f"[dim]a[/dim] [bold #8b949e]{_('allow session')}[/bold #8b949e]   "
                f"[dim]esc[/dim] [bold #8b949e]{_('cancel')}[/bold #8b949e]   "
                f"[dim]←→[/dim] [bold #8b949e]{_('select')}[/bold #8b949e]"
                f"{queue_info}"
            )
        elif self._is_generating:
            self.update(
                f"[bold #38bdf8]⟡ {_('Generating response...')}[/bold #38bdf8]   "
                f"[dim]{_('press esc or ^C to interrupt')}[/dim]"
                f"{queue_info}"
            )
        else:
            self.update(
                f"[dim]enter[/dim] [bold #8b949e]{_('send')}[/bold #8b949e]   "
                f"[dim]\\+enter[/dim] [bold #8b949e]{_('newline')}[/bold #8b949e]   "
                f"[dim]tab[/dim] [bold #8b949e]{_('complete')}[/bold #8b949e]   "
                f"[dim]↑↓[/dim] [bold #8b949e]{_('history')}[/bold #8b949e]   "
                f"[dim]esc[/dim] [bold #8b949e]{_('interrupt')}[/bold #8b949e]   "
                f"[dim]/[/dim] [bold #8b949e]{_('commands')}[/bold #8b949e]"
                f"{queue_info}"
            )


# ─── Prompt Queue Widget ─────────────────────────────────────────────────────

class PromptQueueWidget(Static):
    """Widget displaying currently queued prompts and commands."""

    can_focus = False

    def update_queue(self, queue: Sequence[QueuedItem]) -> None:
        if not queue:
            self.display = False
            return

        self.display = True
        count = len(queue)
        header = f"[bold #38bdf8]⏳ {_('Queued ({count})', count=count)}[/bold #38bdf8]"
        lines = [header]

        for i, item in enumerate(islice(queue, 3), 1):
            text = item.text.replace("\n", " ")
            if len(text) > 60:
                text = text[:57] + "..."
            lines.append(f"  [dim]#{i}[/dim] {escape(text)}")

        if count > 3:
            remaining = count - 3
            lines.append(f"  [dim]... +{remaining} {_('more')}[/dim]")

        self.update("\n".join(lines))


# ─── System Output Widget ────────────────────────────────────────────────────

class SystemOutputWidget(Static):
    """Dedicated TUI widget displaying system notifications and slash command outputs."""

    can_focus = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.display = False

    def show_output(self, content: str | Text, title: str | None = None) -> None:
        self.display = True
        display_title = title if title is not None else _("System Output")
        header = f"[bold #38bdf8]⚙ {escape(display_title) if isinstance(display_title, str) else display_title}[/bold #38bdf8]  [dim]({_('esc to dismiss')})[/dim]"
        if isinstance(content, Text):
            header_text = Text.from_markup(header + "\n")
            self.update(Text.assemble(header_text, content))
        else:
            self.update(f"{header}\n{content}")

    def show_notice(self, notice: str | Text) -> None:
        self.display = True
        self.update(notice)

    def clear_output(self) -> None:
        self.display = False
        self.update("")


# ─── Custom Input ─────────────────────────────────────────────────────────────

class ReplInput(TextArea):
    """Interactive input field that captures Tab/arrow keys for autocomplete."""

    BINDINGS = [
        ("ctrl+v", "paste", _("Paste")),
        ("super+v", "paste", _("Paste")),
        ("shift+insert", "paste", _("Paste")),
    ]

    def action_paste(self) -> None:
        """Paste text from clipboard."""
        if self.app.clipboard:
            self.insert(self.app.clipboard)

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("highlight_cursor_line", False)
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("placeholder", _("Ask anything, / for commands, @ for files..."))
        super().__init__(**kwargs)
        self._history: list[str] = []
        self._history_index: int = 0
        self._temp_input: str = ""

    def on_mount(self) -> None:
        super().on_mount()
        self._update_height()
        self.run_worker(self._load_history())

    def _update_height(self) -> None:
        lines = max(2, min(8, self.document.line_count))
        self.styles.height = lines

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_height()

    async def _load_history(self) -> None:
        try:
            db_entries = await asyncio.to_thread(load_past_user_prompts)
        except HistoryError as exc:
            _log.warning("Prompt history unavailable: %s", exc)
            if hasattr(self.app, "show_system_notice"):
                self.app.show_system_notice(f"[yellow]⚠ {_('Prompt history unavailable: {exc}', exc=exc)}[/yellow]")
            db_entries = []
        self._history = list(db_entries)
        self._history_index = len(self._history)
        self._temp_input = ""

    def add_history_entry(self, entry: str) -> None:
        if not entry or entry.startswith("/"):
            return
        if self._history and self._history[-1] == entry:
            self._history_index = len(self._history)
            self._temp_input = ""
            return
        self._history.append(entry)
        self._history_index = len(self._history)
        self._temp_input = ""

    class Submitted(Message):
        """Emitted when the user submits the input."""
        def __init__(self, input_widget: ReplInput, value: str) -> None:
            super().__init__()
            self.input = input_widget
            self.value = value

    def _handle_autocomplete_key(self, event: events.Key, app: Any, autolist: OptionList) -> bool:
        if not autolist.display or autolist.option_count == 0:
            return False

        if event.key == "down":
            event.stop()
            event.prevent_default()
            if autolist.highlighted is None:
                autolist.highlighted = 0
            elif autolist.highlighted < autolist.option_count - 1:
                autolist.highlighted += 1
            return True
        elif event.key == "up":
            event.stop()
            event.prevent_default()
            if autolist.highlighted is not None and autolist.highlighted > 0:
                autolist.highlighted -= 1
            return True
        elif event.key == "tab":
            event.stop()
            event.prevent_default()
            if autolist.highlighted is not None:
                app.accept_completion(autolist.highlighted)
            return True
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            app.hide_autocomplete()
            return True
        elif event.key == "enter":
            if autolist.highlighted is not None:
                event.stop()
                event.prevent_default()
                app.accept_completion(autolist.highlighted)
                return True
            return False
        return False

    def _handle_history_key(self, event: events.Key) -> bool:
        if event.key == "up":
            if self.document.line_count > 1:
                row, col = self.cursor_location
                if row > 0:
                    return False
                if col > 0:
                    self.action_cursor_line_start()
                    event.stop()
                    event.prevent_default()
                    return True
            event.stop()
            event.prevent_default()
            if self._history:
                if self._history_index == len(self._history):
                    self._temp_input = self.text
                if self._history_index > 0:
                    self._history_index -= 1
                    self.text = self._history[self._history_index]
                    self.action_cursor_line_end()
                    self._update_height()
            return True
        elif event.key == "down":
            if self.document.line_count > 1:
                row, col = self.cursor_location
                last_row = self.document.line_count - 1
                if row < last_row:
                    return False
                last_line_len = len(self.document.get_line(last_row))
                if col < last_line_len:
                    self.action_cursor_line_end()
                    event.stop()
                    event.prevent_default()
                    return True
            event.stop()
            event.prevent_default()
            if self._history:
                if self._history_index < len(self._history):
                    self._history_index += 1
                    if self._history_index == len(self._history):
                        self.text = self._temp_input
                    else:
                        self.text = self._history[self._history_index]
                    self.action_cursor_line_end()
                    self._update_height()
            return True
        return False

    def on_key(self, event: events.Key) -> None:
        app: Any = self.app
        autolist = app.query_one("#autocomplete-list", OptionList)
        if autolist.display:
            if self._handle_autocomplete_key(event, app, autolist):
                return
        else:
            if self._handle_history_key(event):
                return

        if event.key == "enter":
            row, col = self.cursor_location
            current_line = self.document.get_line(row)
            line_before_cursor = current_line[:col]
            if line_before_cursor.rstrip().endswith("\\"):
                event.stop()
                event.prevent_default()
                idx = line_before_cursor.rfind("\\")
                self.delete((row, idx), (row, col))
                self.insert("\n")
                self._update_height()
                return

            event.stop()
            event.prevent_default()
            val = self.text.strip()
            if val:
                self.post_message(self.Submitted(self, val))


# ─── Chat Message Widgets ─────────────────────────────────────────────────────

class UserMessage(Container):
    """Rendered user prompt."""

    def __init__(self, text: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text = text

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold #38bdf8]❯ {_('you')}[/bold #38bdf8]",
            classes="msg-role user-role",
        )
        yield Static(self.text, markup=False, classes="msg-content user-content")


class AgentResponse(Container):
    """Container representing the agent's turn.

    It dynamically hosts thinking, text responses, and tool calls in order.
    """

    def __init__(self, initial_text: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initial_text = initial_text
        self._header = Static(
            f"[bold #34d399]◆ {_('agent')}[/bold #34d399]",
            classes="msg-role agent-role",
        )
        self.current_thinking: Collapsible | None = None
        self.current_thinking_text: Static | None = None
        self._thinking_chunks: list[str] = []
        self.current_text_widget: Markdown | None = None
        self._text_chunks: list[str] = [initial_text] if initial_text else []
        self.thinking_timer: Timer | None = None
        self._thinking_dots_count = 3
        self._text_update_timer: Timer | None = None
        self._last_text_update = 0.0

    def compose(self) -> ComposeResult:
        yield self._header
        if self.initial_text:
            self.current_text_widget = Markdown(self.initial_text, classes="msg-content agent-content")
            yield self.current_text_widget

    def _animate_thinking(self) -> None:
        if self.current_thinking is not None:
            self._thinking_dots_count = (self._thinking_dots_count % 3) + 1
            dots = " ·" * self._thinking_dots_count
            self.current_thinking.title = f"⟡ {_('Thinking')}{dots}"

    def _stop_thinking_animation(self) -> None:
        if self.thinking_timer is not None:
            self.thinking_timer.stop()
            self.thinking_timer = None
        if self.current_thinking is not None:
            self.current_thinking.title = f"⟡ {_('Thought process')}"

    def append_thinking(self, delta: str) -> None:
        self.current_text_widget = None
        if self.current_thinking is None:
            collapse_default = self.app.repl.runtime.settings.runtime.collapse_thinking
            self.current_thinking_text = Static("", classes="msg-content thinking-body")
            self._thinking_chunks = []
            self.current_thinking = Collapsible(
                self.current_thinking_text,
                title=f"⟡ {_('Thinking ···')}",
                collapsed=collapse_default,
            )
            self.mount(self.current_thinking)
            self._thinking_dots_count = 3
            self.thinking_timer = self.set_interval(0.5, self._animate_thinking)

        if self.current_thinking_text is not None:
            self._thinking_chunks.append(delta)
            self.current_thinking_text.update(
                Text("".join(self._thinking_chunks), style="dim italic #8b949e")
            )

    def append_text(self, delta: str) -> None:
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        if self.current_text_widget is None:
            self.current_text_widget = Markdown("", classes="msg-content agent-content")
            self.mount(self.current_text_widget)
            self._text_chunks = []
        self._text_chunks.append(delta)

        now = time.monotonic()
        if now - self._last_text_update > 0.1:
            self.flush_text()
        elif self._text_update_timer is None:
            self._text_update_timer = self.set_timer(0.1, self.flush_text)

    def flush_text(self) -> None:
        if self._text_update_timer is not None:
            self._text_update_timer.stop()
            self._text_update_timer = None
        if self.current_text_widget is not None:
            self.current_text_widget.update("".join(self._text_chunks))
        self._last_text_update = time.monotonic()

    def finish_generation(self) -> None:
        self.flush_text()
        self._stop_thinking_animation()

    def _reset_active_stream(self) -> None:
        self.flush_text()
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        self.current_text_widget = None

    def add_tool_call(self, name: str, agent: str | None = None) -> None:
        self._reset_active_stream()
        self.mount(ToolCallMessage(tool_name=name, agent_name=agent))

    def add_tool_output(self, agent: str | None = None, output_len: int | None = None) -> None:
        self._reset_active_stream()
        self.mount(ToolOutputMessage(agent_name=agent, output_len=output_len))

    def add_error(self, content: str) -> None:
        self._reset_active_stream()
        self.mount(SystemMessage(f"[bold #f87171]✕ {_('Error:')}[/bold #f87171] [red]{escape(content)}[/red]"))

    def add_warning(self, content: str) -> None:
        self._reset_active_stream()
        self.mount(SystemMessage(f"[bold #fbbf24]⚠ {_('Warning:')}[/bold #fbbf24] [yellow]{escape(content)}[/yellow]"))


class ToolCallMessage(Static):
    """One-line tool invocation event."""

    def __init__(self, tool_name: str, agent_name: str | None = None, **kwargs: Any) -> None:
        prefix = f"[dim]{escape(f'[{agent_name}]')}[/dim] " if agent_name else ""
        super().__init__(
            f"  [#fbbf24]⚙[/#fbbf24] {prefix}[bold #e6edf3]{escape(tool_name)}[/bold #e6edf3]",
            **kwargs,
        )


class ToolOutputMessage(Static):
    """One-line tool output acknowledgement."""

    def __init__(self, agent_name: str | None = None, output_len: int | None = None, **kwargs: Any) -> None:
        prefix = f"[dim]{escape(f'[{agent_name}]')}[/dim] " if agent_name else ""
        suffix = f" [dim]({_('{output_len} chars', output_len=output_len)})[/dim]" if output_len is not None else ""
        super().__init__(
            f"  [#34d399]✓[/#34d399] {prefix}[dim]{_('output received')}[/dim]{suffix}",
            **kwargs,
        )


class SystemMessage(Static):
    """Command output or system-level notice."""
    pass


class ToolApprovalWidget(Container):
    """Inline widget prompting the user for approval of sensitive tool calls."""

    BUTTON_IDS = ["approve-btn", "reject-btn", "allow-btn", "cancel-btn"]

    def __init__(self, action_requests: list[dict[str, Any]], app_ref: OllamaAgentApp, scroll: Any, agent_msg: AgentResponse, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_requests = action_requests
        self.app_ref = app_ref
        self.scroll = scroll
        self.agent_msg = agent_msg
        self.buttons_container: Horizontal | None = None

    def compose(self) -> ComposeResult:
        yield Static(f"[bold #fbbf24]⚠ {_('Action Approval Required')}[/bold #fbbf24]", classes="approval-title")
        for req in self.action_requests:
            name = escape(str(req["name"]))
            args = escape(str(req["args"]))
            yield Static(f"{_('Tool:')} [bold #38bdf8]{name}[/bold #38bdf8]\n{_('Arguments:')} [dim]{args}[/dim]", classes="approval-details")

        with Horizontal(classes="approval-buttons") as buttons:
            self.buttons_container = buttons
            yield Button(_("Approve (y)"), id="approve-btn", variant="success", classes="approval-btn")
            yield Button(_("Reject (n)"), id="reject-btn", variant="error", classes="approval-btn")
            yield Button(_("Allow Session (a)"), id="allow-btn", variant="primary", classes="approval-btn")
            yield Button(_("Cancel (c)"), id="cancel-btn", classes="approval-btn")

    def on_mount(self) -> None:
        buttons = self.query("#approve-btn")
        if buttons:
            buttons.first().focus()
        else:
            self.call_after_refresh(lambda: self.query_one("#approve-btn", Button).focus())


    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._handle_decision(event.button.id)

    def on_key(self, event: events.Key) -> None:
        key = event.key.lower()
        if key == "y":
            event.stop()
            self._handle_decision("approve-btn")
        elif key == "n":
            event.stop()
            self._handle_decision("reject-btn")
        elif key == "a":
            event.stop()
            self._handle_decision("allow-btn")
        elif key in ("c", "escape"):
            event.stop()
            self._handle_decision("cancel-btn")
        elif key in ("left", "up", "shift+tab"):
            event.stop()
            event.prevent_default()
            focused = self.app.focused
            current_id = focused.id if focused and focused.id in self.BUTTON_IDS else None
            idx = self.BUTTON_IDS.index(current_id) if current_id else 0
            prev_id = self.BUTTON_IDS[(idx - 1) % len(self.BUTTON_IDS)]
            self.query_one(f"#{prev_id}", Button).focus()
        elif key in ("right", "down", "tab"):
            event.stop()
            event.prevent_default()
            focused = self.app.focused
            current_id = focused.id if focused and focused.id in self.BUTTON_IDS else None
            idx = self.BUTTON_IDS.index(current_id) if current_id else -1
            next_id = self.BUTTON_IDS[(idx + 1) % len(self.BUTTON_IDS)]
            self.query_one(f"#{next_id}", Button).focus()
        elif key in ("enter", "space"):
            focused = self.app.focused
            decision = focused.id if focused and focused.id in self.BUTTON_IDS else "approve-btn"
            event.stop()
            event.prevent_default()
            self._handle_decision(decision)

    def _handle_decision(self, decision_type: str | None) -> None:
        if not self.buttons_container:
            return

        for child in self.buttons_container.query(Button):
            child.disabled = True

        decisions = []
        for req in self.action_requests:
            name = req["name"]
            if decision_type == "approve-btn":
                decisions.append({"type": "approve"})
            elif decision_type == "reject-btn":
                decisions.append({
                    "type": "reject",
                    "message": _("User rejected executing tool '{name}'.", name=name)
                })
            elif decision_type == "allow-btn":
                self.app_ref.repl.runtime.auto_approved_tools.add(name)
                decisions.append({"type": "approve"})
            elif decision_type == "cancel-btn":
                decisions.append({
                    "type": "reject",
                    "message": _("User cancelled the execution.")
                })

        self.buttons_container.remove()
        self.buttons_container = None

        status_text = ""
        if decision_type == "approve-btn":
            status_text = f"[bold #34d399]✓ {_('Approved')}[/bold #34d399]"
        elif decision_type == "reject-btn":
            status_text = f"[bold #f87171]✗ {_('Rejected')}[/bold #f87171]"
        elif decision_type == "allow-btn":
            status_text = f"[bold #38bdf8]✓ {_('Allowed for session & approved')}[/bold #38bdf8]"
        elif decision_type == "cancel-btn":
            status_text = f"[bold #f87171]✗ {_('Cancelled')}[/bold #f87171]"

        self.mount(Static(f"  {status_text}", classes="approval-status"))

        inp = self.app_ref.query_one(ReplInput)
        inp.disabled = False

        footer = self.app_ref.query_one(AgentFooter)
        footer.set_approval(False)
        self.app_ref._is_approval_pending = False

        self.app_ref._current_worker = self.app_ref.run_worker(
            self.app_ref._handle_approval_decision(decisions, self.scroll, self.agent_msg)
        )
