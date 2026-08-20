from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Collapsible, Input, Label, Markdown, OptionList, Static, TextArea

from ..settings.paths import APP_DIR

if TYPE_CHECKING:
    from .repl import OllamaREPL, OllamaAgentApp


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
        tokens = getattr(self.repl.runtime, "last_context_tokens", 0)
        tokens = tokens if isinstance(tokens, (int, float)) else 0
        num_ctx = getattr(ms, "context_window", 0)
        num_ctx = num_ctx if isinstance(num_ctx, (int, float)) else 0

        if num_ctx > 0:
            pct = int((tokens / num_ctx) * 100)
            if pct > 90:
                color = "#f87171"
            elif pct > 75:
                color = "#fbbf24"
            else:
                color = "#38bdf8"
            tok_str = f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(int(tokens))
            ctx_str = f"{num_ctx / 1000:.1f}k" if num_ctx >= 1000 else str(int(num_ctx))
            ctx_info = f"  [dim]│[/dim]  [bold #8b949e]Context:[/bold #8b949e] [bold {color}]{tok_str}/{ctx_str} ({pct}%)[/bold {color}]"
        else:
            ctx_info = ""

        rag_ctx = self.repl._rag_ctx
        rag_db = rag_ctx.rag_manager.current_database if rag_ctx else None
        rag_info = f"  [dim]│[/dim]  [bold #8b949e]RAG:[/bold #8b949e] [bold #a78bfa]{rag_db}[/bold #a78bfa]" if rag_db else ""
        yolo_status = (
            "[bold #f87171 on #3b181e] YOLO: ON [/bold #f87171 on #3b181e]"
            if self.repl.runtime.yolo_mode
            else "[dim #8b949e]YOLO: OFF[/dim #8b949e]"
        )
        self.update(
            f"[bold #38bdf8]● ollama-agent[/bold #38bdf8]  [dim]│[/dim]  "
            f"[bold #8b949e]Model:[/bold #8b949e] [bold #e6edf3]{ms.name}[/bold #e6edf3]{ctx_info}  [dim]│[/dim]  "
            f"[bold #8b949e]Effort:[/bold #8b949e] [#e6edf3]{ms.reasoning_effort}[/#e6edf3]{rag_info}  [dim]│[/dim]  "
            f"{yolo_status}"
        )


# ─── Footer ──────────────────────────────────────────────────────────────────

class AgentFooter(Static):
    """Dynamic TUI Footer displaying keyboard shortcuts and live status."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._is_generating = False

    def on_mount(self) -> None:
        self.update_footer()

    def set_generating(self, is_generating: bool) -> None:
        self._is_generating = is_generating
        self.update_footer()

    def update_footer(self) -> None:
        if self._is_generating:
            self.update(
                "[bold #38bdf8]⟡ Generating response...[/bold #38bdf8]   "
                "[dim]press [bold #f87171]esc[/bold #f87171] or [bold #f87171]^C[/bold #f87171] to interrupt[/dim]"
            )
        else:
            self.update(
                "[dim]enter[/dim] [bold #8b949e]send[/bold #8b949e]   "
                "[dim]\\+enter[/dim] [bold #8b949e]newline[/bold #8b949e]   "
                "[dim]tab[/dim] [bold #8b949e]complete[/bold #8b949e]   "
                "[dim]↑↓[/dim] [bold #8b949e]history[/bold #8b949e]   "
                "[dim]esc[/dim] [bold #8b949e]interrupt[/bold #8b949e]   "
                "[dim]/help[/dim] [bold #8b949e]commands[/bold #8b949e]"
            )


# ─── Custom Input ─────────────────────────────────────────────────────────────

class ReplInput(TextArea):
    """Interactive input field that captures Tab/arrow keys for autocomplete."""

    BINDINGS = [
        ("ctrl+v", "paste", "Paste"),
        ("super+v", "paste", "Paste"),
        ("shift+insert", "paste", "Paste"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("highlight_cursor_line", False)
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("placeholder", "Ask anything, / for commands, @ for files...")
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
        history_path = APP_DIR / "tui_history.txt"
        loaded_set: set[str] = set()
        self._history = []

        def _read_history() -> list[str]:
            res: list[str] = []
            if history_path.exists():
                with open(history_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = line.rstrip("\r\n")
                        if entry and not entry.startswith("/") and entry not in loaded_set:
                            res.append(entry)
                            loaded_set.add(entry)
            return res

        file_entries = await asyncio.to_thread(_read_history)
        self._history.extend(file_entries)
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

        history_path = APP_DIR / "tui_history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

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
            "[bold #38bdf8]❯ you[/bold #38bdf8]",
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
            "[bold #34d399]◆ agent[/bold #34d399]",
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
            self.current_thinking.title = f"⟡ Thinking{dots}"

    def _stop_thinking_animation(self) -> None:
        if self.thinking_timer is not None:
            self.thinking_timer.stop()
            self.thinking_timer = None
        if self.current_thinking is not None:
            self.current_thinking.title = "⟡ Thought process"

    def append_thinking(self, delta: str) -> None:
        self.current_text_widget = None
        if self.current_thinking is None:
            app_ref = getattr(self.app, "repl", None)
            collapse_default = app_ref.runtime.settings.runtime.collapse_thinking if app_ref else True
            self.current_thinking_text = Static("", classes="msg-content thinking-body")
            self._thinking_chunks = []
            self.current_thinking = Collapsible(
                self.current_thinking_text,
                title="⟡ Thinking ···",
                collapsed=collapse_default,
            )
            self.mount(self.current_thinking)
            self._thinking_dots_count = 3
            self.thinking_timer = self.set_interval(0.5, self._animate_thinking)

        if self.current_thinking_text is not None:
            self._thinking_chunks.append(delta)
            self.current_thinking_text.update(
                f"[dim italic #8b949e]{''.join(self._thinking_chunks)}[/dim italic #8b949e]"
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

    def add_tool_call(self, name: str, agent: str | None = None) -> None:
        self.flush_text()
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        self.current_text_widget = None
        self.mount(ToolCallMessage(tool_name=name, agent_name=agent))

    def add_tool_output(self, agent: str | None = None, output_len: int | None = None) -> None:
        self.flush_text()
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        self.current_text_widget = None
        self.mount(ToolOutputMessage(agent_name=agent, output_len=output_len))

    def add_error(self, content: str) -> None:
        self.flush_text()
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        self.current_text_widget = None
        self.mount(SystemMessage(f"[bold #f87171]✕ Error:[/bold #f87171] [red]{content}[/red]"))

    def add_warning(self, content: str) -> None:
        self.flush_text()
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        self.current_text_widget = None
        self.mount(SystemMessage(f"[bold #fbbf24]⚠ Warning:[/bold #fbbf24] [yellow]{content}[/yellow]"))


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
        suffix = f" [dim]({output_len} chars)[/dim]" if output_len is not None else ""
        super().__init__(
            f"  [#34d399]✓[/#34d399] {prefix}[dim]output received[/dim]{suffix}",
            **kwargs,
        )


class SystemMessage(Static):
    """Command output or system-level notice."""
    pass


# ─── Modals ───────────────────────────────────────────────────────────────────

class TaskCreateModal(ModalScreen):
    """Modal dialog form for creating a new Task."""

    def __init__(self, app_ref: OllamaAgentApp, task_id: str, force: bool) -> None:
        super().__init__()
        self.app_ref = app_ref
        self.task_id = task_id
        self.force = force

    def compose(self) -> ComposeResult:
        with Container(id="modal-card"):
            yield Label(f"Create Task: {self.task_id}" if self.task_id else "Create Task", id="modal-title")
            with Container(classes="form-container"):
                with Horizontal(classes="form-row"):
                    yield Label("Task ID:", classes="field-label")
                    yield Input(value=self.task_id, id="task-id-input", classes="modal-input", disabled=bool(self.task_id))
                with Horizontal(classes="form-row"):
                    yield Label("Title:", classes="field-label")
                    yield Input(placeholder="Enter task title", id="title-input", classes="modal-input")
                with Horizontal(classes="form-row"):
                    yield Label("Model:", classes="field-label")
                    yield Input(value=self.app_ref.repl.runtime.settings.model.name, id="model-input", classes="modal-input")
                with Horizontal(classes="form-row"):
                    yield Label("Effort:", classes="field-label")
                    yield Input(value=self.app_ref.repl.runtime.settings.model.reasoning_effort, id="effort-input", classes="modal-input")
                yield Label("Prompt:", classes="prompt-label")
                yield TextArea(id="prompt-area", classes="modal-textarea")
            with Horizontal(id="button-row"):
                yield Button("Cancel", id="cancel-btn", variant="error", classes="modal-button")
                yield Button("Create", id="create-btn", variant="success", classes="modal-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss()
        elif event.button.id == "create-btn":
            self.dismiss((
                self.query_one("#task-id-input", Input).value.strip(),
                self.query_one("#title-input", Input).value.strip(),
                self.query_one("#model-input", Input).value.strip(),
                self.query_one("#effort-input", Input).value.strip(),
                self.query_one("#prompt-area", TextArea).text,
            ))


class SkillCreateModal(ModalScreen):
    """Modal dialog form for creating a new Skill."""

    def __init__(self, app_ref: OllamaAgentApp, skill_id: str, force: bool) -> None:
        super().__init__()
        self.app_ref = app_ref
        self.skill_id = skill_id
        self.force = force

    def compose(self) -> ComposeResult:
        with Container(id="skill-modal-card"):
            yield Label(f"Create Skill: {self.skill_id}" if self.skill_id else "Create Skill", id="skill-modal-title")
            with Container(classes="form-container"):
                with Horizontal(classes="form-row"):
                    yield Label("Skill ID:", classes="field-label")
                    yield Input(value=self.skill_id, id="skill-id-input", classes="modal-input", disabled=bool(self.skill_id))
                with Horizontal(classes="form-row"):
                    yield Label("Name:", classes="field-label")
                    yield Input(placeholder="Enter skill name", id="name-input", classes="modal-input")
                with Horizontal(classes="form-row"):
                    yield Label("Description:", classes="field-label")
                    yield Input(placeholder="Enter description", id="desc-input", classes="modal-input")
                yield Label("Instructions:", classes="prompt-label")
                yield TextArea(id="instructions-area", classes="modal-textarea")
            with Horizontal(id="skill-button-row"):
                yield Button("Cancel", id="cancel-btn", variant="error", classes="modal-button")
                yield Button("Create", id="create-btn", variant="success", classes="modal-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss()
        elif event.button.id == "create-btn":
            self.dismiss((
                self.query_one("#skill-id-input", Input).value.strip(),
                self.query_one("#name-input", Input).value.strip(),
                self.query_one("#desc-input", Input).value.strip(),
                self.query_one("#instructions-area", TextArea).text,
            ))


class ToolApprovalWidget(Container):
    """Inline widget prompting the user for approval of sensitive tool calls."""

    def __init__(self, action_requests: list[dict[str, Any]], app_ref: OllamaAgentApp, scroll: Any, agent_msg: AgentResponse, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_requests = action_requests
        self.app_ref = app_ref
        self.scroll = scroll
        self.agent_msg = agent_msg
        self.buttons_container: Horizontal | None = None

    def compose(self) -> ComposeResult:
        yield Static("[bold #fbbf24]⚠ Action Approval Required[/bold #fbbf24]", classes="approval-title")
        for req in self.action_requests:
            name = req["name"]
            args = req["args"]
            yield Static(f"Tool: [bold #38bdf8]{name}[/bold #38bdf8]\nArguments: [dim]{args}[/dim]", classes="approval-details")

        with Horizontal(classes="approval-buttons") as buttons:
            self.buttons_container = buttons
            yield Button("Approve [y]", id="approve-btn", variant="success", classes="approval-btn")
            yield Button("Reject [n]", id="reject-btn", variant="error", classes="approval-btn")
            yield Button("Allow Session [a]", id="allow-btn", variant="primary", classes="approval-btn")
            yield Button("Cancel [esc]", id="cancel-btn", classes="approval-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._handle_decision(event.button.id)

    def on_key(self, event: events.Key) -> None:
        key = event.key.lower()
        if key in ("y",):
            event.stop()
            self._handle_decision("approve-btn")
        elif key in ("n",):
            event.stop()
            self._handle_decision("reject-btn")
        elif key in ("a",):
            event.stop()
            self._handle_decision("allow-btn")
        elif key in ("escape", "c"):
            event.stop()
            self._handle_decision("cancel-btn")

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
                    "message": f"User rejected executing tool '{name}'."
                })
            elif decision_type == "allow-btn":
                self.app_ref.repl.runtime.auto_approved_tools.add(name)
                decisions.append({"type": "approve"})
            elif decision_type == "cancel-btn":
                decisions.append({
                    "type": "reject",
                    "message": "User cancelled the execution."
                })

        self.buttons_container.remove()
        self.buttons_container = None

        status_text = ""
        if decision_type == "approve-btn":
            status_text = "[bold #34d399]✓ Approved[/bold #34d399]"
        elif decision_type == "reject-btn":
            status_text = "[bold #f87171]✗ Rejected[/bold #f87171]"
        elif decision_type == "allow-btn":
            status_text = "[bold #38bdf8]✓ Allowed for session & approved[/bold #38bdf8]"
        elif decision_type == "cancel-btn":
            status_text = "[bold #f87171]✗ Cancelled[/bold #f87171]"

        self.mount(Static(f"  {status_text}", classes="approval-status"))

        if decision_type != "cancel-btn":
            self.app_ref._current_worker = self.app_ref.run_worker(
                self.app_ref._handle_approval_decision(decisions, self.scroll, self.agent_msg)
            )

