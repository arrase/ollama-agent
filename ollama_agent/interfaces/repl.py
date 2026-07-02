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
from textual.widgets import Button, Input, TextArea, Static, OptionList, Markdown, Collapsible
from textual.widgets.option_list import Option
from textual.screen import ModalScreen
from langgraph.types import Command

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager, set_tool_timeout
from ..rag import RAGContext, RAGManager, load_rag_database
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


class AgentHeader(Static):
    """Dynamic TUI Header displaying agent status information."""

    def __init__(self, repl: "OllamaREPL", **kwargs):
        super().__init__(**kwargs)
        self.repl = repl

    def on_mount(self) -> None:
        self.update_header()
        self.set_interval(2.0, self.update_header)

    def update_header(self) -> None:
        ms = self.repl.runtime.settings.model
        rag_db = (
            self.repl._rag_ctx.rag_manager.current_database
            if (self.repl._rag_ctx and self.repl._rag_ctx.rag_manager)
            else None
        )
        rag_info = f"  │  [bold #f5c2e7]RAG[/bold #f5c2e7] {rag_db}" if rag_db else ""
        yolo_status = "[bold #f38ba8]YOLO: On[/bold #f38ba8]" if self.repl.runtime.yolo_mode else "[bold #a6e3a1]YOLO: Off[/bold #a6e3a1]"
        self.update(
            f"[bold #a6e3a1]🤖 Ollama Agent[/bold #a6e3a1]  │  "
            f"[bold #89b4fa]Model[/bold #89b4fa] {ms.name}  │  "
            f"[bold #fab387]Effort[/bold #fab387] {ms.reasoning_effort}{rag_info}  │  {yolo_status}"
        )


# ─── Custom Input ─────────────────────────────────────────────────────────────


class ReplInput(TextArea):
    """Interactive input field that captures Tab/arrow keys for autocomplete."""

    def __init__(self, **kwargs):
        kwargs.setdefault("highlight_cursor_line", False)
        super().__init__(**kwargs)

    class Submitted(Message):
        """Emitted when the user submits the input."""
        def __init__(self, input_widget: "ReplInput", value: str) -> None:
            super().__init__()
            self.input = input_widget
            self.value = value

    async def on_key(self, event: events.Key) -> None:
        app = self.app
        autolist = app.query_one("#autocomplete-list")
        if autolist.display:
            if event.key == "down":
                event.stop()
                event.prevent_default()
                if autolist.highlighted is None:
                    autolist.highlighted = 0
                elif autolist.highlighted < autolist.option_count - 1:
                    autolist.highlighted += 1
                return
            elif event.key == "up":
                event.stop()
                event.prevent_default()
                if autolist.highlighted is not None and autolist.highlighted > 0:
                    autolist.highlighted -= 1
                return
            elif event.key == "tab":
                event.stop()
                event.prevent_default()
                if autolist.highlighted is not None:
                    app.accept_completion(autolist.highlighted)
                return
            elif event.key == "escape":
                event.stop()
                event.prevent_default()
                app.hide_autocomplete()
                return
            elif event.key == "enter":
                event.stop()
                event.prevent_default()
                if autolist.highlighted is not None:
                    app.accept_completion(autolist.highlighted)
                return

        if event.key == "enter":
            event.stop()
            event.prevent_default()
            val = self.text.strip()
            if val:
                self.post_message(self.Submitted(self, val))


# ─── Chat Message Widgets ─────────────────────────────────────────────────────


class UserMessage(Container):
    """Rendered user prompt bubble."""

    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self.text = text

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold #89b4fa]👤 You[/bold #89b4fa]",
            classes="msg-role",
        )
        yield Static(self.text, classes="msg-content")


class AgentResponse(Container):
    """Container representing the agent's turn.

    It dynamically hosts thinking, text responses, and tool calls in order.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._header = Static(
            "[bold #a6e3a1]🤖 Agent[/bold #a6e3a1]",
            classes="msg-role",
        )
        self.current_thinking = None
        self.current_thinking_text = None
        self.current_text_widget = None
        self._text_chunks = []
        self.thinking_timer = None
        self._thinking_dots_count = 3
        self._text_update_timer = None
        self._last_text_update = 0.0

    def compose(self) -> ComposeResult:
        yield self._header

    def _animate_thinking(self) -> None:
        if self.current_thinking is not None:
            num_dots = (getattr(self, "_thinking_dots_count", 2) % 6) + 1
            self._thinking_dots_count = num_dots
            dots = "." * num_dots
            self.current_thinking.title = f"🧠 Thinking{dots}"

    def _stop_thinking_animation(self) -> None:
        if getattr(self, "thinking_timer", None) is not None:
            self.thinking_timer.stop()
            self.thinking_timer = None
        if self.current_thinking is not None:
            self.current_thinking.title = "🧠 Thinking"

    def append_thinking(self, delta: str) -> None:
        self.current_text_widget = None
        if self.current_thinking is None:
            collapse_default = True
            if self.app and hasattr(self.app, "repl"):
                collapse_default = getattr(
                    self.app.repl.runtime.settings.runtime,
                    "collapse_thinking",
                    True,
                )
            self.current_thinking_text = Static("", classes="msg-content thinking-body")
            self.current_thinking_text._chunks = []
            self.current_thinking = Collapsible(
                self.current_thinking_text,
                title="🧠 Thinking...",
                collapsed=collapse_default,
            )
            self.mount(self.current_thinking)
            self._thinking_dots_count = 3
            self.thinking_timer = self.set_interval(0.5, self._animate_thinking)

        self.current_thinking_text._chunks.append(delta)
        self.current_thinking_text.update(
            f"[dim italic #cba6f7]{''.join(self.current_thinking_text._chunks)}[/dim italic #cba6f7]"
        )

    def append_text(self, delta: str) -> None:
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        if self.current_text_widget is None:
            self.current_text_widget = Markdown("", classes="msg-content")
            self.mount(self.current_text_widget)
            self._text_chunks = []
        self._text_chunks.append(delta)
        
        now = time.monotonic()
        if now - self._last_text_update > 0.1:
            self.flush_text()
        elif self._text_update_timer is None:
            self._text_update_timer = self.set_timer(0.1, self.flush_text)

    def flush_text(self) -> None:
        if getattr(self, "_text_update_timer", None) is not None:
            self._text_update_timer.stop()
            self._text_update_timer = None
        if getattr(self, "current_text_widget", None) is not None:
            self.current_text_widget.update("".join(self._text_chunks))
        self._last_text_update = time.monotonic()

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
        self.mount(SystemMessage(f"[red]❌ {content}[/red]"))

    def add_warning(self, content: str) -> None:
        self.flush_text()
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        self.current_text_widget = None
        self.mount(SystemMessage(f"[yellow]⚠ {content}[/yellow]"))


class ToolCallMessage(Static):
    """One-line tool invocation event."""

    def __init__(self, tool_name: str, agent_name: str | None = None, **kwargs):
        prefix = f"[{agent_name}] " if agent_name else ""
        super().__init__(
            f"  [#f9e2af]✦ {prefix}[/#f9e2af][bold #f9e2af]{tool_name}[/bold #f9e2af]",
            **kwargs,
        )


class ToolOutputMessage(Static):
    """One-line tool output acknowledgement."""

    def __init__(self, agent_name: str | None = None, output_len: int | None = None, **kwargs):
        prefix = f"[{agent_name}] " if agent_name else ""
        suffix = f" ({output_len} chars)" if output_len is not None else ""
        super().__init__(
            f"  [#94e2d5]✓ {prefix}output received{suffix}[/#94e2d5]",
            **kwargs,
        )


class SystemMessage(Static):
    """Command output or system-level notice."""
    pass


class Label(Static):
    """Simple wrapper used inside modals."""
    pass


# ─── Modals ───────────────────────────────────────────────────────────────────


class TaskCreateModal(ModalScreen):
    """Modal dialog form for creating a new Task."""

    CSS = """
    TaskCreateModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    #modal-card {
        width: 65;
        height: auto;
        background: #1e1e2e;
        border: solid #89b4fa;
        padding: 1 2;
    }
    #modal-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: #89b4fa;
        margin-bottom: 1;
    }
    .form-container {
        width: 100%;
        height: auto;
        layout: vertical;
    }
    .form-row {
        height: 1;
        margin-bottom: 1;
        align: left middle;
    }
    .field-label {
        width: 12;
        content-align: right middle;
        color: #bac2de;
        margin-right: 2;
    }
    .modal-input {
        width: 1fr;
        background: #11111b !important;
        border: none !important;
        color: #cdd6f4 !important;
        height: 1;
        padding: 0 1;
    }
    .modal-input:focus {
        background: #313244 !important;
        color: #cdd6f4 !important;
        border: none !important;
    }
    .modal-input:disabled {
        color: #6c7086 !important;
        background: #11111b !important;
        border: none !important;
    }
    .modal-input > .input--cursor {
        background: #cdd6f4 !important;
        color: #11111b !important;
    }
    .modal-input > .input--selection {
        background: #585b70 !important;
        color: #cdd6f4 !important;
    }
    .prompt-label {
        color: #bac2de;
        margin-bottom: 0;
        margin-top: 1;
    }
    #prompt-area {
        height: 5;
        background: #11111b !important;
        border: solid #45475a !important;
        color: #cdd6f4 !important;
        margin-top: 1;
        margin-bottom: 1;
    }
    #prompt-area:focus {
        background: #11111b !important;
        color: #cdd6f4 !important;
        border: solid #89b4fa !important;
    }
    #button-row {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    .modal-button {
        height: 1;
        border: none;
        margin: 0 2;
        min-width: 12;
    }
    """

    def __init__(self, app_ref: "OllamaAgentApp", task_id: str, force: bool):
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
                yield TextArea(id="prompt-area")
            with Horizontal(id="button-row"):
                yield Button("Cancel", id="cancel-btn", variant="error", classes="modal-button")
                yield Button("Create", id="create-btn", variant="success", classes="modal-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss()
        elif event.button.id == "create-btn":
            self.dismiss((
                self.query_one("#task-id-input").value.strip(),
                self.query_one("#title-input").value.strip(),
                self.query_one("#model-input").value.strip(),
                self.query_one("#effort-input").value.strip(),
                self.query_one("#prompt-area").text,
            ))


class SkillCreateModal(ModalScreen):
    """Modal dialog form for creating a new Skill."""

    CSS = """
    SkillCreateModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    #skill-modal-card {
        width: 65;
        height: auto;
        background: #1e1e2e;
        border: solid #a6e3a1;
        padding: 1 2;
    }
    #skill-modal-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: #a6e3a1;
        margin-bottom: 1;
    }
    .form-container {
        width: 100%;
        height: auto;
        layout: vertical;
    }
    .form-row {
        height: 1;
        margin-bottom: 1;
        align: left middle;
    }
    .field-label {
        width: 12;
        content-align: right middle;
        color: #bac2de;
        margin-right: 2;
    }
    .modal-input {
        width: 1fr;
        background: #11111b !important;
        border: none !important;
        color: #cdd6f4 !important;
        height: 1;
        padding: 0 1;
    }
    .modal-input:focus {
        background: #313244 !important;
        color: #cdd6f4 !important;
        border: none !important;
    }
    .modal-input:disabled {
        color: #6c7086 !important;
        background: #11111b !important;
        border: none !important;
    }
    .modal-input > .input--cursor {
        background: #cdd6f4 !important;
        color: #11111b !important;
    }
    .modal-input > .input--selection {
        background: #585b70 !important;
        color: #cdd6f4 !important;
    }
    .prompt-label {
        color: #bac2de;
        margin-bottom: 0;
        margin-top: 1;
    }
    #instructions-area {
        height: 5;
        background: #11111b !important;
        border: solid #45475a !important;
        color: #cdd6f4 !important;
        margin-top: 1;
        margin-bottom: 1;
    }
    #instructions-area:focus {
        background: #11111b !important;
        color: #cdd6f4 !important;
        border: solid #a6e3a1 !important;
    }
    #skill-button-row {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    .modal-button {
        height: 1;
        border: none;
        margin: 0 2;
        min-width: 12;
    }
    """

    def __init__(self, app_ref: "OllamaAgentApp", skill_id: str, force: bool):
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
                yield TextArea(id="instructions-area")
            with Horizontal(id="skill-button-row"):
                yield Button("Cancel", id="cancel-btn", variant="error", classes="modal-button")
                yield Button("Create", id="create-btn", variant="success", classes="modal-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss()
        elif event.button.id == "create-btn":
            self.dismiss((
                self.query_one("#skill-id-input").value.strip(),
                self.query_one("#name-input").value.strip(),
                self.query_one("#desc-input").value.strip(),
                self.query_one("#instructions-area").text,
            ))


class ToolApprovalWidget(Container):
    """Inline widget prompting the user for approval of sensitive tool calls."""

    def __init__(self, action_requests: list[dict], app_ref: "OllamaAgentApp", scroll, agent_msg: AgentResponse, **kwargs):
        super().__init__(**kwargs)
        self.action_requests = action_requests
        self.app_ref = app_ref
        self.scroll = scroll
        self.agent_msg = agent_msg
        self.buttons_container = None

    def compose(self) -> ComposeResult:
        yield Static("⚠️ Sensitive Tool Approval Required", classes="approval-title")
        for req in self.action_requests:
            name = req.get("name", "unknown")
            args = req.get("args", {})
            yield Static(f"Tool: [bold]{name}[/bold]\nArguments: {args}", classes="approval-details")
        
        with Horizontal(classes="approval-buttons") as buttons:
            self.buttons_container = buttons
            yield Button("Approve", id="approve-btn", variant="success", classes="approval-btn")
            yield Button("Reject", id="reject-btn", variant="error", classes="approval-btn")
            yield Button("Allow Session", id="allow-btn", variant="primary", classes="approval-btn")
            yield Button("Cancel", id="cancel-btn", classes="approval-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        
        # Disable all buttons to prevent double-click
        for child in self.buttons_container.query(Button):
            child.disabled = True
            
        decision_type = event.button.id
        decisions = []
        
        for req in self.action_requests:
            name = req.get("name", "unknown")
            if decision_type == "approve-btn":
                decisions.append({"type": "approve"})
            elif decision_type == "reject-btn":
                decisions.append({
                    "type": "reject",
                    "message": f"User rejected executing tool '{name}'."
                })
            elif decision_type == "allow-btn":
                # Add this tool type to auto-approved tools for this session
                self.app_ref.repl.runtime.auto_approved_tools.add(name)
                decisions.append({"type": "approve"})
            elif decision_type == "cancel-btn":
                decisions.append({
                    "type": "reject",
                    "message": "User cancelled the execution."
                })

        # Remove the buttons container from view or replace with status
        self.buttons_container.remove()
        
        # Display the chosen action
        status_text = ""
        if decision_type == "approve-btn":
            status_text = "[green]✓ Approved[/green]"
        elif decision_type == "reject-btn":
            status_text = "[red]✗ Rejected[/red]"
        elif decision_type == "allow-btn":
            status_text = "[blue]✓ Allowed for session & approved[/blue]"
        elif decision_type == "cancel-btn":
            status_text = "[red]✗ Cancelled[/red]"
            
        self.mount(Static(f"  {status_text}"))
        
        if decision_type != "cancel-btn":
            # Resume agent stream
            self.app_ref._current_worker = self.app_ref.run_worker(
                self.app_ref._handle_approval_decision(decisions, self.scroll, self.agent_msg)
            )


# ─── Main TUI App ────────────────────────────────────────────────────────────
class OllamaAgentApp(App):
    """Main Textual Application representing the Agent's interactive TUI."""

    BINDINGS = [
        ("escape", "cancel_generation", "Interrumpir"),
        ("ctrl+c", "cancel_or_quit", "Interrumpir/Salir"),
    ]

    def action_cancel_generation(self) -> None:
        if self._is_generating and hasattr(self, "_current_worker") and self._current_worker is not None:
            self._current_worker.cancel()

    def action_cancel_or_quit(self) -> None:
        if self._is_generating:
            if hasattr(self, "_current_worker") and self._current_worker is not None:
                self._current_worker.cancel()
        else:
            self.exit()

    CSS = """
    /* ── Base ─────────────────────────────────────── */
    Screen {
        background: #1e1e2e;
        layout: vertical;
    }

    /* ── Header ───────────────────────────────────── */
    AgentHeader {
        background: #1e1e2e;
        color: #cdd6f4;
        height: 2;
        content-align: center middle;
        padding: 0 1;
        border-bottom: solid #313244;
    }

    /* ── Chat area ────────────────────────────────── */
    #chat-scroll {
        background: #1e1e2e;
        height: 1fr;
        padding: 0 1;
        scrollbar-color: #585b70;
        scrollbar-color-active: #89b4fa;
        scrollbar-color-hover: #a6adc8;
    }

    /* ── Autocomplete popup ───────────────────────── */
    #autocomplete-list {
        background: #181825;
        border: round #45475a;
        max-height: 10;
        display: none;
        color: #cdd6f4;
        scrollbar-size: 1 1;
        dock: bottom;
        width: 60%;
        margin: 0 0 0 2;
    }

    /* ── Input bar ────────────────────────────────── */
    #input-bar {
        dock: bottom;
        height: auto;
        background: #1e1e2e;
        border-top: solid #313244;
        padding: 0 1;
    }
    ReplInput {
        background: #1e1e2e !important;
        border: none !important;
        color: #cdd6f4 !important;
        width: 1fr;
        height: 3;
        max-height: 10;
        padding: 0 !important;
    }
    ReplInput:focus {
        background: #1e1e2e !important;
        color: #cdd6f4 !important;
        border: none !important;
    }
    ReplInput .text-area--cursor {
        background: #cdd6f4 !important;
        color: #1e1e2e !important;
    }
    ReplInput .text-area--cursor-line {
        background: transparent !important;
    }
    ReplInput .text-area--selection {
        background: #585b70 !important;
        color: #cdd6f4 !important;
    }
    #prompt-char {
        width: 3;
        color: #89b4fa;
        text-style: bold;
        content-align: left top;
        padding: 0;
    }

    /* ── Tool Approval ────────────────────────────── */
    ToolApprovalWidget {
        background: #1e1e2e;
        border-left: solid #eed49f;
        margin: 1 0;
        padding: 0 0 0 1;
        height: auto;
        layout: vertical;
    }
    .approval-title {
        color: #eed49f;
        text-style: bold;
        margin-bottom: 0;
        padding-left: 1;
    }
    .approval-details {
        color: #cdd6f4;
        margin-bottom: 1;
        padding-left: 2;
    }
    .approval-buttons {
        height: 3;
        align: left middle;
        padding-left: 2;
    }
    .approval-btn {
        margin-right: 1;
        min-width: 10;
        height: 1;
        border: none;
    }

    /* ── Message roles ────────────────────────────── */
    .msg-role {
        text-style: bold;
        margin-bottom: 0;
        padding: 0 0 0 1;
        height: auto;
    }
    .msg-content {
        padding: 0 0 0 1;
        height: auto;
        color: #cdd6f4;
    }

    /* ── User bubble ──────────────────────────────── */
    UserMessage {
        margin: 1 0;
        padding: 0 0 0 1;
        border-left: solid #89b4fa;
        height: auto;
        background: transparent;
    }

    /* ── Agent thinking ───────────────────────────── */
    .thinking-body {
        color: #cba6f7;
        text-style: italic;
        margin: 0 0 1 1;
        padding: 0;
    }
    Collapsible {
        background: transparent !important;
        border: none !important;
        margin: 0;
        padding: 0 0 1 0;
    }
    Collapsible > Contents {
        background: transparent !important;
        border: none !important;
        margin: 0;
        padding: 0 0 0 1;
    }
    CollapsibleTitle {
        background: transparent !important;
        border: none !important;
        color: #cba6f7 !important;
        text-style: italic !important;
        padding: 0;
    }
    CollapsibleTitle:hover {
        background: transparent !important;
        color: #cba6f7 !important;
    }
    CollapsibleTitle:focus {
        background: transparent !important;
        color: #cba6f7 !important;
    }

    /* ── Agent response ───────────────────────────── */
    AgentResponse {
        margin: 1 0;
        padding: 0 0 0 1;
        border-left: solid #a6e3a1;
        height: auto;
        background: transparent;
    }

    /* ── Tool events ──────────────────────────────── */
    ToolCallMessage {
        margin: 0 0 0 2;
        padding: 0;
        height: auto;
        color: #eed49f;
        background: transparent;
    }
    ToolOutputMessage {
        margin: 0 0 0 2;
        padding: 0;
        height: auto;
        color: #8bd5ca;
        background: transparent;
    }

    /* ── System messages ──────────────────────────── */
    SystemMessage {
        margin: 1 0;
        padding: 0 0 0 1;
        border-left: solid #cba6f7;
        color: #cdd6f4;
        height: auto;
        background: transparent;
    }
    """

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
                if hasattr(self, "_timer"):
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

        app = OllamaAgentApp(self)
        try:
            await app.run_async()
        except KeyboardInterrupt:
            pass

    async def _switch_model(self, model_name: str) -> None:
        await set_model(self.console, model_name, runtime=self.runtime)

    async def _handle_new_session(self, args: list[str]) -> None:
        self.runtime.thread_id = new_session(self.console)
