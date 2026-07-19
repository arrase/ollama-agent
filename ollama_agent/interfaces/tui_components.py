from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea, Collapsible, Markdown
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..settings.paths import APP_DIR, HISTORY_DB_PATH

if TYPE_CHECKING:
    from textual.worker import Worker
    from ..agent import AgentRuntime
    from .repl import OllamaREPL, OllamaAgentApp


# ─── Header ──────────────────────────────────────────────────────────────────

class AgentHeader(Static):
    """Dynamic TUI Header displaying agent status information."""

    def __init__(self, repl: OllamaREPL, **kwargs):
        super().__init__(**kwargs)
        self.repl = repl

    def on_mount(self) -> None:
        self.update_header()
        self.set_interval(2.0, self.update_header)

    def update_header(self) -> None:
        ms = self.repl.runtime.settings.model
        rag_ctx = self.repl._rag_ctx
        rag_db = rag_ctx.rag_manager.current_database if rag_ctx else None
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
        self._history: list[str] = []
        self._history_index: int = 0
        self._temp_input: str = ""

    async def on_mount(self) -> None:
        super().on_mount()
        await self._load_history()

    async def _load_history(self) -> None:
        history_path = APP_DIR / "tui_history.txt"
        loaded_set = set()
        self._history = []

        def _read_history() -> list[str]:
            res = []
            if history_path.exists():
                try:
                    with open(history_path, "r", encoding="utf-8") as f:
                        for line in f:
                            entry = line.strip("\n")
                            if entry and not entry.startswith("/") and entry not in loaded_set:
                                res.append(entry)
                                loaded_set.add(entry)
                except Exception:
                    pass
            return res

        import asyncio
        file_entries = await asyncio.to_thread(_read_history)
        self._history.extend(file_entries)

        # If the history is short or empty, try to populate it with previous user prompts from the DB checkpointer
        if len(self._history) < 50:
            if HISTORY_DB_PATH.exists():
                try:
                    async with AsyncSqliteSaver.from_conn_string(str(HISTORY_DB_PATH)) as saver:
                        async for checkpoint_tuple in saver.alist(None):
                            checkpoint = checkpoint_tuple.checkpoint
                            values = checkpoint.get("channel_values", {})
                            for key, val in values.items():
                                if isinstance(val, list):
                                    for msg in val:
                                        msg_type = getattr(msg, "type", None) or type(msg).__name__.lower()
                                        if "human" in msg_type or "user" in msg_type:
                                            content = getattr(msg, "content", None)
                                            if content and isinstance(content, str):
                                                entry = content.strip()
                                                if entry and not entry.startswith("/") and entry not in loaded_set:
                                                    self._history.insert(0, entry)
                                                    loaded_set.add(entry)
                except Exception:
                    pass

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
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception:
            pass

    class Submitted(Message):
        """Emitted when the user submits the input."""
        def __init__(self, input_widget: ReplInput, value: str) -> None:
            super().__init__()
            self.input = input_widget
            self.value = value

    def _handle_autocomplete_key(self, event: events.Key, app: Any, autolist: Any) -> bool:
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
            event.stop()
            event.prevent_default()
            if autolist.highlighted is not None:
                app.accept_completion(autolist.highlighted)
            return True
        return False

    def _handle_history_key(self, event: events.Key) -> bool:
        if event.key == "up":
            event.stop()
            event.prevent_default()
            if self._history:
                if self._history_index == len(self._history):
                    self._temp_input = self.text
                if self._history_index > 0:
                    self._history_index -= 1
                    self.text = self._history[self._history_index]
                    self.action_cursor_line_end()
            return True
        elif event.key == "down":
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
            return True
        return False

    async def on_key(self, event: events.Key) -> None:
        app: Any = self.app
        autolist = app.query_one("#autocomplete-list")
        if autolist.display:
            if self._handle_autocomplete_key(event, app, autolist):
                return
        else:
            if self._handle_history_key(event):
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
        self.current_thinking: Collapsible | None = None
        self.current_thinking_text: Static | None = None
        self.current_text_widget: Markdown | None = None
        self._text_chunks: list[str] = []
        self.thinking_timer: Any = None
        self._thinking_dots_count = 3
        self._text_update_timer: Any = None
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
                app_ref: Any = self.app
                collapse_default = getattr(
                    app_ref.repl.runtime.settings.runtime,
                    "collapse_thinking",
                    True,
                )
            self.current_thinking_text = Static("", classes="msg-content thinking-body")
            self.current_thinking_text._chunks = []  # type: ignore[attr-defined]
            self.current_thinking = Collapsible(
                self.current_thinking_text,
                title="🧠 Thinking...",
                collapsed=collapse_default,
            )
            self.mount(self.current_thinking)
            self._thinking_dots_count = 3
            self.thinking_timer = self.set_interval(0.5, self._animate_thinking)

        if self.current_thinking_text is not None:
            self.current_thinking_text._chunks.append(delta)  # type: ignore[attr-defined]
            self.current_thinking_text.update(
                f"[dim italic #cba6f7]{''.join(self.current_thinking_text._chunks)}[/dim italic #cba6f7]"  # type: ignore[attr-defined]
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
        if getattr(self, "current_text_widget", None) is not None and self.current_text_widget is not None:
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

    def __init__(self, app_ref: OllamaAgentApp, task_id: str, force: bool):
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

    def __init__(self, app_ref: OllamaAgentApp, skill_id: str, force: bool):
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

    def __init__(self, action_requests: list[dict], app_ref: OllamaAgentApp, scroll: Any, agent_msg: AgentResponse, **kwargs):
        super().__init__(**kwargs)
        self.action_requests = action_requests
        self.app_ref = app_ref
        self.scroll = scroll
        self.agent_msg = agent_msg
        self.buttons_container: Any = None

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
