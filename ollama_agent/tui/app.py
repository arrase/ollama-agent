"""Terminal user interface (TUI) using Textual."""

from typing import Optional

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog

from ..agent import OllamaAgent, set_tool_timeout
from ..core import extract_text
from ..streaming import TUIStreamingRenderer, stream_agent_events_with_renderer
from ..tasks import Task, TaskManager
from .screens import CreateTaskScreen, SessionListScreen, TaskListScreen


class ChatInterface(App):
    """Chat interface to interact with the AI agent."""

    CSS = """
    #chat-container {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    #chat-log {
        height: 1fr;
        scrollbar-gutter: stable;
        overflow-x: hidden;
    }

    #input-container {
        height: 3;
        border: solid $accent;
        padding: 0 1;
    }

    #user-input {
        width: 100%;
        height: 100%;
        border: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+r", "reset_session", "New Session"),
        Binding("ctrl+s", "load_session", "Load Session"),
        Binding("ctrl+l", "list_tasks", "Tasks"),
        Binding("ctrl+t", "create_task", "Create Task"),
    ]

    def __init__(self, agent: OllamaAgent, tool_timeout: int = 30):
        super().__init__()
        self.agent = agent
        self.task_manager = TaskManager()
        set_tool_timeout(tool_timeout)

    @property
    def chat_log(self) -> RichLog:
        return self.query_one("#chat-log", RichLog)

    @property
    def input_field(self) -> Input:
        return self.query_one("#user-input", Input)

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        with Container(id="input-container"):
            yield Input(placeholder="Type your message here...", id="user-input")
        yield Footer()

    async def on_mount(self) -> None:
        await self.agent.initialize()
        session_id = self.agent.session_manager.get_session_id()
        self.title = "Ollama Agent - Chat"
        self._update_subtitle(session_id)

        self._log("Welcome to Ollama Agent!", "italic cyan")
        self._log(f"Session ID: {session_id}", "italic cyan")
        self._log("Shortcuts: Ctrl+R=New | Ctrl+S=Load | Ctrl+T=Task | Ctrl+L=Tasks", "italic cyan")
        self._log("")
        self.input_field.focus()

    async def on_unmount(self) -> None:
        await self.agent.cleanup()

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _log(
        self,
        message: str,
        style: str = "",
        *,
        prefix: Optional[str] = None,
        markdown: bool = False,
    ) -> None:
        """Write a message to the chat log."""
        if not message:
            self.chat_log.write("")
            return
        if markdown:
            if prefix:
                self.chat_log.write(Text(f"{prefix}:", style=style))
            self.chat_log.write(RichMarkdown(message))
        else:
            text = f"{prefix}: {message}" if prefix else message
            self.chat_log.write(Text(text, style=style) if style else text)

    def _update_subtitle(self, session_id: Optional[str]) -> None:
        session_str = f" | Session: {session_id[:8]}..." if session_id else ""
        self.sub_title = f"Model: {self.agent.model}{session_str}"

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return
        self.input_field.value = ""
        self._log(message, "bold blue", prefix="User")
        await self._stream_response(message)

    async def _stream_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        await stream_agent_events_with_renderer(
            self.agent,
            prompt,
            TUIStreamingRenderer(self.chat_log),
            model=model,
            reasoning_effort=reasoning_effort,
            ignore={"agent_update"},
        )

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_reset_session(self) -> None:
        session_id = self.agent.session_manager.reset_session()
        self.chat_log.clear()
        self._log("New session started!", "italic cyan")
        self._log(f"Session ID: {session_id}", "italic cyan")
        self._log("")
        self._update_subtitle(session_id)

    def action_load_session(self) -> None:
        async def on_select(action: Optional[str]) -> None:
            if action and action.startswith("load:"):
                await self._load_session(action.removeprefix("load:"))

        self.push_screen(SessionListScreen(self.agent), on_select)

    def action_create_task(self) -> None:
        def on_save(task: Optional[Task]) -> None:
            if task:
                task_id = self.task_manager.save(task)
                self._log(f"Task saved: {task.title} ({task_id})", "italic cyan")
                self._log("")

        self.push_screen(CreateTaskScreen(self.agent), on_save)

    def action_list_tasks(self) -> None:
        def on_select(action: Optional[str]) -> None:
            if action and action.startswith("run:"):
                self.run_worker(self._run_task(action.removeprefix("run:")))

        self.push_screen(TaskListScreen(self.task_manager), on_select)

    # -------------------------------------------------------------------------
    # Session / Task helpers
    # -------------------------------------------------------------------------

    async def _load_session(self, session_id: str) -> None:
        sm = self.agent.session_manager
        sm.load_session(session_id)
        self.chat_log.clear()

        self._log(f"Loaded session: {session_id}", "italic cyan")
        self._log(f"Session ID: {session_id}", "italic cyan")
        self._log("")

        for item in await sm.get_session_history(session_id):
            if not isinstance(item, dict):
                continue
            role, text = item.get("role", ""), extract_text(item.get("content", ""))
            if role == "user" and text:
                self._log(text, "bold blue", prefix="User")
            elif role == "assistant" and text:
                self._log(text, "bold green", prefix="Agent", markdown=True)

        self._log("")
        self.chat_log.scroll_end(animate=False)
        self._update_subtitle(session_id)

    async def _run_task(self, task_id: str) -> None:
        task = self.task_manager.load(task_id)
        if not task:
            self._log(f"Task not found: {task_id}", "bold red", prefix="Error")
            return

        self._log(f"Executing task: {task.title} ({task_id})", "italic cyan")
        self._log(task.prompt, "bold blue", prefix="User")
        await self._stream_response(
            task.prompt, model=task.model, reasoning_effort=task.reasoning_effort
        )
