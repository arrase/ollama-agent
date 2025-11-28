"""Terminal user interface (TUI) using Textual."""

from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog

from ..agent import OllamaAgent, set_tool_timeout
from ..streaming import TUIStreamingRenderer, stream_agent_events
from ..tasks import TaskManager
from .actions import UIActions
from .chat_logger import ChatLogger


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
        self._chat_log: RichLog | None = None
        self._input_field: Input | None = None
        self._chat_logger: ChatLogger | None = None
        self._actions: UIActions | None = None
        set_tool_timeout(tool_timeout)

    @property
    def chat_log(self) -> RichLog:
        if self._chat_log is None:
            raise RuntimeError("Chat log not ready")
        return self._chat_log

    @property
    def input_field(self) -> Input:
        if self._input_field is None:
            raise RuntimeError("Input widget not ready")
        return self._input_field

    @property
    def chat_logger(self) -> ChatLogger:
        if self._chat_logger is None:
            raise RuntimeError("Chat logger not ready")
        return self._chat_logger

    @property
    def actions(self) -> UIActions:
        if self._actions is None:
            raise RuntimeError("UI actions not ready")
        return self._actions

    def _set_subtitle(self, session_id: Optional[str]) -> None:
        session_piece = f" | Session: {session_id[:8]}..." if session_id else ""
        self.sub_title = f"Model: {self.agent.model}{session_piece}"

    def compose(self) -> ComposeResult:
        """Create the interface widgets."""
        yield Header()

        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)

        with Container(id="input-container"):
            yield Input(
                placeholder="Type your message here (Ctrl+V to paste)...",
                id="user-input"
            )

        yield Footer()

    async def on_mount(self) -> None:
        """Execute when the application is mounted."""
        await self.agent.initialize()
        session_id = self.agent.session_manager.get_session_id()
        self.title = "Ollama Agent - Chat"
        self._chat_log = self.query_one("#chat-log", RichLog)
        self._input_field = self.query_one("#user-input", Input)
        self._chat_logger = ChatLogger(self.chat_log)
        self._actions = UIActions(self)

        self._set_subtitle(session_id)
        self.chat_logger.write_message("Welcome to Ollama Agent!", style="italic cyan")
        self.chat_logger.write_message(
            f"Session ID: {session_id}", style="italic cyan")
        self.chat_logger.write_message(
            "Type your message and press Enter to send. Use Ctrl+V to paste text.",
            style="italic cyan",
        )
        self.chat_logger.write_message(
            "Shortcuts: Ctrl+R=New Session | Ctrl+S=Load Session | Ctrl+T=Create Task | Ctrl+L=Tasks",
            style="italic cyan",
        )
        self.chat_logger.blank_line()
        self.input_field.focus()

    async def on_unmount(self) -> None:
        """Execute when the application is unmounted."""
        # Cleanup MCP servers
        await self.agent.cleanup()

    async def _stream_agent_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Render a streamed agent response into the chat log."""
        renderer = TUIStreamingRenderer(self.chat_log)
        try:
            await stream_agent_events(
                self.agent,
                prompt,
                renderer,
                model=model,
                reasoning_effort=reasoning_effort,
                ignore={"agent_update"},
            )
        finally:
            renderer.close()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user message submission."""
        message = event.value.strip()
        if not message:
            return

        # Clear the input
        self.input_field.value = ""

        self.chat_logger.write_message(message, style="bold blue", prefix="User")
        await self._stream_agent_response(message)

    def action_reset_session(self) -> None:
        self.actions.reset_session()

    def action_load_session(self) -> None:
        self.actions.load_session()

    def action_create_task(self) -> None:
        self.actions.create_task()

    def action_list_tasks(self) -> None:
        self.actions.list_tasks()
