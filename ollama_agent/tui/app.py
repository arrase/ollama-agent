"""Terminal user interface (TUI) using Textual."""

from typing import Optional

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog

from ..agent import OllamaAgent
from ..tasks import Task, TaskManager
from ..agent.tools import set_builtin_tool_timeout
from ..utils import extract_text
from .create_task_screen import CreateTaskScreen
from .renderers import ReasoningRenderer, StreamingMarkdownRenderer
from .session_list_screen import SessionListScreen
from .task_list_screen import TaskListScreen


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

    def __init__(self, agent: OllamaAgent, builtin_tool_timeout: int = 30):
        super().__init__()
        self.agent = agent
        self.task_manager = TaskManager()
        self._chat_log: RichLog | None = None
        self._input_field: Input | None = None
        set_builtin_tool_timeout(builtin_tool_timeout)

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

    def _blank_line(self) -> None:
        self.chat_log.write("")

    def _set_subtitle(self, session_id: Optional[str]) -> None:
        session_piece = f" | Session: {session_id[:8]}..." if session_id else ""
        self.sub_title = f"Model: {self.agent.model}{session_piece}"

    def _write_message(
        self,
        message: str,
        *,
        style: str,
        prefix: Optional[str] = None,
        markdown: bool = False,
    ) -> None:
        if markdown:
            if prefix:
                self.chat_log.write(Text(f"{prefix}:", style=style))
            self.chat_log.write(RichMarkdown(message))
            return

        text = f"{prefix}: {message}" if prefix else message
        self.chat_log.write(Text(text, style=style))

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

    def on_mount(self) -> None:
        """Execute when the application is mounted."""
        session_id = self.agent.get_session_id()
        self.title = "Ollama Agent - Chat"
        self._chat_log = self.query_one("#chat-log", RichLog)
        self._input_field = self.query_one("#user-input", Input)

        self._set_subtitle(session_id)
        self._write_message("Welcome to Ollama Agent!", style="italic cyan")
        self._write_message(
            f"Session ID: {session_id}", style="italic cyan")
        self._write_message(
            "Type your message and press Enter to send. Use Ctrl+V to paste text.",
            style="italic cyan",
        )
        self._write_message(
            "Shortcuts: Ctrl+R=New Session | Ctrl+S=Load Session | Ctrl+T=Create Task | Ctrl+L=Tasks",
            style="italic cyan",
        )
        self._blank_line()
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
        text_renderer = StreamingMarkdownRenderer(self.chat_log)
        reasoning_renderer = ReasoningRenderer(self.chat_log)

        def handle_text_delta(event: dict) -> None:
            if reasoning_renderer.is_active:
                reasoning_renderer.finalize_reasoning()
            text_renderer.append_token(event.get("content", ""))

        def handle_reasoning_delta(event: dict) -> None:
            reasoning_renderer.start_reasoning()
            reasoning_renderer.append_reasoning_token(event.get("content", ""))

        def handle_reasoning_summary(event: dict) -> None:
            if reasoning_renderer.is_active:
                return
            preview = event.get("content", "")[:100]
            if preview:
                self.chat_log.write(Text(f"💭 Reasoning: {preview}...",
                                         style="dim italic magenta"))

        def handle_tool_call(event: dict) -> None:
            if reasoning_renderer.is_active:
                reasoning_renderer.finalize_reasoning()
            tool_name = event.get("name", "unknown tool")
            self.chat_log.write(Text(f"🔧 Calling tool: {tool_name}",
                                     style="bold yellow"))

        def handle_tool_output(event: dict) -> None:
            output = event.get("output", "")
            preview = f"{output[:100]}..." if len(output) > 100 else output
            self.chat_log.write(Text(f"📤 Tool output: {preview}",
                                     style="cyan"))

        event_handlers = {
            "text_delta": handle_text_delta,
            "reasoning_delta": handle_reasoning_delta,
            "reasoning_summary": handle_reasoning_summary,
            "tool_call": handle_tool_call,
            "tool_output": handle_tool_output,
        }

        try:
            async for event in self.agent.run_async_streamed(
                prompt,
                model=model,
                reasoning_effort=reasoning_effort,
            ):
                event_type = event.get("type")

                if event_type == "error":
                    self._write_message(
                        event.get("content", "Unknown error"),
                        style="bold red",
                        prefix="Error",
                    )
                    break

                if event_type == "agent_update":
                    continue

                if not isinstance(event_type, str):
                    continue

                handler = event_handlers.get(event_type)
                if handler:
                    handler(event)

        except Exception as exc:
            self._write_message(str(exc), style="bold red", prefix="Error")
        finally:
            if reasoning_renderer.is_active:
                reasoning_renderer.finalize_reasoning()
            text_renderer.finalize()
            self._blank_line()
            self.chat_log.scroll_end(animate=False)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user message submission."""
        message = event.value.strip()
        if not message:
            return

        # Clear the input
        self.input_field.value = ""

        self._write_message(message, style="bold blue", prefix="User")
        await self._stream_agent_response(message)

    def action_reset_session(self) -> None:
        """Reset the session and start a new conversation."""
        session_id = self.agent.reset_session()
        self.chat_log.clear()
        self._write_message("New session started!", style="italic cyan")
        self._write_message(
            f"Session ID: {session_id}", style="italic cyan")
        self._write_message(
            "Previous conversation history has been cleared.",
            style="italic cyan",
        )
        self._blank_line()

        # Update subtitle with new session ID
        self._set_subtitle(session_id)

    def action_load_session(self) -> None:
        """Show the session list dialog."""
        sessions = self.agent.list_sessions()

        async def handle_session_action(action: str | None) -> None:
            """Handle the selected action."""
            if action and action.startswith("load:"):
                session_id = action.replace("load:", "")
                await self._load_selected_session(session_id)

        self.push_screen(SessionListScreen(
            sessions, self.agent), handle_session_action)

    async def _load_selected_session(self, session_id: str) -> None:
        """Load the selected session and display its history."""
        self.agent.load_session(session_id)

        log = self.chat_log
        log.clear()
        self._write_message(
            f"Loaded session: {session_id}", style="italic cyan")
        self._blank_line()

        history = await self.agent.get_session_history(session_id)

        for item in history:
            if isinstance(item, dict):
                role = item.get('role', 'unknown')
                content = item.get('content', '')
                text = extract_text(content)

                if role == 'user' and text:
                    self._write_message(text, style="bold blue", prefix="User")
                elif role == 'assistant' and text:
                    self._write_message(
                        text,
                        style="bold green",
                        prefix="Agent",
                        markdown=True,
                    )

        self._blank_line()
        log.scroll_end(animate=False)

        # Update subtitle
        self._set_subtitle(session_id)

    def action_create_task(self) -> None:
        """Show the create task dialog."""
        def handle_task_creation(task: Optional[Task]) -> None:
            """Handle the created task."""
            if task:
                task_id = self.task_manager.save_task(task)
                self._write_message(
                    f"Task saved: {task.title} ({task_id})",
                    style="italic cyan",
                )
                self._blank_line()

        self.push_screen(CreateTaskScreen(self.agent), handle_task_creation)

    def action_list_tasks(self) -> None:
        """Show the task list dialog."""
        def handle_task_action(action: Optional[str]) -> None:
            """Handle the selected action."""
            if action and action.startswith("run:"):
                task_id = action.replace("run:", "")
                self.run_worker(self._run_selected_task(task_id))

        self.push_screen(TaskListScreen(self.task_manager), handle_task_action)

    async def _run_selected_task(self, task_id: str) -> None:
        """Execute the selected task."""
        task = self.task_manager.load_task(task_id)

        if not task:
            self._write_message(
                f"Task not found: {task_id}", style="bold red", prefix="Error")
            return

        self._write_message(
            f"Executing task: {task.title} ({task_id})",
            style="italic cyan",
        )
        self._write_message(task.prompt, style="bold blue", prefix="User")

        await self._stream_agent_response(
            task.prompt,
            model=task.model,
            reasoning_effort=task.reasoning_effort,
        )
