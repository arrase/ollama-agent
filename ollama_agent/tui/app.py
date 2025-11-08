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
        self.chat_log: RichLog | None = None
        self.input_widget: Input | None = None
        set_builtin_tool_timeout(builtin_tool_timeout)

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
        self.sub_title = f"Model: {self.agent.model} | Session: {session_id[:8]}..." if session_id else f"Model: {self.agent.model}"

        self.chat_log = self.query_one("#chat-log", RichLog)
        self.input_widget = self.query_one("#user-input", Input)

        self._write_system_message("Welcome to Ollama Agent!")
        self._write_system_message(f"Session ID: {session_id}")
        self._write_system_message(
            "Type your message and press Enter to send. Use Ctrl+V to paste text.")
        self._write_system_message(
            "Shortcuts: Ctrl+R=New Session | Ctrl+S=Load Session | Ctrl+T=Create Task | Ctrl+L=Tasks")
        if self.chat_log is not None:
            self.chat_log.write("")

        if self.input_widget is not None:
            self.input_widget.focus()

    async def on_unmount(self) -> None:
        """Execute when the application is unmounted."""
        # Cleanup MCP servers
        await self.agent.cleanup()

    def _write_system_message(self, message: str) -> None:
        if self.chat_log is None:
            return
        self.chat_log.write(Text(message, style="italic cyan"))

    def _write_user_message(self, message: str) -> None:
        if self.chat_log is None:
            return
        self.chat_log.write(Text(f"User: {message}", style="bold blue"))

    def _write_agent_message(self, message: str) -> None:
        if self.chat_log is None:
            return
        self.chat_log.write(Text("Agent:", style="bold green"))
        self.chat_log.write(RichMarkdown(message))

    def _write_error_message(self, message: str) -> None:
        if self.chat_log is None:
            return
        self.chat_log.write(Text(f"Error: {message}", style="bold red"))

    async def _stream_agent_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Render a streamed agent response into the chat log."""
        if self.chat_log is None:
            return

        text_renderer = StreamingMarkdownRenderer(self.chat_log)
        reasoning_renderer = ReasoningRenderer(self.chat_log)

        try:
            async for event in self.agent.run_async_streamed(
                prompt,
                model=model,
                reasoning_effort=reasoning_effort,
            ):
                if event["type"] == "text_delta":
                    # Regular text tokens
                    if reasoning_renderer.is_active:
                        reasoning_renderer.finalize_reasoning()
                    # Append token (start_rendering called automatically on first token)
                    text_renderer.append_token(event["content"])
                
                elif event["type"] == "reasoning_delta":
                    # Reasoning/thinking tokens
                    reasoning_renderer.start_reasoning()
                    reasoning_renderer.append_reasoning_token(event["content"])
                
                elif event["type"] == "reasoning_summary":
                    # Full reasoning summary (if available)
                    if not reasoning_renderer.is_active:
                        self.chat_log.write(Text(f"💭 Reasoning: {event['content'][:100]}...", 
                                           style="dim italic magenta"))
                
                elif event["type"] == "tool_call":
                    if reasoning_renderer.is_active:
                        reasoning_renderer.finalize_reasoning()
                    self.chat_log.write(Text(f"🔧 Calling tool: {event['name']}", 
                                       style="bold yellow"))
                
                elif event["type"] == "tool_output":
                    output_preview = event["output"][:100] + "..." if len(event["output"]) > 100 else event["output"]
                    self.chat_log.write(Text(f"📤 Tool output: {output_preview}", 
                                       style="cyan"))
                
                elif event["type"] == "agent_update":
                    # Skip the initial agent update message
                    pass
                
                elif event["type"] == "error":
                    self._write_error_message(event["content"])
                    break

        except Exception as exc:
            self._write_error_message(str(exc))
        finally:
            if reasoning_renderer.is_active:
                reasoning_renderer.finalize_reasoning()
            text_renderer.finalize()
            if self.chat_log is not None:
                self.chat_log.write("")
                self.chat_log.scroll_end(animate=False)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user message submission."""
        message = event.value.strip()
        if not message:
            return

        # Clear the input
        if self.input_widget is not None:
            self.input_widget.value = ""

        self._write_user_message(message)
        await self._stream_agent_response(message)

    def action_reset_session(self) -> None:
        """Reset the session and start a new conversation."""
        session_id = self.agent.reset_session()
        if self.chat_log is not None:
            self.chat_log.clear()
        self._write_system_message("New session started!")
        self._write_system_message(f"Session ID: {session_id}")
        self._write_system_message(
            "Previous conversation history has been cleared.")
        if self.chat_log is not None:
            self.chat_log.write("")

        # Update subtitle with new session ID
        self.sub_title = f"Model: {self.agent.model} | Session: {session_id[:8]}..."

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
        if log is None:
            return

        log.clear()
        self._write_system_message(f"Loaded session: {session_id}")
        log.write("")

        history = await self.agent.get_session_history(session_id)

        for item in history:
            if isinstance(item, dict):
                role = item.get('role', 'unknown')
                content = item.get('content', '')
                text = extract_text(content)

                if role == 'user' and text:
                    self._write_user_message(text)
                elif role == 'assistant' and text:
                    self._write_agent_message(text)

        log.write("")
        log.scroll_end(animate=False)

        # Update subtitle
        self.sub_title = f"Model: {self.agent.model} | Session: {session_id[:8]}..."

    def action_create_task(self) -> None:
        """Show the create task dialog."""
        def handle_task_creation(task: Optional[Task]) -> None:
            """Handle the created task."""
            if task:
                task_id = self.task_manager.save_task(task)
                self._write_system_message(
                    f"Task saved: {task.title} ({task_id})")
                if self.chat_log is not None:
                    self.chat_log.write("")

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
            self._write_error_message(f"Task not found: {task_id}")
            return

        self._write_system_message(
            f"Executing task: {task.title} ({task_id})")
        self._write_user_message(task.prompt)

        await self._stream_agent_response(
            task.prompt,
            model=task.model,
            reasoning_effort=task.reasoning_effort,
        )
