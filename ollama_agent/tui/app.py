"""Terminal user interface (TUI) using Textual."""

from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown

from ..agent import OllamaAgent
from ..tasks import Task, TaskManager
from ..tools import set_builtin_tool_timeout
from ..utils import extract_text
from .session_list_screen import SessionListScreen
from .create_task_screen import CreateTaskScreen
from .task_list_screen import TaskListScreen


class StreamingMarkdownRenderer:
    """Render markdown content progressively inside a RichLog widget."""

    def __init__(self, chat_log: RichLog, update_frequency: int = 5):
        self.chat_log = chat_log
        self.update_frequency = max(1, update_frequency)
        self.buffer = ""
        self.token_count = 0
        self._start_line_count = 0
        self._rendering_started = False

    def start_rendering(self) -> None:
        """Prepare the log for streaming output (only once)."""
        if not self._rendering_started:
            self.chat_log.write(Text("Agent:", style="bold green"))
            self._start_line_count = len(self.chat_log.lines)
            self._rendering_started = True

    def append_token(self, token: str) -> None:
        """Append a token and refresh the view at the configured cadence."""
        if not self._rendering_started:
            self.start_rendering()
        
        self.buffer += token
        self.token_count += 1
        if self.token_count % self.update_frequency == 0:
            self._update_display()

    def finalize(self) -> None:
        """Render the buffered markdown one last time."""
        self._update_display()

    def _update_display(self) -> None:
        if not self.buffer:
            return

        current_lines = len(self.chat_log.lines)
        for _ in range(current_lines - self._start_line_count):
            if len(self.chat_log.lines) > self._start_line_count:
                self.chat_log.lines.pop()

        # Directly clear the private _line_cache attribute.
        # There is no public API for clearing the line cache in RichLog.
        # This is necessary to ensure the markdown is re-rendered correctly.
        # If a public method becomes available in future versions of Textual, use that instead.
        self.chat_log._line_cache.clear()
        self.chat_log.write(RichMarkdown(self.buffer))
        self.chat_log.scroll_end(animate=False)
        self.chat_log.refresh()


class ReasoningRenderer:
    """Render reasoning/thinking process in real-time."""

    def __init__(self, chat_log: RichLog):
        self.chat_log = chat_log
        self.reasoning_buffer = ""
        self.is_active = False
        self._start_line_count = 0
        self._update_counter = 0
        self._update_frequency = 5  # Update every N tokens for performance

    def start_reasoning(self) -> None:
        """Start capturing reasoning tokens."""
        if not self.is_active:
            self.chat_log.write("")  # Blank line before reasoning
            self.is_active = True
            self.reasoning_buffer = ""
            self._start_line_count = len(self.chat_log.lines)
            self._update_counter = 0

    def append_reasoning_token(self, token: str) -> None:
        """Append a reasoning token and update display in real-time."""
        self.reasoning_buffer += token
        self._update_counter += 1
        
        # Update display every N tokens for performance
        if self._update_counter % self._update_frequency == 0:
            self._update_display()

    def finalize_reasoning(self) -> None:
        """Finish the reasoning display with final update."""
        if self.is_active and self.reasoning_buffer:
            # Final update to ensure all tokens are shown
            self._update_display()
            self.chat_log.write("")  # Add a blank line after reasoning
        self.is_active = False
        self.reasoning_buffer = ""
        self._start_line_count = 0

    def _update_display(self) -> None:
        """Update the reasoning line in the chat log."""
        # Remove any lines we added before (if updating)
        current_line_count = len(self.chat_log.lines)
        lines_to_remove = current_line_count - self._start_line_count
        
        for _ in range(lines_to_remove):
            if len(self.chat_log.lines) > self._start_line_count:
                self.chat_log.lines.pop()
        
        # Clear the line cache to force re-render
        if hasattr(self.chat_log, '_line_cache'):
            self.chat_log._line_cache.clear()
        
        # Write the updated reasoning line
        reasoning_line = Text()
        reasoning_line.append("🧠 Thinking: ", style="bold magenta")
        reasoning_line.append(self.reasoning_buffer, style="dim italic magenta")
        self.chat_log.write(reasoning_line)
        
        # Scroll to end to show the update
        self.chat_log.scroll_end(animate=False)
        self.chat_log.refresh()


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
        """
        Initialize the interface.

        Args:
            agent: The AI agent to use.
            builtin_tool_timeout: Built-in tool execution timeout in seconds.
        """
        super().__init__()
        self.agent = agent
        self.task_manager = TaskManager()
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

        chat_log = self.query_one("#chat-log", RichLog)
        self._write_system_message(chat_log, "Welcome to Ollama Agent!")
        self._write_system_message(chat_log, f"Session ID: {session_id}")
        self._write_system_message(
            chat_log, "Type your message and press Enter to send. Use Ctrl+V to paste text.")
        self._write_system_message(
            chat_log, "Shortcuts: Ctrl+R=New Session | Ctrl+S=Load Session | Ctrl+T=Create Task | Ctrl+L=Tasks")
        chat_log.write("")

        # Focus the input
        self.query_one("#user-input", Input).focus()

    async def on_unmount(self) -> None:
        """Execute when the application is unmounted."""
        # Cleanup MCP servers
        await self.agent.cleanup()

    def _write_system_message(self, chat_log: RichLog, message: str) -> None:
        """
        Write a system message to the chat log.

        Args:
            chat_log: The RichLog widget.
            message: The message to write.
        """
        chat_log.write(Text(message, style="italic cyan"))

    def _write_user_message(self, chat_log: RichLog, message: str) -> None:
        """
        Write a user message to the chat log.

        Args:
            chat_log: The RichLog widget.
            message: The message to write.
        """
        chat_log.write(Text(f"User: {message}", style="bold blue"))

    def _write_agent_message(self, chat_log: RichLog, message: str) -> None:
        """
        Write an agent message to the chat log.

        Args:
            chat_log: The RichLog widget.
            message: The message to write.
        """
        chat_log.write(Text("Agent:", style="bold green"))
        markdown = RichMarkdown(message)
        chat_log.write(markdown)

    def _write_error_message(self, chat_log: RichLog, message: str) -> None:
        """
        Write an error message to the chat log.

        Args:
            chat_log: The RichLog widget.
            message: The message to write.
        """
        chat_log.write(Text(f"Error: {message}", style="bold red"))

    async def _stream_agent_response(
        self,
        prompt: str,
        chat_log: RichLog,
        *,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Render a streamed agent response into the chat log."""
        text_renderer = StreamingMarkdownRenderer(chat_log)
        reasoning_renderer = ReasoningRenderer(chat_log)

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
                        chat_log.write(Text(f"💭 Reasoning: {event['content'][:100]}...", 
                                           style="dim italic magenta"))
                
                elif event["type"] == "tool_call":
                    if reasoning_renderer.is_active:
                        reasoning_renderer.finalize_reasoning()
                    chat_log.write(Text(f"🔧 Calling tool: {event['name']}", 
                                       style="bold yellow"))
                
                elif event["type"] == "tool_output":
                    output_preview = event["output"][:100] + "..." if len(event["output"]) > 100 else event["output"]
                    chat_log.write(Text(f"📤 Tool output: {output_preview}", 
                                       style="cyan"))
                
                elif event["type"] == "agent_update":
                    # Skip the initial agent update message
                    pass
                
                elif event["type"] == "error":
                    self._write_error_message(chat_log, event["content"])
                    break

        except Exception as exc:
            self._write_error_message(chat_log, str(exc))
        finally:
            if reasoning_renderer.is_active:
                reasoning_renderer.finalize_reasoning()
            text_renderer.finalize()
            chat_log.write("")
            chat_log.scroll_end(animate=False)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user message submission."""
        message = event.value.strip()
        if not message:
            return

        # Clear the input
        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""

        chat_log = self.query_one("#chat-log", RichLog)
        self._write_user_message(chat_log, message)
        await self._stream_agent_response(message, chat_log)

    def action_reset_session(self) -> None:
        """Reset the session and start a new conversation."""
        session_id = self.agent.reset_session()
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        self._write_system_message(chat_log, "New session started!")
        self._write_system_message(chat_log, f"Session ID: {session_id}")
        self._write_system_message(
            chat_log, "Previous conversation history has been cleared.")
        chat_log.write("")

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

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()

        self._write_system_message(chat_log, f"Loaded session: {session_id}")
        chat_log.write("")

        history = await self.agent.get_session_history(session_id)

        for item in history:
            if isinstance(item, dict):
                role = item.get('role', 'unknown')
                content = item.get('content', '')
                text = extract_text(content)

                if role == 'user' and text:
                    self._write_user_message(chat_log, text)
                elif role == 'assistant' and text:
                    self._write_agent_message(chat_log, text)

        chat_log.write("")
        chat_log.scroll_end(animate=False)

        # Update subtitle
        self.sub_title = f"Model: {self.agent.model} | Session: {session_id[:8]}..."

    def action_create_task(self) -> None:
        """Show the create task dialog."""
        def handle_task_creation(task: Optional[Task]) -> None:
            """Handle the created task."""
            if task:
                task_id = self.task_manager.save_task(task)
                chat_log = self.query_one("#chat-log", RichLog)
                self._write_system_message(
                    chat_log, f"Task saved: {task.title} ({task_id})")
                chat_log.write("")

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
            chat_log = self.query_one("#chat-log", RichLog)
            self._write_error_message(chat_log, f"Task not found: {task_id}")
            return

        chat_log = self.query_one("#chat-log", RichLog)
        self._write_system_message(
            chat_log, f"Executing task: {task.title} ({task_id})")
        self._write_user_message(chat_log, task.prompt)

        await self._stream_agent_response(
            task.prompt,
            chat_log,
            model=task.model,
            reasoning_effort=task.reasoning_effort,
        )
