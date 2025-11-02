"""Terminal user interface (TUI) using Textual."""

from datetime import datetime
from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown

from .agent import ALLOWED_REASONING_EFFORTS, OllamaAgent, validate_reasoning_effort
from .tasks import Task, TaskManager


class SessionListScreen(ModalScreen):
    """Modal screen to list and select sessions."""
    
    CSS = """
    SessionListScreen {
        align: center middle;
    }
    
    #session-dialog {
        width: 90;
        height: 30;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
        overflow-x: hidden;
    }
    
    #session-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $accent;
    }
    
    #session-list {
        height: 18;
        border: solid $primary;
        margin: 1 0;
        overflow-x: hidden;
    }
    
    #button-container {
        height: 3;
        align: center middle;
    }
    
    .session-row {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    
    .session-info {
        width: 3fr;
        height: auto;
    }
    
    .session-load-btn {
        width: 1fr;
        min-width: 10;
        margin-left: 1;
    }
    
    .session-delete-btn {
        width: 1fr;
        min-width: 10;
        margin-left: 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]
    
    def __init__(self, sessions: list[dict], agent: OllamaAgent):
        """
        Initialize the session list screen.
        
        Args:
            sessions: List of session dictionaries.
            agent: The agent instance for deleting sessions.
        """
        super().__init__()
        self.sessions = sessions
        self.agent = agent
    
    def compose(self) -> ComposeResult:
        """Create the session list dialog."""
        with Container(id="session-dialog"):
            yield Label("Select a Session to Load or Delete", id="session-title")
            
            if not self.sessions:
                yield Label("No sessions found", id="no-sessions")
            else:
                with VerticalScroll(id="session-list"):
                    for session in self.sessions:
                        session_id = session['session_id']
                        count = session['message_count']
                        preview = session['preview']
                        last_time = session.get('last_message', 'Unknown')
                        
                        # Format the time
                        time_str = last_time
                        if last_time != 'Unknown':
                            try:
                                dt = datetime.fromisoformat(last_time)
                                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                            except:
                                pass
                        
                        # Create a row with session info and buttons
                        with Horizontal(classes="session-row"):
                            session_text = f"[bold]{session_id[:8]}...[/bold] ({count} msgs)\n{time_str}\n{preview[:40]}..."
                            yield Label(session_text, classes="session-info")
                            yield Button("Load", variant="primary", id=f"load-{session_id}", classes="session-load-btn")
                            yield Button("Delete", variant="error", id=f"delete-{session_id}", classes="session-delete-btn")
            
            with Container(id="button-container"):
                yield Button("Close", variant="default", id="cancel-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        button_id = event.button.id
        
        if button_id == "cancel-button":
            self.dismiss(None)
        elif button_id and button_id.startswith("load-"):
            session_id = button_id.replace("load-", "")
            self.dismiss(f"load:{session_id}")
        elif button_id and button_id.startswith("delete-"):
            session_id = button_id.replace("delete-", "")
            # Delete the session
            success = self.agent.delete_session(session_id)
            if success:
                # Refresh the session list
                self.sessions = self.agent.list_sessions()
                self.refresh(recompose=True)
            else:
                # Could show an error message here
                pass


class CreateTaskScreen(ModalScreen):
    """Modal screen to create a new task."""
    
    CSS = """
    CreateTaskScreen {
        align: center middle;
    }
    
    #task-dialog {
        width: 80;
        height: 30;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
    }
    
    #task-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $accent;
    }
    
    .field-label {
        margin-top: 1;
        text-style: bold;
    }
    
    .field-input {
        margin-bottom: 1;
    }
    
    #button-container {
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]
    
    def __init__(self, agent: OllamaAgent):
        """
        Initialize the create task screen.
        
        Args:
            agent: The agent instance for getting current settings.
        """
        super().__init__()
        self.agent = agent
    
    def compose(self) -> ComposeResult:
        """Create the task creation dialog."""
        with Container(id="task-dialog"):
            yield Label("Create New Task", id="task-title")
            
            yield Label("Title:", classes="field-label")
            yield Input(placeholder="Enter task title...", id="task-title-input", classes="field-input")
            
            yield Label("Prompt:", classes="field-label")
            yield Input(placeholder="Enter task prompt...", id="task-prompt-input", classes="field-input")
            
            yield Label("Model:", classes="field-label")
            yield Input(value=self.agent.model, id="task-model-input", classes="field-input")
            
            yield Label("Reasoning Effort:", classes="field-label")
            yield Select(
                [(effort, effort) for effort in ALLOWED_REASONING_EFFORTS],
                value=self.agent.reasoning_effort,
                id="task-effort-select",
                classes="field-input"
            )
            
            with Horizontal(id="button-container"):
                yield Button("Save", variant="primary", id="save-button")
                yield Button("Cancel", variant="default", id="cancel-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "save-button":
            task = self._create_task_from_inputs()
            if task:
                self.dismiss(task)
    
    def _create_task_from_inputs(self) -> Optional[Task]:
        """
        Create a task from user inputs.
        
        Returns:
            Task instance if all inputs are valid, None otherwise.
        """
        title = self.query_one("#task-title-input", Input).value.strip()
        prompt = self.query_one("#task-prompt-input", Input).value.strip()
        model = self.query_one("#task-model-input", Input).value.strip()
        
        if not all([title, prompt, model]):
            return None
        
        effort_select = self.query_one("#task-effort-select", Select)
        effort = str(effort_select.value) if effort_select.value else self.agent.reasoning_effort
        
        return Task(
            title=title,
            prompt=prompt,
            model=model,
            reasoning_effort=validate_reasoning_effort(effort)
        )


class TaskListScreen(ModalScreen):
    """Modal screen to list, execute, and delete tasks."""
    
    CSS = """
    TaskListScreen {
        align: center middle;
    }
    
    #task-list-dialog {
        width: 90;
        height: 30;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
        overflow-x: hidden;
    }
    
    #task-list-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $accent;
    }
    
    #tasks-list {
        height: 18;
        border: solid $primary;
        margin: 1 0;
        overflow-x: hidden;
    }
    
    #button-container {
        height: 3;
        align: center middle;
    }
    
    .task-row {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    
    .task-info {
        width: 3fr;
        height: auto;
    }
    
    .task-run-btn {
        width: 1fr;
        min-width: 10;
        margin-left: 1;
    }
    
    .task-delete-btn {
        width: 1fr;
        min-width: 10;
        margin-left: 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]
    
    def __init__(self, task_manager: TaskManager):
        """
        Initialize the task list screen.
        
        Args:
            task_manager: The task manager instance.
        """
        super().__init__()
        self.task_manager = task_manager
        self.tasks = task_manager.list_tasks()
    
    def compose(self) -> ComposeResult:
        """Create the task list dialog."""
        with Container(id="task-list-dialog"):
            yield Label("Saved Tasks", id="task-list-title")
            
            if not self.tasks:
                yield Label("No tasks found", id="no-tasks")
            else:
                with VerticalScroll(id="tasks-list"):
                    for task_id, task in self.tasks:
                        # Create a row with task info and buttons
                        with Horizontal(classes="task-row"):
                            task_text = f"[bold]{task.title}[/bold] ({task_id})\nModel: {task.model} | Effort: {task.reasoning_effort}\n{task.prompt[:50]}..."
                            yield Label(task_text, classes="task-info")
                            yield Button("Run", variant="primary", id=f"run-{task_id}", classes="task-run-btn")
                            yield Button("Delete", variant="error", id=f"delete-{task_id}", classes="task-delete-btn")
            
            with Container(id="button-container"):
                yield Button("Close", variant="default", id="cancel-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        button_id = event.button.id
        
        if not button_id:
            return
        
        if button_id == "cancel-button":
            self.dismiss(None)
        elif button_id.startswith("run-"):
            task_id = button_id.removeprefix("run-")
            self.dismiss(f"run:{task_id}")
        elif button_id.startswith("delete-"):
            task_id = button_id.removeprefix("delete-")
            self._delete_and_refresh(task_id)
    
    def _delete_and_refresh(self, task_id: str) -> None:
        """Delete a task and refresh the list."""
        if self.task_manager.delete_task(task_id):
            self.tasks = self.task_manager.list_tasks()
            self.refresh(recompose=True)


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
    
    def __init__(self, agent: OllamaAgent):
        """
        Initialize the interface.
        
        Args:
            agent: The AI agent to use.
        """
        super().__init__()
        self.agent = agent
        self.task_manager = TaskManager()

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Normalize agent history payloads into plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("text")
            ).strip()
        return ""

    def compose(self) -> ComposeResult:
        """Create the interface widgets."""
        yield Header()
        
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        
        with Container(id="input-container"):
            yield Input(placeholder="Type your message here...", id="user-input")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Execute when the application is mounted."""
        session_id = self.agent.get_session_id()
        self.title = "Ollama Agent - Chat"
        self.sub_title = f"Model: {self.agent.model} | Session: {session_id[:8]}..." if session_id else f"Model: {self.agent.model}"
        
        chat_log = self.query_one("#chat-log", RichLog)
        self._write_system_message(chat_log, "Welcome to Ollama Agent!")
        self._write_system_message(chat_log, f"Session ID: {session_id}")
        self._write_system_message(chat_log, "Type your message and press Enter to send.")
        self._write_system_message(chat_log, "Shortcuts: Ctrl+R=New Session | Ctrl+S=Load Session | Ctrl+T=Create Task | Ctrl+L=Tasks")
        chat_log.write("")
        
        # Focus the input
        self.query_one("#user-input", Input).focus()
    
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
        
        # Show thinking indicator
        thinking_text = Text("Agent: thinking...", style="italic yellow")
        chat_log.write(thinking_text)
        chat_log.scroll_end(animate=False)
        
        try:
            response = await self.agent.run_async(message)
            self._write_agent_message(chat_log, response)
        except Exception as e:
            self._write_error_message(chat_log, str(e))
        
        chat_log.write("")
        chat_log.scroll_end(animate=False)
    
    def action_reset_session(self) -> None:
        """Reset the session and start a new conversation."""
        session_id = self.agent.reset_session()
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        self._write_system_message(chat_log, "New session started!")
        self._write_system_message(chat_log, f"Session ID: {session_id}")
        self._write_system_message(chat_log, "Previous conversation history has been cleared.")
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
        
        self.push_screen(SessionListScreen(sessions, self.agent), handle_session_action)
    
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
                text = self._extract_text(content)

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
                self._write_system_message(chat_log, f"Task saved: {task.title} ({task_id})")
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
        self._write_system_message(chat_log, f"Executing task: {task.title} ({task_id})")
        self._write_user_message(chat_log, task.prompt)
        
        # Show thinking indicator
        thinking_text = Text("Agent: thinking...", style="italic yellow")
        chat_log.write(thinking_text)
        chat_log.scroll_end(animate=False)
        
        try:
            response = await self.agent.run_async(task.prompt, model=task.model, reasoning_effort=task.reasoning_effort)
            self._write_agent_message(chat_log, response)
        except Exception as e:
            self._write_error_message(chat_log, str(e))
        
        chat_log.write("")
        chat_log.scroll_end(animate=False)
