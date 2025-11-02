from datetime import datetime
from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown

from ..agent import OllamaAgent
from ..tasks import Task, TaskManager
from ..utils import ALLOWED_REASONING_EFFORTS, validate_reasoning_effort

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
        if not button_id:
            return
        
        if button_id == "cancel-button":
            self.dismiss(None)
        elif button_id.startswith("load-"):
            session_id = button_id.removeprefix("load-")
            self.dismiss(f"load:{session_id}")
        elif button_id.startswith("delete-"):
            session_id = button_id.removeprefix("delete-")
            if self.agent.delete_session(session_id):
                self.sessions = self.agent.list_sessions()
                self.refresh(recompose=True)


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
