from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from ..tasks import TaskManager


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

    def compose(self) -> "ComposeResult":
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
                            yield Button("Delete", variant="error", id=f"delete-{task_id}",
                                         classes="task-delete-btn")

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
