from typing import Iterable, Sequence, cast

from textual.containers import Horizontal
from textual.widgets import Button, Label

from ..tasks import Task, TaskManager
from .list_modal import ListModalScreen


class TaskListScreen(ListModalScreen):
    """Modal screen to list, execute, and delete tasks."""

    def __init__(self, task_manager: TaskManager):
        super().__init__("Saved Tasks", "No tasks found")
        self.task_manager = task_manager

    def load_items(self) -> Iterable[object]:
        return self.task_manager.list_tasks()

    def render_rows(self, items: Sequence[object]):
        for item in items:
            task_id, task = cast(tuple[str, Task], item)
            text = (
                f"[bold]{task.title}[/bold] ({task_id})\n"
                f"Model: {task.model} | Effort: {task.reasoning_effort}\n"
                f"{task.prompt[:50]}..."
            )
            yield Horizontal(
                Label(text, classes="entry-info"),
                Button("Run", variant="primary", id=f"run-{task_id}", classes="entry-btn"),
                Button("Delete", variant="error", id=f"delete-{task_id}", classes="entry-btn"),
                classes="entry-row",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "cancel-button":
            self.dismiss(None)
        elif button_id.startswith("run-"):
            self.dismiss(f"run:{button_id.removeprefix('run-')}")
        elif button_id.startswith("delete-"):
            task_id = button_id.removeprefix("delete-")
            if self.task_manager.delete_task(task_id):
                self.refresh(recompose=True)
