"""Task list modal screen."""

from typing import Iterable, cast

from textual.widget import Widget

from ...tasks import Task, TaskManager
from .base import ListModalScreen, make_row


class TaskListScreen(ListModalScreen):
    """Modal screen to list and manage tasks."""

    def __init__(self, task_manager: TaskManager):
        super().__init__("Saved Tasks", "No tasks found")
        self.task_manager = task_manager

    def get_items(self) -> Iterable[object]:
        return self.task_manager.list_tasks()

    def render_items(self, items: list[object]) -> Iterable[Widget]:
        for item in items:
            task_id, task = cast(tuple[str, Task], item)
            text = (
                f"[bold]{task.title}[/bold] ({task_id})\n"
                f"Model: {task.model} | Effort: {task.reasoning_effort}\n"
                f"{task.prompt[:50]}..."
            )
            yield make_row(text, task_id, [
                ("run", "Run", "primary"),
                ("delete", "Delete", "error"),
            ])

    def handle_action(self, action: str, item_id: str) -> bool:
        if action == "delete":
            self.task_manager.delete_task(item_id)
            return False  # Refresh
        return True  # Dismiss for "run"
