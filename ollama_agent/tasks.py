"""Task management utilities."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

from .utils import DEFAULT_REASONING_EFFORT, ReasoningEffortValue, validate_reasoning_effort

logger = logging.getLogger(__name__)
_HASH_LENGTH = 8


@dataclass(slots=True)
class Task:
    title: str
    prompt: str
    model: str
    reasoning_effort: ReasoningEffortValue = field(default=DEFAULT_REASONING_EFFORT)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(self.reasoning_effort)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Task":
        return cls(
            title=str(data["title"]),
            prompt=str(data["prompt"]),
            model=str(data["model"]),
            reasoning_effort=validate_reasoning_effort(
                str(data.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
            ),
        )


def compute_task_id(title: str) -> str:
    digest = hashlib.blake2s(title.encode("utf-8"), digest_size=16).hexdigest()
    return digest[:_HASH_LENGTH]


class TaskManager:
    def __init__(self, tasks_dir: Optional[Path] = None) -> None:
        self.tasks_dir = tasks_dir or (Path.home() / ".ollama-agent" / "tasks")
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.yaml"

    def save_task(self, task: Task) -> str:
        task_id = compute_task_id(task.title)
        self._task_path(task_id).write_text(
            yaml.safe_dump(task.to_dict(), allow_unicode=True),
            encoding="utf-8",
        )
        return task_id

    def load_task(self, task_id: str) -> Optional[Task]:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return Task.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error loading task %s: %s", task_id, exc)
            return None

    def delete_task(self, task_id: str) -> bool:
        try:
            self._task_path(task_id).unlink()
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("Error deleting task %s: %s", task_id, exc)
            return False

    def list_tasks(self) -> list[tuple[str, Task]]:
        tasks = [
            (path.stem, task)
            for path in self.tasks_dir.glob("*.yaml")
            if (task := self.load_task(path.stem))
        ]
        return sorted(tasks, key=lambda item: item[1].title.lower())

    def find_task_by_prefix(self, prefix: str) -> Optional[tuple[str, Task]]:
        matches = [
            (path.stem, task)
            for path in self.tasks_dir.glob(f"{prefix}*.yaml")
            if (task := self.load_task(path.stem))
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning("Ambiguous task ID prefix '%s'. Matches:", prefix)
            for task_id, task in matches:
                logger.warning("  %s: %s", task_id, task.title)
        return None
