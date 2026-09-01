"""Task management utilities."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..core import (
    BaseFileStoreManager,
    DEFAULT_REASONING_EFFORT,
    ReasoningEffortValue,
    validate_reasoning_effort,
    validate_identifier,
)
from ..i18n import _
from ..settings.paths import TASKS_DIR


@dataclass(slots=True)
class Task:
    """A saved task with title, prompt, model, and reasoning effort."""

    title: str
    prompt: str
    model: str
    reasoning_effort: ReasoningEffortValue = field(default=DEFAULT_REASONING_EFFORT)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(self.reasoning_effort)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            title=d["title"],
            prompt=d["prompt"],
            model=d["model"],
            reasoning_effort=d["reasoning_effort"],
        )


class TaskManager(BaseFileStoreManager[Task]):
    """Manages task persistence using YAML files."""

    _ext: str = ".yaml"

    def __init__(self, tasks_dir: Path = TASKS_DIR) -> None:
        super().__init__(tasks_dir)

    @staticmethod
    def validate_task_id(task_id: str) -> str:
        """Validate task_id: letters, numbers, underscore, dash only."""
        return validate_identifier(task_id, "task_id")

    def save(self, task_id: str, task: Task, *, overwrite: bool = False) -> str:
        """Save a task and return its ID."""
        task_id = self.validate_task_id(task_id)
        path = self._path(task_id)
        if path.exists() and not overwrite:
            raise FileExistsError(_("Task already exists: {task_id}", task_id=task_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            yaml.safe_dump(asdict(task), allow_unicode=True), encoding="utf-8"
        )
        os.replace(tmp_path, path)
        return task_id

    def find_matches(self, prefix: str) -> list[tuple[str, Task]]:
        """Return all tasks whose id starts with prefix."""
        prefix = self.validate_task_id(prefix)
        try:
            return [(prefix, self.get(prefix))]
        except FileNotFoundError:
            pass
        return [
            (p.stem, self.get(p.stem))
            for p in self.base_dir.glob(f"{prefix}*.yaml")
        ]

    def get(self, item_id: str) -> Task:
        """Retrieve a task by ID. Raise FileNotFoundError if missing."""
        path = self._path(self.validate_task_id(item_id))
        if not path.exists():
            raise FileNotFoundError(str(path))
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Task.from_dict(raw)

    def delete(self, item_id: str) -> None:
        """Delete a task by ID. Raise FileNotFoundError if missing."""
        self._path(self.validate_task_id(item_id)).unlink()

    def list_all(self) -> list[tuple[str, Task]]:
        """List all tasks sorted by title."""
        tasks = [(p.stem, self.get(p.stem)) for p in self.base_dir.glob("*.yaml")]
        return sorted(tasks, key=lambda x: x[1].title.lower())
