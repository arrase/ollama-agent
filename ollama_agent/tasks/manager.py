"""Task management utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
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
from ..settings.paths import TASKS_DIR

logger = logging.getLogger(__name__)


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
            reasoning_effort=d.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
        )


class TaskManager(BaseFileStoreManager[Task]):
    """Manages task persistence using YAML files."""

    DEFAULT_DIR = TASKS_DIR
    _ext: str = ".yaml"

    def __init__(self, tasks_dir: Path = TASKS_DIR) -> None:
        super().__init__(tasks_dir)

    @property
    def tasks_dir(self) -> Path:
        return self.base_dir

    @staticmethod
    def validate_task_id(task_id: str) -> str:
        """Validate task_id: letters, numbers, underscore, dash only."""
        return validate_identifier(task_id, "task_id")

    def save(self, task_id: str, task: Task, *, overwrite: bool = False) -> str:
        """Save a task and return its ID."""
        task_id = self.validate_task_id(task_id)
        path = self._path(task_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Task already exists: {task_id}")
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            yaml.safe_dump(asdict(task), allow_unicode=True), encoding="utf-8"
        )
        os.replace(tmp_path, path)
        return task_id

    def find_matches(self, prefix: str) -> list[tuple[str, Task]]:
        """Return all tasks whose id starts with prefix."""
        if not (prefix := prefix.strip()):
            return []
        if (task := self.get(prefix)) is not None:  # Fast-path: exact match
            return [(prefix, task)]
        return [
            (p.stem, t)
            for p in self.base_dir.iterdir()
            if p.is_file() and p.suffix == ".yaml" and p.stem.startswith(prefix) and (t := self.get(p.stem)) is not None
        ]

    def get(self, item_id: str) -> Task | None:
        """Retrieve a task by ID."""
        path = self._path(item_id)
        if not path.exists():
            return None
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            return Task.from_dict(raw)
        except (yaml.YAMLError, KeyError, TypeError, OSError) as e:
            logger.error("Error loading task %s: %s", item_id, e)
            return None

    def delete(self, item_id: str) -> bool:
        """Delete a task by ID."""
        try:
            self._path(item_id).unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.error("Error deleting task %s: %s", item_id, e)
            return False

    def list_all(self) -> list[tuple[str, Task]]:
        """List all tasks sorted by title."""
        tasks = [
            (p.stem, t)
            for p in self.base_dir.glob("*.yaml")
            if (t := self.get(p.stem))
        ]
        return sorted(tasks, key=lambda x: x[1].title.lower())

