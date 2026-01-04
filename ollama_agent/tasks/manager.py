"""Task management utilities."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from ..core import DEFAULT_REASONING_EFFORT, ReasoningEffortValue, validate_reasoning_effort

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
    def from_dict(cls, d: dict) -> Task:
        return cls(str(d.get("title", "")), str(d.get("prompt", "")), str(d.get("model", "")),
                   str(d.get("reasoning_effort", DEFAULT_REASONING_EFFORT)))


class TaskManager:
    """Manages task persistence using YAML files."""

    DEFAULT_DIR = Path.home() / ".ollama-agent" / "tasks"

    def __init__(self, tasks_dir: Path | None = None) -> None:
        self.tasks_dir = tasks_dir or self.DEFAULT_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.yaml"

    @staticmethod
    def validate_task_id(task_id: str) -> str:
        """Validate task_id: letters, numbers, underscore, dash only."""
        task_id = (task_id or "").strip()
        if not task_id or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise ValueError("Invalid task_id. Use only letters, numbers, '_' and '-'.")
        return task_id

    def save(self, task_id: str, task: Task, *, overwrite: bool = False) -> str:
        """Save a task and return its ID."""
        path = self._path(task_id := self.validate_task_id(task_id))
        if path.exists() and not overwrite:
            raise FileExistsError(f"Task already exists: {task_id}")
        path.write_text(yaml.safe_dump(asdict(task), allow_unicode=True), encoding="utf-8")
        return task_id

    def find_matches(self, prefix: str) -> list[tuple[str, Task]]:
        """Return all tasks whose id starts with prefix."""
        if not (prefix := (prefix or "").strip()):
            return []
        if (task := self.load(prefix)) is not None:  # Fast-path: exact match
            return [(prefix, task)]
        return [(p.stem, t) for p in self.tasks_dir.glob(f"{prefix}*.yaml") if (t := self.load(p.stem))]

    def load(self, task_id: str) -> Task | None:
        """Load a task by ID."""
        path = self._path(task_id)
        if not path.exists():
            return None
        try:
            return Task.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        except Exception as e:
            logger.error("Error loading task %s: %s", task_id, e)
            return None

    def delete(self, task_id: str) -> bool:
        """Delete a task by ID."""
        try:
            self._path(task_id).unlink()
            return True
        except (FileNotFoundError, OSError) as e:
            if isinstance(e, OSError):
                logger.error("Error deleting task %s: %s", task_id, e)
            return False

    def list_all(self) -> list[tuple[str, Task]]:
        """List all tasks sorted by title."""
        tasks = [(p.stem, t) for p in self.tasks_dir.glob("*.yaml") if (t := self.load(p.stem))]
        return sorted(tasks, key=lambda x: x[1].title.lower())
