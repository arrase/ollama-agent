"""Task management utilities."""

from __future__ import annotations

import hashlib
import logging
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
    def from_dict(cls, data: dict) -> Task:
        return cls(
            title=str(data.get("title", "")),
            prompt=str(data.get("prompt", "")),
            model=str(data.get("model", "")),
            reasoning_effort=str(data.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), allow_unicode=True)


class TaskManager:
    """Manages task persistence using YAML files."""

    DEFAULT_DIR = Path.home() / ".ollama-agent" / "tasks"

    def __init__(self, tasks_dir: Path | None = None) -> None:
        self.tasks_dir = tasks_dir or self.DEFAULT_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.yaml"

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.blake2s(text.encode(), digest_size=4).hexdigest()

    def save(self, task: Task) -> str:
        """Save a task and return its ID."""
        task_id = self._hash(task.title)
        self._path(task_id).write_text(task.to_yaml(), encoding="utf-8")
        return task_id

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
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error("Error deleting task %s: %s", task_id, e)
            return False

    def list_all(self) -> list[tuple[str, Task]]:
        """List all tasks sorted by title."""
        tasks = [
            (p.stem, task)
            for p in self.tasks_dir.glob("*.yaml")
            if (task := self.load(p.stem))
        ]
        return sorted(tasks, key=lambda x: x[1].title.lower())

    def find_by_prefix(self, prefix: str) -> tuple[str, Task] | None:
        """Find a task by ID prefix. Returns None if ambiguous or not found."""
        matches = [
            (p.stem, task)
            for p in self.tasks_dir.glob(f"{prefix}*.yaml")
            if (task := self.load(p.stem))
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            logger.warning("Ambiguous prefix '%s': %s", prefix, [m[0] for m in matches])
        return None
