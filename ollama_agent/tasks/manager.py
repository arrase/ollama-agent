"""Task management utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..core import (
    DEFAULT_REASONING_EFFORT,
    BaseFileStoreManager,
    ReasoningEffortValue,
    atomic_write_text,
    validate_identifier,
    validate_reasoning_effort,
)
from ..i18n import _
from ..settings.config import render_prompt_template
from ..settings.paths import TASKS_DIR


@dataclass(slots=True)
class TaskInput:
    """Input definition for parameterized task templates."""

    description: str = ""
    default: Any = None
    required: bool = False
    type: str = "string"  # "string", "boolean", "number"


def _coerce_value(name: str, val: Any, expected_type: str) -> Any:
    if val is None:
        return None
    if expected_type == "boolean":
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            norm = val.strip().lower()
            if norm in ("true", "1", "yes"):
                return True
            if norm in ("false", "0", "no"):
                return False
        elif isinstance(val, (int, float)):
            if val in (1, 1.0):
                return True
            if val in (0, 0.0):
                return False
        raise ValueError(_("Invalid boolean value for input '{name}': {val}", name=name, val=val))
    if expected_type == "number":
        if isinstance(val, str):
            val_str = val.strip()
            try:
                return int(val_str)
            except ValueError:
                try:
                    return float(val_str)
                except ValueError:
                    pass
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            return val
        raise ValueError(_("Invalid number value for input '{name}': {val}", name=name, val=val))
    if expected_type == "string":
        if isinstance(val, str):
            return val
        return str(val)
    raise ValueError(_("Unsupported input type '{type}' for input '{name}'", type=expected_type, name=name))


@dataclass(slots=True)
class Task:
    """A saved task with title, prompt, model, reasoning effort, and inputs."""

    title: str
    prompt: str
    model: str
    reasoning_effort: ReasoningEffortValue = field(default=DEFAULT_REASONING_EFFORT)
    inputs: dict[str, TaskInput] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reasoning_effort = validate_reasoning_effort(self.reasoning_effort)

    @classmethod
    def from_dict(cls, d: Any) -> Task:
        if not isinstance(d, dict):
            raise ValueError(_("Expected mapping for Task, got {type_name}", type_name=type(d).__name__))
        inputs: dict[str, TaskInput] = {}
        if "inputs" in d:
            if not isinstance(d["inputs"], dict):
                raise ValueError(_("Expected mapping for inputs in Task"))
            for name, input_data in d["inputs"].items():
                if isinstance(input_data, TaskInput):
                    inputs[name] = input_data
                elif isinstance(input_data, dict):
                    inputs[name] = TaskInput(**input_data)
                else:
                    raise ValueError(_("Invalid input definition for '{name}'", name=name))
        return cls(
            title=d["title"],
            prompt=d["prompt"],
            model=d["model"],
            reasoning_effort=d["reasoning_effort"],
            inputs=inputs,
        )

    def render(self, variables: dict[str, Any] | None = None) -> str:
        """Render the task prompt template using provided variables."""
        merged_vars: dict[str, Any] = dict(variables) if variables else {}

        for name, inp in self.inputs.items():
            if name not in merged_vars and inp.default is not None:
                merged_vars[name] = inp.default
            if inp.required and (name not in merged_vars or merged_vars[name] is None):
                raise ValueError(_("Missing required input: {name}", name=name))
            if name in merged_vars:
                merged_vars[name] = _coerce_value(name, merged_vars[name], inp.type)

        return render_prompt_template(self.prompt, merged_vars)


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
        text = yaml.safe_dump(asdict(task), allow_unicode=True)
        atomic_write_text(path, text)
        return task_id

    def find_matches(self, prefix: str) -> list[tuple[str, Task]]:
        """Return all tasks whose id starts with prefix."""
        prefix = self.validate_task_id(prefix)
        matches = [(p.stem, self.get(p.stem)) for p in self.base_dir.glob(f"{prefix}*.yaml")]
        return sorted(matches, key=lambda x: x[0])

    def get(self, item_id: str) -> Task:
        """Retrieve a task by ID. Raise FileNotFoundError if missing."""
        path = self._path(self.validate_task_id(item_id))
        if not path.is_file():
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
