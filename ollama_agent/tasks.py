"""Task management for saving and executing agent prompts."""

import hashlib
import logging
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .utils import DEFAULT_REASONING_EFFORT, ReasoningEffortValue, validate_reasoning_effort

logger = logging.getLogger(__name__)
HASH_LENGTH = 8  # Use first 8 characters of Blake2s hash


@dataclass
class Task:
    """A saved task with prompt and execution parameters."""
    title: str
    prompt: str
    model: str
    reasoning_effort: ReasoningEffortValue
    
    def to_dict(self) -> dict:
        """Convert task to dictionary for YAML serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """Create task from dictionary loaded from YAML."""
        return cls(
            title=data['title'],
            prompt=data['prompt'],
            model=data['model'],
            reasoning_effort=validate_reasoning_effort(data.get('reasoning_effort', DEFAULT_REASONING_EFFORT))
        )


def compute_task_id(title: str) -> str:
    """Compute a short task ID from the title using Blake2s."""
    hash_digest = hashlib.blake2s(title.encode('utf-8'), digest_size=16).hexdigest()
    return hash_digest[:HASH_LENGTH]


class TaskManager:
    """Manages task storage and retrieval."""
    
    def __init__(self, tasks_dir: Optional[Path] = None):
        """Initialize the task manager."""
        self.tasks_dir = tasks_dir or (Path.home() / ".ollama-agent" / "tasks")
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
    
    def save_task(self, task: Task) -> str:
        """Save a task to disk and return its ID."""
        task_id = compute_task_id(task.title)
        task_path = self.tasks_dir / f"{task_id}.yaml"
        
        with open(task_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(task.to_dict(), f, default_flow_style=False, allow_unicode=True)
        
        return task_id
    
    def load_task(self, task_id: str) -> Optional[Task]:
        """Load a task from disk."""
        task_path = self.tasks_dir / f"{task_id}.yaml"
        
        if not task_path.exists():
            return None
        
        try:
            with open(task_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return Task.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading task {task_id}: {e}")
            return None
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task from disk."""
        task_path = self.tasks_dir / f"{task_id}.yaml"
        
        if not task_path.exists():
            return False
        
        try:
            task_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
            return False
    
    def list_tasks(self) -> list[tuple[str, Task]]:
        """List all saved tasks sorted by title."""
        tasks = []
        
        for task_file in self.tasks_dir.glob("*.yaml"):
            task_id = task_file.stem
            task = self.load_task(task_id)
            if task:
                tasks.append((task_id, task))
        
        # Sort by title
        tasks.sort(key=lambda x: x[1].title)
        
        return tasks
    
    def find_task_by_prefix(self, prefix: str) -> Optional[tuple[str, Task]]:
        """Find a task by ID prefix. Returns None if not found or ambiguous."""
        matches = []
        for task_file in self.tasks_dir.glob(f"{prefix}*.yaml"):
            task = self.load_task(task_file.stem)
            if task:
                matches.append((task_file.stem, task))
        
        if not matches:
            return None
        
        if len(matches) == 1:
            return matches[0]
        
        # Ambiguous prefix - log all matches
        logger.warning(f"Ambiguous task ID prefix '{prefix}'. Matches:")
        for task_id, task in matches:
            logger.warning(f"  {task_id}: {task.title}")
        return None
