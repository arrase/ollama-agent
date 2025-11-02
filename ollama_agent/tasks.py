"""Task management for saving and executing agent prompts."""

import hashlib
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .agent import DEFAULT_REASONING_EFFORT, ReasoningEffortValue, validate_reasoning_effort


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
    """
    Compute a short task ID from the title using Blake2s.
    
    Args:
        title: The task title.
        
    Returns:
        Short hash string (8 characters).
    """
    hash_digest = hashlib.blake2s(title.encode('utf-8'), digest_size=16).hexdigest()
    return hash_digest[:HASH_LENGTH]


class TaskManager:
    """Manages task storage and retrieval."""
    
    def __init__(self, tasks_dir: Optional[Path] = None):
        """
        Initialize the task manager.
        
        Args:
            tasks_dir: Directory to store task files. Defaults to ~/.ollama-agent/tasks
        """
        self.tasks_dir = tasks_dir or (Path.home() / ".ollama-agent" / "tasks")
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_task_path(self, task_id: str) -> Path:
        """Get the file path for a task ID."""
        return self.tasks_dir / f"{task_id}.yaml"
    
    def save_task(self, task: Task) -> str:
        """
        Save a task to disk.
        
        Args:
            task: The task to save.
            
        Returns:
            The task ID (hash).
        """
        task_id = compute_task_id(task.title)
        task_path = self._get_task_path(task_id)
        
        with open(task_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(task.to_dict(), f, default_flow_style=False, allow_unicode=True)
        
        return task_id
    
    def load_task(self, task_id: str) -> Optional[Task]:
        """
        Load a task from disk.
        
        Args:
            task_id: The task ID to load.
            
        Returns:
            The loaded task or None if not found.
        """
        task_path = self._get_task_path(task_id)
        
        if not task_path.exists():
            return None
        
        try:
            with open(task_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return Task.from_dict(data)
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
            return None
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from disk.
        
        Args:
            task_id: The task ID to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        task_path = self._get_task_path(task_id)
        
        if not task_path.exists():
            return False
        
        try:
            task_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting task {task_id}: {e}")
            return False
    
    def list_tasks(self) -> list[tuple[str, Task]]:
        """
        List all saved tasks.
        
        Returns:
            List of tuples (task_id, task).
        """
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
        """
        Find a task by ID prefix.
        
        Args:
            prefix: The task ID prefix to search for.
            
        Returns:
            Tuple of (task_id, task) or None if not found or ambiguous.
        """
        matches = [
            (task_file.stem, task)
            for task_file in self.tasks_dir.glob(f"{prefix}*.yaml")
            if (task := self.load_task(task_file.stem)) is not None
        ]
        
        if not matches:
            return None
        
        if len(matches) == 1:
            return matches[0]
        
        # Ambiguous prefix - show all matches
        print(f"Ambiguous task ID prefix '{prefix}'. Matches:")
        for task_id, task in matches:
            print(f"  {task_id}: {task.title}")
        return None
