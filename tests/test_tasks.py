from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_agent.tasks.manager import Task, TaskManager


class TestTaskManager(unittest.TestCase):
    """Unit tests for Task and TaskManager persistence."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tasks_dir = Path(self.temp_dir.name)
        self.mgr = TaskManager(self.tasks_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_task_model_initialization(self) -> None:
        t = Task(title="Review PR", prompt="Check diff", model="llama3:8b", reasoning_effort="high")
        self.assertEqual(t.title, "Review PR")
        self.assertEqual(t.model, "llama3:8b")
        self.assertEqual(t.reasoning_effort, "high")

    def test_save_and_get_task(self) -> None:
        t = Task(title="Run tests", prompt="pytest -v", model="gemma4:26b")
        saved_id = self.mgr.save("run-tests", t)
        self.assertEqual(saved_id, "run-tests")

        loaded = self.mgr.get("run-tests")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.title, "Run tests")
        self.assertEqual(loaded.prompt, "pytest -v")

    def test_get_missing_task_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.mgr.get("missing-task")

    def test_get_corrupted_task_raises(self) -> None:
        (self.tasks_dir / "broken.yaml").write_text(
            "title: Broken\nprompt: x\nmodel: m\n", encoding="utf-8"
        )
        with self.assertRaises(KeyError):
            self.mgr.get("broken")

    def test_find_matches_rejects_invalid_prefix(self) -> None:
        with self.assertRaises(ValueError):
            self.mgr.find_matches("../escape")
        with self.assertRaises(ValueError):
            self.mgr.get("../escape")
        with self.assertRaises(ValueError):
            self.mgr.delete("../escape")

    def test_save_existing_task_without_overwrite_raises(self) -> None:
        t = Task(title="T1", prompt="P1", model="M1")
        self.mgr.save("task-1", t)
        with self.assertRaises(FileExistsError):
            self.mgr.save("task-1", t, overwrite=False)

    def test_find_matches_exact_and_prefix(self) -> None:
        t1 = Task(title="Build docker", prompt="docker build", model="M1")
        t2 = Task(title="Build docs", prompt="mkdocs build", model="M1")
        self.mgr.save("build-docker", t1)
        self.mgr.save("build-docs", t2)

        matches = self.mgr.find_matches("build-")
        self.assertEqual(len(matches), 2)

        exact = self.mgr.find_matches("build-docker")
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0][0], "build-docker")

    def test_delete_task(self) -> None:
        t = Task(title="Temporary", prompt="Temp", model="M1")
        self.mgr.save("temp-task", t)
        self.mgr.delete("temp-task")
        self.assertFalse((self.tasks_dir / "temp-task.yaml").exists())
        with self.assertRaises(FileNotFoundError):
            self.mgr.get("temp-task")
        with self.assertRaises(FileNotFoundError):
            self.mgr.delete("temp-task")

    def test_list_all_sorted_by_title(self) -> None:
        self.mgr.save("t-b", Task(title="Beta Task", prompt="p", model="m"))
        self.mgr.save("t-a", Task(title="Alpha Task", prompt="p", model="m"))

        all_tasks = self.mgr.list_all()
        self.assertEqual(len(all_tasks), 2)
        self.assertEqual(all_tasks[0][1].title, "Alpha Task")
        self.assertEqual(all_tasks[1][1].title, "Beta Task")


if __name__ == "__main__":
    unittest.main()
