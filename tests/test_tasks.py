from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jinja2.exceptions import UndefinedError

from ollama_agent.tasks.manager import Task, TaskInput, TaskManager


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

    def test_task_from_dict_non_dict_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Task.from_dict("not-a-dict")
        with self.assertRaises(ValueError):
            Task.from_dict([1, 2, 3])

    def test_find_matches_no_shadowing_and_sorted_deterministically(self) -> None:
        self.mgr.save("build-docs", Task(title="Docs", prompt="p", model="m"))
        self.mgr.save("build", Task(title="Build Base", prompt="p", model="m"))
        self.mgr.save("build-docker", Task(title="Docker", prompt="p", model="m"))

        matches = self.mgr.find_matches("build")
        self.assertEqual(len(matches), 3)
        self.assertEqual([m[0] for m in matches], ["build", "build-docker", "build-docs"])

    def test_get_directory_raises_file_not_found(self) -> None:
        dir_as_task = self.tasks_dir / "folder.yaml"
        dir_as_task.mkdir()
        with self.assertRaises(FileNotFoundError):
            self.mgr.get("folder")

    def test_task_input_defaults(self) -> None:
        inp = TaskInput()
        self.assertEqual(inp.description, "")
        self.assertIsNone(inp.default)
        self.assertFalse(inp.required)
        self.assertEqual(inp.type, "string")

    def test_task_from_dict_with_inputs(self) -> None:
        data = {
            "title": "Refactor Code",
            "prompt": "Refactor {{ file }} using {{ pattern }}",
            "model": "qwen2.5-coder:7b",
            "reasoning_effort": "medium",
            "inputs": {
                "file": {
                    "description": "Path to file",
                    "required": True,
                    "type": "string",
                },
                "pattern": {
                    "description": "Design pattern",
                    "default": "factory",
                    "required": False,
                    "type": "string",
                },
                "dry_run": {
                    "description": "Dry run mode",
                    "default": False,
                    "type": "boolean",
                },
            },
        }
        task = Task.from_dict(data)
        self.assertEqual(len(task.inputs), 3)
        self.assertEqual(task.inputs["file"].description, "Path to file")
        self.assertTrue(task.inputs["file"].required)
        self.assertEqual(task.inputs["file"].type, "string")
        self.assertEqual(task.inputs["pattern"].default, "factory")
        self.assertFalse(task.inputs["pattern"].required)
        self.assertFalse(task.inputs["dry_run"].default)
        self.assertEqual(task.inputs["dry_run"].type, "boolean")

    def test_task_save_and_get_with_inputs(self) -> None:
        task = Task(
            title="Deploy App",
            prompt="Deploy {{ app_name }} to {{ env }}",
            model="llama3:8b",
            inputs={
                "app_name": TaskInput(description="App name", required=True),
                "env": TaskInput(description="Environment", default="staging"),
            },
        )
        self.mgr.save("deploy-app", task)
        loaded = self.mgr.get("deploy-app")
        self.assertEqual(len(loaded.inputs), 2)
        self.assertTrue(loaded.inputs["app_name"].required)
        self.assertEqual(loaded.inputs["env"].default, "staging")

    def test_render_backward_compatibility_no_inputs(self) -> None:
        task_static = Task(title="Static", prompt="Run standard linter", model="m")
        self.assertEqual(task_static.render(), "Run standard linter")
        self.assertEqual(task_static.render({"unused": "ignored"}), "Run standard linter")

        task_dynamic = Task(title="Dynamic", prompt="Hello {{ name }}!", model="m")
        self.assertEqual(task_dynamic.render({"name": "Alice"}), "Hello Alice!")
        with self.assertRaises(UndefinedError):
            task_dynamic.render()

    def test_render_with_variables_and_defaults(self) -> None:
        task = Task(
            title="Greet",
            prompt="{{ greeting }}, {{ name }}!",
            model="m",
            inputs={
                "greeting": TaskInput(default="Hello"),
                "name": TaskInput(default="World"),
            },
        )
        self.assertEqual(task.render(), "Hello, World!")
        self.assertEqual(task.render({"name": "Alice"}), "Hello, Alice!")
        self.assertEqual(task.render({"greeting": "Hi", "name": "Bob"}), "Hi, Bob!")

    def test_render_missing_required_variable_raises(self) -> None:
        task = Task(
            title="Search",
            prompt="Search for {{ query }}",
            model="m",
            inputs={
                "query": TaskInput(required=True),
            },
        )
        with self.assertRaises(ValueError) as ctx:
            task.render()
        self.assertIn("query", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            task.render({})
        self.assertIn("query", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            task.render({"query": None})
        self.assertIn("query", str(ctx.exception))

        self.assertEqual(task.render({"query": "python 3.14"}), "Search for python 3.14")

    def test_render_jinja_conditionals(self) -> None:
        task = Task(
            title="Build",
            prompt="{% if optimize %}Optimize mode{% else %}Debug mode{% endif %}",
            model="m",
            inputs={
                "optimize": TaskInput(type="boolean", default=False),
            },
        )
        self.assertEqual(task.render(), "Debug mode")
        self.assertEqual(task.render({"optimize": True}), "Optimize mode")
        self.assertEqual(task.render({"optimize": "yes"}), "Optimize mode")
        self.assertEqual(task.render({"optimize": "no"}), "Debug mode")

    def test_render_jinja_loops(self) -> None:
        task = Task(
            title="Batch",
            prompt="Files:\n{% for f in files %}- {{ f }}\n{% endfor %}",
            model="m",
        )
        rendered = task.render({"files": ["a.py", "b.py", "c.py"]})
        self.assertEqual(rendered, "Files:\n- a.py\n- b.py\n- c.py\n")

    def test_render_jinja_default_filter(self) -> None:
        task = Task(
            title="Filter",
            prompt="Welcome {{ user | default('Guest') }}",
            model="m",
        )
        self.assertEqual(task.render(), "Welcome Guest")
        self.assertEqual(task.render({"user": "Bob"}), "Welcome Bob")

    def test_render_type_coercion_boolean(self) -> None:
        task = Task(
            title="BoolTest",
            prompt="{% if flag %}TRUE{% else %}FALSE{% endif %}",
            model="m",
            inputs={"flag": TaskInput(type="boolean")},
        )
        for truthy in [True, "true", "True", "TRUE", "yes", "YES", "1", 1, 1.0]:
            self.assertEqual(task.render({"flag": truthy}), "TRUE")

        for falsy in [False, "false", "False", "FALSE", "no", "NO", "0", 0, 0.0]:
            self.assertEqual(task.render({"flag": falsy}), "FALSE")

        for invalid in ["maybe", "invalid", 2, -1, [1], {"a": 1}]:
            with self.assertRaises(ValueError):
                task.render({"flag": invalid})

    def test_render_type_coercion_number(self) -> None:
        task = Task(
            title="NumTest",
            prompt="Result: {{ value * 2 }}",
            model="m",
            inputs={"value": TaskInput(type="number")},
        )
        self.assertEqual(task.render({"value": 5}), "Result: 10")
        self.assertEqual(task.render({"value": "5"}), "Result: 10")
        self.assertEqual(task.render({"value": 2.5}), "Result: 5.0")
        self.assertEqual(task.render({"value": "2.5"}), "Result: 5.0")

        for invalid in ["abc", True, False, [10], {"num": 1}]:
            with self.assertRaises(ValueError):
                task.render({"value": invalid})

    def test_render_type_coercion_string(self) -> None:
        task = Task(
            title="StrTest",
            prompt="Value: {{ text }}",
            model="m",
            inputs={"text": TaskInput(type="string")},
        )
        self.assertEqual(task.render({"text": "hello"}), "Value: hello")
        self.assertEqual(task.render({"text": 123}), "Value: 123")
        self.assertEqual(task.render({"text": True}), "Value: True")

    def test_render_falsy_defaults(self) -> None:
        task = Task(
            title="FalsyDefaults",
            prompt="count={{ count }}, active={{ active }}, prefix='{{ prefix }}'",
            model="m",
            inputs={
                "count": TaskInput(type="number", default=0),
                "active": TaskInput(type="boolean", default=False),
                "prefix": TaskInput(type="string", default=""),
            },
        )
        self.assertEqual(task.render(), "count=0, active=False, prefix=''")
        self.assertEqual(task.render({"count": 5}), "count=5, active=False, prefix=''")
        self.assertEqual(task.render({"active": True}), "count=0, active=True, prefix=''")

    def test_render_undefined_variable_in_block_raises(self) -> None:
        task = Task(
            title="BlockTest",
            prompt="{% if missing_cond %}YES{% endif %}",
            model="m",
        )
        with self.assertRaises(UndefinedError):
            task.render()

    def test_task_from_dict_with_task_input_instances(self) -> None:
        raw = {
            "title": "Direct TaskInput",
            "prompt": "{{ text }}",
            "model": "m",
            "reasoning_effort": "low",
            "inputs": {
                "text": TaskInput(description="Sample", default="demo", type="string"),
            },
        }
        task = Task.from_dict(raw)
        self.assertIsInstance(task.inputs["text"], TaskInput)
        self.assertEqual(task.inputs["text"].description, "Sample")
        self.assertEqual(task.render(), "demo")

    def test_render_complex_template_with_mixed_inputs(self) -> None:
        prompt = (
            "Model: {{ model_name }}\n"
            "Iterations: {{ count }}\n"
            "{% if verbose %}\n"
            "Verbose mode enabled.\n"
            "{% endif %}\n"
            "Tags:\n"
            "{% for tag in tags %}\n"
            "- {{ tag }}\n"
            "{% endfor %}"
        )
        task = Task(
            title="ComplexTask",
            prompt=prompt,
            model="m",
            inputs={
                "model_name": TaskInput(required=True, type="string"),
                "count": TaskInput(type="number", default=1),
                "verbose": TaskInput(type="boolean", default=False),
            },
        )
        rendered = task.render({
            "model_name": "qwen2.5:32b",
            "count": "3",
            "verbose": "yes",
            "tags": ["prod", "gpu"],
        })
        expected = (
            "Model: qwen2.5:32b\n"
            "Iterations: 3\n"
            "Verbose mode enabled.\n"
            "Tags:\n"
            "- prod\n"
            "- gpu\n"
        )
        self.assertEqual(rendered, expected)


if __name__ == "__main__":
    unittest.main()
