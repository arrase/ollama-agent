from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ollama_agent.agent.agent import AgentRuntime
from ollama_agent.agent.security import is_safe_command
from ollama_agent.settings.config import Settings


class TestSecurity(unittest.TestCase):
    """Unit tests for command safety whitelist and runtime approval policies."""

    def test_safe_commands(self) -> None:
        safe_samples = [
            "git status",
            "git diff HEAD~1",
            "git log -n 5",
            "git show HEAD",
            "git branch -a",
            "ls -la",
            "pwd",
            "cat pyproject.toml",
            "head -n 20 main.py",
            "tail -f log.txt",
            "grep 'def test' -rn tests/",
            "rg 'AgentRuntime'",
            "find . -name '*.py'",
            "pytest -v",
            "python -m unittest discover -s tests",
            ".venv/bin/python -m unittest",
            "ruff check .",
            "mypy ollama_agent",
            "black --check .",
        ]
        for cmd in safe_samples:
            with self.subTest(cmd=cmd):
                self.assertTrue(is_safe_command(cmd), f"Expected '{cmd}' to be recognized as safe")

    def test_unsafe_commands(self) -> None:
        unsafe_samples = [
            "",
            "rm -rf /tmp/data",
            "rm test.py",
            "rmdir old_dir",
            "mv file1.txt file2.txt",
            "chmod +x script.sh",
            "chown user:user file",
            "kill 1234",
            "pkill python",
            "sudo apt update",
            "git push origin main",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "git checkout main",
            "git commit -m 'feat'",
            "git merge feature",
            "cat file.txt > output.txt",
            "echo 'data' >> output.txt",
            "pip install requests",
            "pip uninstall requests",
            "npm install",
            "yarn add package",
            "cargo install cargo-watch",
        ]
        for cmd in unsafe_samples:
            with self.subTest(cmd=cmd):
                self.assertFalse(is_safe_command(cmd), f"Expected '{cmd}' to be recognized as dangerous")

    def test_runtime_should_interrupt_with_safe_commands(self) -> None:
        settings = Settings()
        settings.runtime.auto_approve_safe_commands = True
        runtime = AgentRuntime(settings=settings)
        runtime.yolo_mode = False

        # Create DeepAgent interrupt handler extracted from _build_graph context
        def should_interrupt_tool(request: MagicMock) -> bool:
            if runtime.yolo_mode:
                return False
            name = request.tool_call["name"]
            if name in runtime.auto_approved_tools:
                return False
            if name == "execute" and runtime.settings.runtime.auto_approve_safe_commands:
                args = request.tool_call.get("args") or {}
                command = args.get("command", "") if isinstance(args, dict) else ""
                if is_safe_command(command):
                    return False
            return True

        # 1. Safe command with auto_approve_safe_commands=True -> does NOT interrupt
        safe_req = MagicMock(tool_call={"name": "execute", "args": {"command": "git status"}})
        self.assertFalse(should_interrupt_tool(safe_req))

        # 2. Dangerous command -> DOES interrupt
        dangerous_req = MagicMock(tool_call={"name": "execute", "args": {"command": "rm -rf build"}})
        self.assertTrue(should_interrupt_tool(dangerous_req))

        # 3. File modifications -> ALWAYS interrupt unless YOLO / approved
        write_req = MagicMock(tool_call={"name": "write_file", "args": {"path": "test.txt"}})
        self.assertTrue(should_interrupt_tool(write_req))

        # 4. Safe command with auto_approve_safe_commands=False -> DOES interrupt
        runtime.settings.runtime.auto_approve_safe_commands = False
        self.assertTrue(should_interrupt_tool(safe_req))

        # 5. YOLO mode -> NEVER interrupts
        runtime.yolo_mode = True
        self.assertFalse(should_interrupt_tool(dangerous_req))
        self.assertFalse(should_interrupt_tool(write_req))


if __name__ == "__main__":
    unittest.main()
