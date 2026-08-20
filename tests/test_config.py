from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_agent.settings.config import (
    LangSmithSettings,
    MentionSettings,
    ModelSettings,
    RAGSettings,
    RuntimeSettings,
    Settings,
    SubAgentMCPServer,
    SubAgentSettings,
    ensure_agents_file,
    ensure_memory_file,
    find_agents_file,
    load_fs_policy_sandboxed,
    load_fs_policy_traversal,
    load_instructions,
    load_settings,
    reset_config,
    save_settings,
)
from ollama_agent.settings.paths import AGENTS_MD_NAME


class TestConfigManagement(unittest.TestCase):
    """Unit tests for configuration classes and serialization routines."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_file = Path(self.temp_dir.name) / "settings.yaml"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_settings_instantiation(self) -> None:
        s = Settings()
        self.assertEqual(s.model.name, "gemma4:26b")
        self.assertEqual(s.model.context_window, 10000)
        self.assertEqual(s.runtime.builtin_tool_timeout, 30)
        self.assertEqual(s.mentions.max_files, 100)
        self.assertEqual(s.rag.default_top_k, 5)

    def test_settings_serialization_cycle(self) -> None:
        original = Settings(
            model=ModelSettings(name="llama3.3:70b", reasoning_effort="high"),
            runtime=RuntimeSettings(allow_traversal=True, builtin_tool_timeout=60),
            subagents=[
                SubAgentSettings(
                    name="coder",
                    description="Expert coder",
                    mcp_servers=[SubAgentMCPServer(name="git", command="npx", args=["-y", "mcp-server-git"])],
                )
            ],
        )

        save_settings(original, self.settings_file)
        self.assertTrue(self.settings_file.exists())

        loaded = load_settings(self.settings_file)
        self.assertEqual(loaded.model.name, "llama3.3:70b")
        self.assertEqual(loaded.model.reasoning_effort, "high")
        self.assertTrue(loaded.runtime.allow_traversal)
        self.assertEqual(loaded.runtime.builtin_tool_timeout, 60)
        self.assertEqual(len(loaded.subagents), 1)
        self.assertEqual(loaded.subagents[0].name, "coder")
        self.assertEqual(len(loaded.subagents[0].mcp_servers), 1)
        self.assertEqual(loaded.subagents[0].mcp_servers[0].name, "git")

    def test_setup_environment_injects_langsmith_variables(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            s = Settings(
                langsmith=LangSmithSettings(
                    api_key="test_key_123",
                    tracing="true",
                    project="test_proj",
                    endpoint="https://api.smith.langchain.com",
                )
            )
            s.setup_environment()

            self.assertEqual(os.environ.get("LANGSMITH_KEY", os.environ.get("LANGSMITH_API_KEY")), "test_key_123")
            self.assertEqual(os.environ.get("LANGSMITH_TRACING"), "true")
            self.assertEqual(os.environ.get("LANGSMITH_PROJECT"), "test_proj")
            self.assertEqual(os.environ.get("LANGSMITH_ENDPOINT"), "https://api.smith.langchain.com")

    def test_ensure_memory_file_creates_file_with_scaffold(self) -> None:
        memory_file = Path(self.temp_dir.name) / "MEMORY.md"
        self.assertFalse(memory_file.exists())

        result = ensure_memory_file(memory_file)
        self.assertEqual(result, memory_file)
        self.assertTrue(memory_file.exists())
        self.assertIn("Long-Term Memory", memory_file.read_text(encoding="utf-8"))

    def test_ensure_agents_file_creates_file_with_scaffold(self) -> None:
        agents_file = Path(self.temp_dir.name) / AGENTS_MD_NAME
        self.assertFalse(agents_file.exists())

        result = ensure_agents_file(agents_file)
        self.assertEqual(result, agents_file)
        self.assertTrue(agents_file.exists())
        self.assertIn("Agent Guidelines", agents_file.read_text(encoding="utf-8"))

    def test_find_agents_file_resolution(self) -> None:
        proj_dir = Path(self.temp_dir.name) / "my_project"
        proj_dir.mkdir()
        agents_file = proj_dir / "AGENTS.md"
        agents_file.write_text("# Project Guidelines\n", encoding="utf-8")

        found = find_agents_file(proj_dir)
        self.assertEqual(found, agents_file)

    def test_load_instructions_creates_default_if_missing(self) -> None:
        instructions_file = Path(self.temp_dir.name) / "instructions.md"
        content = load_instructions(instructions_file)
        self.assertTrue(instructions_file.exists())
        self.assertTrue(len(content) > 0)

    def test_load_fs_policies(self) -> None:
        traversal_file = Path(self.temp_dir.name) / "fs_traversal.md"
        sandboxed_file = Path(self.temp_dir.name) / "fs_sandboxed.md"

        traversal_content = load_fs_policy_traversal(traversal_file)
        self.assertTrue(traversal_file.exists())
        self.assertTrue(len(traversal_content) > 0)

        sandboxed_content = load_fs_policy_sandboxed(sandboxed_file)
        self.assertTrue(sandboxed_file.exists())
        self.assertTrue(len(sandboxed_content) > 0)

    def test_reset_config_options(self) -> None:
        with self.assertRaises(ValueError):
            reset_config("invalid_option")

        msgs = reset_config("config-file")
        self.assertIsInstance(msgs, list)
        self.assertTrue(len(msgs) > 0)

        msgs_all = reset_config("all")
        self.assertTrue(len(msgs_all) >= 2)


if __name__ == "__main__":
    unittest.main()

