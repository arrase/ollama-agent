from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_agent.settings.config import (
    LangSmithSettings,
    ModelSettings,
    RuntimeSettings,
    Settings,
    SubAgentMCPServer,
    SubAgentSettings,
    ensure_memory_file,
    ensure_prompt_files,
    find_agents_file,
    load_fs_policy_sandboxed,
    load_fs_policy_traversal,
    load_instructions,
    load_rag_policy,
    load_settings,
    reset_config,
    save_settings,
)


class TestConfigManagement(unittest.TestCase):
    """Unit tests for configuration classes and serialization routines."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_file = Path(self.temp_dir.name) / "settings.yaml"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_settings_instantiation(self) -> None:
        s = Settings()
        self.assertEqual(s.model.name, "")
        self.assertEqual(s.model.context_window, 10000)
        self.assertIsNone(s.model.temperature)
        self.assertIsNone(s.model.top_p)
        self.assertIsNone(s.model.top_k)
        self.assertIsNone(s.model.min_p)
        self.assertIsNone(s.model.presence_penalty)
        self.assertIsNone(s.model.repeat_penalty)
        self.assertEqual(s.runtime.builtin_tool_timeout, 30)
        self.assertEqual(s.mentions.max_files, 100)
        self.assertEqual(s.rag.default_top_k, 5)

        # Default model dict should not include unset sampling parameters
        model_dict = s.to_dict()["model"]
        self.assertNotIn("temperature", model_dict)
        self.assertNotIn("top_p", model_dict)
        self.assertNotIn("top_k", model_dict)
        self.assertNotIn("min_p", model_dict)
        self.assertNotIn("presence_penalty", model_dict)
        self.assertNotIn("repeat_penalty", model_dict)

    def test_settings_serialization_cycle(self) -> None:
        original = Settings(
            model=ModelSettings(
                name="llama3.3:70b",
                reasoning_effort="high",
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                min_p=0.05,
                presence_penalty=0.5,
                repeat_penalty=1.2,
            ),
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
        self.assertEqual(loaded.model.temperature, 0.7)
        self.assertEqual(loaded.model.top_p, 0.95)
        self.assertEqual(loaded.model.top_k, 50)
        self.assertEqual(loaded.model.min_p, 0.05)
        self.assertEqual(loaded.model.presence_penalty, 0.5)
        self.assertEqual(loaded.model.repeat_penalty, 1.2)
        self.assertTrue(loaded.runtime.allow_traversal)
        self.assertEqual(loaded.runtime.builtin_tool_timeout, 60)
        self.assertEqual(len(loaded.subagents), 1)
        self.assertEqual(loaded.subagents[0].name, "coder")
        self.assertEqual(len(loaded.subagents[0].mcp_servers), 1)
        self.assertEqual(loaded.subagents[0].mcp_servers[0].name, "git")

    def test_model_repetition_penalty_alias(self) -> None:
        raw = {
            "model": {
                "name": "llama3.2:3b",
                "repetition_penalty": 1.3,
            }
        }
        s = Settings.from_dict(raw)
        self.assertEqual(s.model.repeat_penalty, 1.3)

    def test_model_and_subagent_context_window_max(self) -> None:
        raw = {
            "model": {
                "name": "qwen2.5-coder:32b",
                "context_window": "max",
            },
            "subagents": [
                {
                    "name": "researcher",
                    "description": "Deep researcher",
                    "context_window": "max",
                }
            ],
        }
        s = Settings.from_dict(raw)
        self.assertEqual(s.model.context_window, "max")
        self.assertEqual(s.subagents[0].context_window, "max")

        save_settings(s, self.settings_file)
        loaded = load_settings(self.settings_file)
        self.assertEqual(loaded.model.context_window, "max")
        self.assertEqual(loaded.subagents[0].context_window, "max")

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

    def test_load_rag_policy(self) -> None:
        rag_policy_file = Path(self.temp_dir.name) / "rag_policy.md"
        rag_content = load_rag_policy(rag_policy_file)
        self.assertTrue(rag_policy_file.exists())
        self.assertTrue(len(rag_content) > 0)
        self.assertIn("rag_search", rag_content)

        # Preserves custom content
        custom_rag = "# Custom RAG\nUse search."
        rag_policy_file.write_text(custom_rag, encoding="utf-8")
        loaded = load_rag_policy(rag_policy_file)
        self.assertEqual(loaded, custom_rag)

    def test_ensure_prompt_files_creates_all_templates(self) -> None:
        prompts_dir = Path(self.temp_dir.name) / "prompts_test"
        inst = prompts_dir / "instructions.md"
        trav = prompts_dir / "fs_traversal.md"
        sand = prompts_dir / "fs_sandboxed.md"
        rag = prompts_dir / "rag_policy.md"

        ensure_prompt_files(
            instructions_path=inst,
            traversal_path=trav,
            sandboxed_path=sand,
            rag_policy_path=rag,
        )

        self.assertTrue(inst.exists())
        self.assertTrue(trav.exists())
        self.assertTrue(sand.exists())
        self.assertTrue(rag.exists())
        self.assertIn("{RAG_POLICY}", inst.read_text(encoding="utf-8"))
        self.assertIn("rag_search", rag.read_text(encoding="utf-8"))

    def test_load_settings_does_not_overwrite_existing_file(self) -> None:
        custom_yaml = "model:\n  name: my-custom-model:latest\n  temperature: 0.8\n"
        self.settings_file.write_text(custom_yaml, encoding="utf-8")

        loaded = load_settings(self.settings_file)
        self.assertEqual(loaded.model.name, "my-custom-model:latest")
        self.assertEqual(loaded.model.temperature, 0.8)
        self.assertEqual(self.settings_file.read_text(encoding="utf-8"), custom_yaml)

    def test_load_instructions_preserves_custom_content(self) -> None:
        instructions_file = Path(self.temp_dir.name) / "instructions.md"
        custom_prompt = "# Custom Agent Instructions\nAlways be concise."
        instructions_file.write_text(custom_prompt, encoding="utf-8")

        loaded = load_instructions(instructions_file)
        self.assertEqual(loaded, custom_prompt)
        self.assertEqual(instructions_file.read_text(encoding="utf-8"), custom_prompt)

    def test_reset_config_options(self) -> None:
        with self.assertRaises(ValueError):
            reset_config("invalid_option")

        inst_path = Path(self.temp_dir.name) / "instructions.md"
        trav_path = Path(self.temp_dir.name) / "fs_traversal.md"
        sand_path = Path(self.temp_dir.name) / "fs_sandboxed.md"
        rag_path = Path(self.temp_dir.name) / "rag_policy.md"

        msgs = reset_config(
            "config-file",
            settings_path=self.settings_file,
            instructions_path=inst_path,
            traversal_path=trav_path,
            sandboxed_path=sand_path,
            rag_policy_path=rag_path,
        )
        self.assertIsInstance(msgs, list)
        self.assertTrue(len(msgs) > 0)
        self.assertTrue(self.settings_file.exists())

        msgs_all = reset_config(
            "all",
            settings_path=self.settings_file,
            instructions_path=inst_path,
            traversal_path=trav_path,
            sandboxed_path=sand_path,
            rag_policy_path=rag_path,
        )
        self.assertTrue(len(msgs_all) >= 2)
        self.assertTrue(inst_path.exists())
        self.assertTrue(trav_path.exists())
        self.assertTrue(sand_path.exists())
        self.assertTrue(rag_path.exists())


if __name__ == "__main__":
    unittest.main()

