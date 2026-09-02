from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jinja2

from ollama_agent.settings.config import (
    LangSmithSettings,
    ModelSettings,
    RuntimeSettings,
    Settings,
    SubAgentMCPServer,
    SubAgentSettings,
    _dataclass_from_dict,
    _subagents_from_list,
    ensure_memory_file,
    ensure_prompt_files,
    find_agents_file,
    load_instructions,
    load_settings,
    render_prompt_template,
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

    def test_unknown_setting_keys_raise_value_error(self) -> None:
        raw = {
            "model": {
                "name": "llama3.2:3b",
                "repetition_penalty": 1.3,
            }
        }
        with self.assertRaises(ValueError) as ctx:
            Settings.from_dict(raw)
        self.assertIn("repetition_penalty", str(ctx.exception))

    def test_empty_rag_dir_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Settings.from_dict({"rag": {"rag_dir": ""}})

    def test_load_settings_non_mapping_raises_value_error(self) -> None:
        self.settings_file.write_text("just-a-scalar\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_settings(self.settings_file)
        self.assertIn("YAML mapping", str(ctx.exception))
        self.assertIn(str(self.settings_file), str(ctx.exception))

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
        self.assertIn("CORE OBJECTIVE", content)
        self.assertIn("MEMORY GUIDELINES", content)
        self.assertIn("runtime.allow_traversal", content)
        self.assertIn("RAG POLICY", content)

    def test_ensure_prompt_files_creates_instructions(self) -> None:
        prompts_dir = Path(self.temp_dir.name) / "prompts_test"
        inst = prompts_dir / "instructions.md"

        ensure_prompt_files(instructions_path=inst)

        self.assertTrue(inst.exists())
        content = inst.read_text(encoding="utf-8")
        self.assertIn("CORE OBJECTIVE", content)
        self.assertIn("runtime.allow_traversal", content)
        self.assertIn("RAG POLICY", content)

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

    def test_load_instructions_returns_empty_file_content_as_is(self) -> None:
        instructions_file = Path(self.temp_dir.name) / "instructions.md"
        instructions_file.write_text("", encoding="utf-8")

        self.assertEqual(load_instructions(instructions_file), "")

    def test_render_prompt_template(self) -> None:
        # Basic substitution
        simple_tpl = "Hello, {{ name }}!"
        self.assertEqual(render_prompt_template(simple_tpl, {"name": "Assistant"}), "Hello, Assistant!")

        # Conditionals: traversal enabled vs disabled
        template_str = "{% if runtime.allow_traversal %}\nTRAVERSAL MODE\n{% else %}\nSANDBOXED MODE\n{% endif %}"
        rendered_traversal = render_prompt_template(template_str, {"runtime": {"allow_traversal": True}})
        self.assertEqual(rendered_traversal.strip(), "TRAVERSAL MODE")

        rendered_sandboxed = render_prompt_template(template_str, {"runtime": {"allow_traversal": False}})
        self.assertEqual(rendered_sandboxed.strip(), "SANDBOXED MODE")

        # Strict undefined check raises error when variable is missing
        with self.assertRaises(jinja2.UndefinedError):
            render_prompt_template("{{ missing_var }}", {})

        with self.assertRaises(jinja2.UndefinedError):
            render_prompt_template("{% if runtime.allow_traversal %}ok{% endif %}", {})

        # Render full default instructions template
        default_inst = load_instructions(Path(self.temp_dir.name) / "default_inst.md")
        rendered_full_traversal = render_prompt_template(
            default_inst,
            {"runtime": {"allow_traversal": True}, "rag_active": False},
        )
        self.assertIn("You have full access to the host filesystem", rendered_full_traversal)
        self.assertNotIn("operate on a virtual root", rendered_full_traversal)
        self.assertNotIn("# RAG POLICY", rendered_full_traversal)

        rendered_full_sandboxed = render_prompt_template(
            default_inst,
            {"runtime": {"allow_traversal": False}, "rag_active": True, "rag_database": "test_kb"},
        )
        self.assertIn("operate on a virtual root", rendered_full_sandboxed)
        self.assertNotIn("You have full access to the host filesystem", rendered_full_sandboxed)
        self.assertIn("# RAG POLICY", rendered_full_sandboxed)
        self.assertIn("('test_kb')", rendered_full_sandboxed)

    def test_reset_config_options(self) -> None:
        with self.assertRaises(ValueError):
            reset_config("invalid_option")

        inst_path = Path(self.temp_dir.name) / "instructions.md"

        msgs = reset_config(
            "config-file",
            settings_path=self.settings_file,
            instructions_path=inst_path,
        )
        self.assertIsInstance(msgs, list)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(self.settings_file.exists())

        # Modify instructions to test reset of system-prompt
        inst_path.write_text("Custom prompt", encoding="utf-8")
        msgs_prompt = reset_config(
            "system-prompt",
            settings_path=self.settings_file,
            instructions_path=inst_path,
        )
        self.assertIn("Reset: Restored default system prompt", msgs_prompt[0])
        self.assertIn("CORE OBJECTIVE", inst_path.read_text(encoding="utf-8"))

        msgs_all = reset_config(
            "all",
            settings_path=self.settings_file,
            instructions_path=inst_path,
        )
        self.assertEqual(len(msgs_all), 2)
        self.assertTrue(self.settings_file.exists())
        self.assertTrue(inst_path.exists())

    def test_settings_from_dict_rejects_unknown_root_keys(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Settings.from_dict({"modeel": {"name": "llama3"}})
        self.assertIn("modeel", str(ctx.exception))

    def test_find_agents_file_default_start_dir(self) -> None:
        orig_cwd = Path.cwd()
        proj_dir = Path(self.temp_dir.name) / "default_dir"
        proj_dir.mkdir()
        agents_file = proj_dir / "AGENTS.md"
        agents_file.write_text("# Default\n", encoding="utf-8")
        try:
            os.chdir(proj_dir)
            found = find_agents_file()
            self.assertEqual(found, agents_file)
        finally:
            os.chdir(orig_cwd)

    def test_dataclass_from_dict_none_returns_default(self) -> None:
        model = _dataclass_from_dict(ModelSettings, None)
        self.assertIsInstance(model, ModelSettings)
        self.assertEqual(model.base_url, "http://localhost:11434")

    def test_dataclass_from_dict_rejects_non_dict(self) -> None:
        with self.assertRaises(ValueError):
            _dataclass_from_dict(ModelSettings, "not-a-dict")
        with self.assertRaises(ValueError):
            _dataclass_from_dict(ModelSettings, 123)

    def test_subagents_from_list_parsing(self) -> None:
        self.assertEqual(_subagents_from_list(None), [])
        raw = [
            {
                "name": "explorer",
                "description": "Explores",
            },
            {
                "name": "tester",
                "mcp_servers": [{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}],
            },
        ]
        subagents = _subagents_from_list(raw)
        self.assertEqual(len(subagents), 2)
        self.assertEqual(subagents[0].name, "explorer")
        self.assertEqual(subagents[0].mcp_servers, [])
        self.assertEqual(subagents[1].name, "tester")
        self.assertEqual(len(subagents[1].mcp_servers), 1)
        self.assertEqual(subagents[1].mcp_servers[0].name, "fetch")

    def test_subagents_from_list_validation(self) -> None:
        with self.assertRaises(ValueError):
            _subagents_from_list("not-a-list")
        with self.assertRaises(ValueError):
            _subagents_from_list(["not-a-dict"])
        with self.assertRaises(ValueError):
            _subagents_from_list([{"name": "coder", "mcp_servers": "invalid"}])

    def test_load_settings_falsy_scalar_yaml_raises_value_error(self) -> None:
        for val in ["false\n", "0\n", "[]\n"]:
            self.settings_file.write_text(val, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(self.settings_file)


if __name__ == "__main__":
    unittest.main()
