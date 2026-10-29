from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jinja2
import pytest

from ollama_agent.core import render_prompt_template
from ollama_agent.settings.config import (
    LangSmithSettings,
    ModelSettings,
    RuntimeSettings,
    Settings,
    SubAgentMCPServer,
    SubAgentSettings,
    _dataclass_from_dict,
    ensure_memory_file,
    load_instructions,
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
        assert s.model.name == ""
        assert s.model.context_window == 10000
        assert s.model.temperature is None
        assert s.model.top_p is None
        assert s.model.top_k is None
        assert s.model.min_p is None
        assert s.model.presence_penalty is None
        assert s.model.repeat_penalty is None
        assert s.runtime.builtin_tool_timeout == 30
        assert s.mentions.max_files == 100
        assert s.rag.default_top_k == 5
        assert s.rag.embedder_model == "embeddinggemma-2:740m"
        assert s.rag.embedding_dims == 768

        # Default model dict should not include unset sampling parameters
        model_dict = s.to_dict()["model"]
        assert "temperature" not in model_dict
        assert "top_p" not in model_dict
        assert "top_k" not in model_dict
        assert "min_p" not in model_dict
        assert "presence_penalty" not in model_dict
        assert "repeat_penalty" not in model_dict

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
        assert self.settings_file.exists()

        loaded = load_settings(self.settings_file)
        assert loaded.model.name == "llama3.3:70b"
        assert loaded.model.reasoning_effort == "high"
        assert loaded.model.temperature == 0.7
        assert loaded.model.top_p == 0.95
        assert loaded.model.top_k == 50
        assert loaded.model.min_p == 0.05
        assert loaded.model.presence_penalty == 0.5
        assert loaded.model.repeat_penalty == 1.2
        assert loaded.runtime.allow_traversal
        assert loaded.runtime.builtin_tool_timeout == 60
        assert len(loaded.subagents) == 1
        assert loaded.subagents[0].name == "coder"
        assert len(loaded.subagents[0].mcp_servers) == 1
        assert loaded.subagents[0].mcp_servers[0].name == "git"

    def test_unknown_setting_keys_raise_value_error(self) -> None:
        raw = {
            "model": {
                "name": "llama3.2:3b",
                "repetition_penalty": 1.3,
            }
        }
        with pytest.raises(ValueError) as ctx:
            Settings.from_dict(raw)
        assert "repetition_penalty" in str(ctx.value)

    def test_empty_rag_dir_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Settings.from_dict({"rag": {"rag_dir": ""}})

    def test_load_settings_non_mapping_raises_value_error(self) -> None:
        self.settings_file.write_text("just-a-scalar\n", encoding="utf-8")
        with pytest.raises(ValueError) as ctx:
            load_settings(self.settings_file)
        assert "YAML mapping" in str(ctx.value)
        assert str(self.settings_file) in str(ctx.value)

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
        assert s.model.context_window == "max"
        assert s.subagents[0].context_window == "max"

        save_settings(s, self.settings_file)
        loaded = load_settings(self.settings_file)
        assert loaded.model.context_window == "max"
        assert loaded.subagents[0].context_window == "max"

    def test_setup_environment_injects_langsmith_variables(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            s = Settings(
                langsmith=LangSmithSettings(
                    api_key="test_key_123",
                    tracing=True,
                    project="test_proj",
                    endpoint="https://api.smith.langchain.com",
                )
            )
            s.setup_environment()

            assert os.environ["LANGSMITH_API_KEY"] == "test_key_123"
            assert os.environ["LANGSMITH_TRACING"] == "true"
            assert os.environ["LANGSMITH_PROJECT"] == "test_proj"
            assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"

    def test_setup_environment_injects_langsmith_boolean_tracing(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            s = Settings(
                langsmith=LangSmithSettings(
                    tracing=True,
                )
            )
            s.setup_environment()
            assert os.environ["LANGSMITH_TRACING"] == "true"

    def test_ensure_memory_file_creates_file_with_scaffold(self) -> None:
        memory_file = Path(self.temp_dir.name) / "MEMORY.md"
        assert not memory_file.exists()

        result = ensure_memory_file(memory_file)
        assert result == memory_file
        assert memory_file.exists()
        assert "Long-Term Memory" in memory_file.read_text(encoding="utf-8")

    def test_load_instructions_creates_default_if_missing(self) -> None:
        instructions_file = Path(self.temp_dir.name) / "instructions.md"
        content = load_instructions(instructions_file)
        assert instructions_file.exists()
        assert len(content) > 0
        assert "CORE OBJECTIVE" in content
        assert "MEMORY GUIDELINES" in content
        assert "runtime.allow_traversal" in content
        assert "RAG POLICY" in content

    def test_load_settings_does_not_overwrite_existing_file(self) -> None:
        custom_yaml = "model:\n  name: my-custom-model:latest\n  temperature: 0.8\n"
        self.settings_file.write_text(custom_yaml, encoding="utf-8")

        loaded = load_settings(self.settings_file)
        assert loaded.model.name == "my-custom-model:latest"
        assert loaded.model.temperature == 0.8
        assert self.settings_file.read_text(encoding="utf-8") == custom_yaml

    def test_load_instructions_preserves_custom_content(self) -> None:
        instructions_file = Path(self.temp_dir.name) / "instructions.md"
        custom_prompt = "# Custom Agent Instructions\nAlways be concise."
        instructions_file.write_text(custom_prompt, encoding="utf-8")

        loaded = load_instructions(instructions_file)
        assert loaded == custom_prompt
        assert instructions_file.read_text(encoding="utf-8") == custom_prompt

    def test_load_instructions_returns_empty_file_content_as_is(self) -> None:
        instructions_file = Path(self.temp_dir.name) / "instructions.md"
        instructions_file.write_text("", encoding="utf-8")

        assert load_instructions(instructions_file) == ""

    def test_render_prompt_template(self) -> None:
        # Basic substitution
        simple_tpl = "Hello, {{ name }}!"
        assert render_prompt_template(simple_tpl, {"name": "Assistant"}) == "Hello, Assistant!"

        # Conditionals: traversal enabled vs disabled
        template_str = "{% if runtime.allow_traversal %}\nTRAVERSAL MODE\n{% else %}\nSANDBOXED MODE\n{% endif %}"
        rendered_traversal = render_prompt_template(template_str, {"runtime": {"allow_traversal": True}})
        assert rendered_traversal.strip() == "TRAVERSAL MODE"

        rendered_sandboxed = render_prompt_template(template_str, {"runtime": {"allow_traversal": False}})
        assert rendered_sandboxed.strip() == "SANDBOXED MODE"

        # Strict undefined check raises error when variable is missing
        with pytest.raises(jinja2.UndefinedError):
            render_prompt_template("{{ missing_var }}", {})

        with pytest.raises(jinja2.UndefinedError):
            render_prompt_template("{% if runtime.allow_traversal %}ok{% endif %}", {})

        # Render full default instructions template
        default_inst = load_instructions(Path(self.temp_dir.name) / "default_inst.md")
        rendered_full_traversal = render_prompt_template(
            default_inst,
            {"runtime": {"allow_traversal": True}, "rag_active": False},
        )
        assert "You have full access to the host filesystem" in rendered_full_traversal
        assert "operate on a virtual root" not in rendered_full_traversal
        assert "# RAG POLICY" not in rendered_full_traversal

        rendered_full_sandboxed = render_prompt_template(
            default_inst,
            {"runtime": {"allow_traversal": False}, "rag_active": True, "rag_database": "test_kb"},
        )
        assert "operate on a virtual root" in rendered_full_sandboxed
        assert "You have full access to the host filesystem" not in rendered_full_sandboxed
        assert "# RAG POLICY" in rendered_full_sandboxed
        assert "('test_kb')" in rendered_full_sandboxed

    def test_reset_config_options(self) -> None:
        with pytest.raises(ValueError):
            reset_config("invalid_option")

        inst_path = Path(self.temp_dir.name) / "instructions.md"

        msgs = reset_config(
            "config-file",
            settings_path=self.settings_file,
            instructions_path=inst_path,
        )
        assert isinstance(msgs, list)
        assert len(msgs) == 1
        assert self.settings_file.exists()

        # Modify instructions to test reset of system-prompt
        inst_path.write_text("Custom prompt", encoding="utf-8")
        msgs_prompt = reset_config(
            "system-prompt",
            settings_path=self.settings_file,
            instructions_path=inst_path,
        )
        assert "Reset: Restored default system prompt" in msgs_prompt[0]
        assert "CORE OBJECTIVE" in inst_path.read_text(encoding="utf-8")

        msgs_all = reset_config(
            "all",
            settings_path=self.settings_file,
            instructions_path=inst_path,
        )
        assert len(msgs_all) == 2
        assert self.settings_file.exists()
        assert inst_path.exists()

    def test_settings_from_dict_rejects_unknown_root_keys(self) -> None:
        with pytest.raises(ValueError) as ctx:
            Settings.from_dict({"modeel": {"name": "llama3"}})
        assert "modeel" in str(ctx.value)

    def test_dataclass_from_dict_rejects_non_dict(self) -> None:
        with pytest.raises(ValueError):
            _dataclass_from_dict(ModelSettings, None)
        with pytest.raises(ValueError):
            _dataclass_from_dict(ModelSettings, "not-a-dict")
        with pytest.raises(ValueError):
            _dataclass_from_dict(ModelSettings, 123)

    def test_subagents_from_list_parsing(self) -> None:
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
        subagents = Settings.from_dict({"subagents": raw}).subagents
        assert len(subagents) == 2
        assert subagents[0].name == "explorer"
        assert subagents[0].mcp_servers == []
        assert subagents[1].name == "tester"
        assert len(subagents[1].mcp_servers) == 1
        assert subagents[1].mcp_servers[0].name == "fetch"

    def test_subagents_from_list_validation(self) -> None:
        with pytest.raises(ValueError):
            Settings.from_dict({"subagents": None})
        with pytest.raises(ValueError):
            Settings.from_dict({"subagents": "not-a-list"})
        with pytest.raises(ValueError):
            Settings.from_dict({"subagents": ["not-a-dict"]})
        with pytest.raises(ValueError):
            Settings.from_dict({"subagents": [{"name": "coder", "mcp_servers": "invalid"}]})

    def test_load_settings_empty_file_raises_value_error(self) -> None:
        self.settings_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError) as ctx:
            load_settings(self.settings_file)
        assert "YAML mapping" in str(ctx.value)

    def test_load_settings_falsy_scalar_yaml_raises_value_error(self) -> None:
        for val in ["", "false\n", "0\n", "[]\n"]:
            self.settings_file.write_text(val, encoding="utf-8")
            with pytest.raises(ValueError):
                load_settings(self.settings_file)


if __name__ == "__main__":
    unittest.main()
