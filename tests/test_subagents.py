from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import jinja2
from rich.console import Console

from ollama_agent.agent.subagents import build_subagents, list_subagents
from ollama_agent.settings.config import (
    ModelSettings,
    Settings,
    SubAgentMCPServer,
    SubAgentSettings,
)


class TestSubAgents(unittest.IsolatedAsyncioTestCase):
    """Unit tests for subagent building, listing, and Jinja2 templating."""

    async def test_build_subagents_jinja2_rendering(self) -> None:
        ms = ModelSettings(
            name="qwen3:32b",
            base_url="http://localhost:11434",
            reasoning_effort="high",
        )
        sa_list = [
            SubAgentSettings(
                name="researcher",
                description="Web research analyst",
                system_prompt=(
                    "You are {{ subagent.name }}, a {{ subagent.description }}.\n"
                    "Base model: {{ model_settings.name }}.\n"
                    "{% if model_settings.reasoning_effort == 'high' %}"
                    "Reasoning mode: DEEP."
                    "{% endif %}"
                ),
            )
        ]
        mock_model = MagicMock()

        with patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)):
            specs = await build_subagents(sa_list, model_settings=ms)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0]["name"], "researcher")
            self.assertEqual(specs[0]["description"], "Web research analyst")

            prompt = specs[0]["system_prompt"]
            self.assertIn("You are researcher, a Web research analyst.", prompt)
            self.assertIn("Base model: qwen3:32b.", prompt)
            self.assertIn("Reasoning mode: DEEP.", prompt)
            self.assertIn("ENVIRONMENT", prompt)
            self.assertIn("Operating System:", prompt)
            self.assertNotIn("Working Directory:", prompt)

    async def test_build_subagents_jinja2_strict_undefined_raises(self) -> None:
        ms = ModelSettings(name="qwen3:32b")
        sa_list = [
            SubAgentSettings(
                name="coder",
                description="Writes code",
                system_prompt="Invalid: {{ non_existent_variable }}",
            )
        ]
        with self.assertRaises(jinja2.UndefinedError):
            await build_subagents(sa_list, model_settings=ms)

    async def test_build_subagents_valid(self) -> None:
        ms = ModelSettings(name="gemma4:26b", base_url="http://localhost:11434")
        sa_list = [
            SubAgentSettings(
                name="coder",
                description="Writes code",
                system_prompt="You are an expert coder.",
            )
        ]
        mock_model = MagicMock()

        with patch(
            "ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)
        ) as mock_create:
            specs = await build_subagents(sa_list, model_settings=ms)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0]["name"], "coder")
            self.assertEqual(specs[0]["description"], "Writes code")
            self.assertIn("You are an expert coder.", specs[0]["system_prompt"])
            mock_create.assert_awaited_once()

    async def test_build_subagents_custom_model_and_context(self) -> None:
        ms = ModelSettings(name="default_model:latest", base_url="http://localhost:11434", context_window=8192)
        sa_list = [
            SubAgentSettings(
                name="specialist",
                description="Custom model subagent",
                system_prompt="Specialist instructions.",
                model="deepseek-r1:70b",
                context_window=32768,
            )
        ]
        mock_model = MagicMock()

        with patch(
            "ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)
        ) as mock_create:
            specs = await build_subagents(sa_list, model_settings=ms)
            self.assertEqual(len(specs), 1)
            kwargs = mock_create.call_args.kwargs
            self.assertEqual(kwargs["model"], "deepseek-r1:70b")
            self.assertEqual(kwargs["context_window"], 32768)

    async def test_build_subagents_with_mcp_servers(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [
            SubAgentSettings(
                name="mcp_agent",
                description="Agent with MCP tools",
                system_prompt="MCP instructions.",
                mcp_servers=[SubAgentMCPServer(name="git", command="npx")],
            )
        ]
        mock_model = MagicMock()
        mock_tool = MagicMock()

        with (
            patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)),
            patch(
                "ollama_agent.agent.subagents.load_subagent_mcp_tools", AsyncMock(return_value=[mock_tool])
            ) as mock_mcp,
        ):
            specs = await build_subagents(sa_list, model_settings=ms)
            self.assertEqual(len(specs), 1)
            self.assertIn("tools", specs[0])
            self.assertEqual(specs[0]["tools"], [mock_tool])
            mock_mcp.assert_awaited_once_with("mcp_agent", sa_list[0].mcp_servers)

    async def test_build_subagents_invalid_name_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [SubAgentSettings(name="", description="Missing name", system_prompt="prompt")]
        with self.assertRaises(ValueError):
            await build_subagents(sa_list, model_settings=ms)

    async def test_build_subagents_empty_description_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [SubAgentSettings(name="coder", description="", system_prompt="prompt")]
        with self.assertRaises(ValueError):
            await build_subagents(sa_list, model_settings=ms)

    async def test_build_subagents_missing_system_prompt_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [SubAgentSettings(name="coder", description="Writes code", system_prompt="")]
        with self.assertRaises(ValueError):
            await build_subagents(sa_list, model_settings=ms)

    def test_list_subagents_empty(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120)
        settings = Settings()
        list_subagents(console, settings)
        output = console.export_text()
        self.assertIn("No subagents configured.", output)
        self.assertIn("Configure subagents in", output)

    def test_list_subagents_populated(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120)
        settings = Settings()
        settings.subagents = [
            SubAgentSettings(
                name="researcher",
                description="Web research specialist",
                system_prompt="You are a research analyst.",
                model="gemma4:26b",
                context_window=32768,
                mcp_servers=[
                    SubAgentMCPServer(name="brave-search", command="npx"),
                    SubAgentMCPServer(name="fetch", command="uvx"),
                ],
            ),
        ]
        list_subagents(console, settings)
        output = console.export_text()
        self.assertIn("Configured Subagents", output)
        self.assertIn("researcher", output)
        self.assertIn("Web research", output)
        self.assertIn("gemma4:26b", output)
        self.assertIn("32768", output)
        self.assertIn("brave-search, fetch", output)

    def test_list_subagents_inherited_fields(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120)
        settings = Settings()
        settings.model.name = "qwen3.8:27b"
        settings.model.context_window = 10000
        settings.subagents = [
            SubAgentSettings(
                name="coder",
                description="Code writer",
                system_prompt="You write code.",
                model="",
                context_window=0,
                mcp_servers=[],
            )
        ]
        list_subagents(console, settings)
        output = console.export_text()
        self.assertIn("Configured Subagents", output)
        self.assertIn("coder", output)
        self.assertIn("qwen3.8:27b (inherited)", output)
        self.assertIn("10000 (inherited)", output)
        self.assertIn("-", output)


if __name__ == "__main__":
    unittest.main()
