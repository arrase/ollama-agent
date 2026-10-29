from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import jinja2
import pytest
from rich.console import Console

from ollama_agent.agent.builtin_tools import BUILTIN_TOOLS, rag_search, set_rag_manager
from ollama_agent.agent.middleware import stream_tool_events_mw
from ollama_agent.agent.subagents import build_subagents
from ollama_agent.interfaces.commands.models import list_subagents
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
            specs = await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)
            assert len(specs) == 1
            assert specs[0]["name"] == "researcher"
            assert specs[0]["description"] == "Web research analyst"

            prompt = specs[0]["system_prompt"]
            assert "You are researcher, a Web research analyst." in prompt
            assert "Base model: qwen3:32b." in prompt
            assert "Reasoning mode: DEEP." in prompt
            assert "ENVIRONMENT" in prompt
            assert "Operating System:" in prompt
            assert "Working Directory:" not in prompt

    async def test_build_subagents_jinja2_strict_undefined_raises(self) -> None:
        ms = ModelSettings(name="qwen3:32b")
        sa_list = [
            SubAgentSettings(
                name="coder",
                description="Writes code",
                system_prompt="Invalid: {{ non_existent_variable }}",
            )
        ]
        with pytest.raises(jinja2.UndefinedError):
            await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)

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
            specs = await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)
            assert len(specs) == 1
            assert specs[0]["name"] == "coder"
            assert specs[0]["description"] == "Writes code"
            assert "You are an expert coder." in specs[0]["system_prompt"]
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

        with (
            patch("ollama_agent.agent.subagents.ensure_model_supports_tools", AsyncMock()) as mock_ensure,
            patch(
                "ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)
            ) as mock_create,
        ):
            specs = await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)
            assert len(specs) == 1
            mock_ensure.assert_awaited_once_with("deepseek-r1:70b", "http://localhost:11434")
            kwargs = mock_create.call_args.kwargs
            assert kwargs["model"] == "deepseek-r1:70b"
            assert kwargs["context_window"] == 32768

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
        mock_tool.name = "git_status"

        with (
            patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)),
            patch(
                "ollama_agent.agent.subagents.load_subagent_mcp_tools", AsyncMock(return_value=[mock_tool])
            ) as mock_mcp,
        ):
            specs = await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)
            assert len(specs) == 1
            assert "tools" in specs[0]
            assert specs[0]["tools"] == [*BUILTIN_TOOLS, mock_tool]
            assert specs[0]["middleware"] == [stream_tool_events_mw]
            mock_mcp.assert_awaited_once_with("mcp_agent", sa_list[0].mcp_servers)

            assert "interrupt_on" in specs[0]
            interrupt_on = specs[0]["interrupt_on"]
            assert "git_status" in interrupt_on
            assert "execute" in interrupt_on
            assert "write_file" in interrupt_on
            assert "edit_file" in interrupt_on
            assert interrupt_on["git_status"]["allowed_decisions"] == ["approve", "reject"]
            assert callable(interrupt_on["git_status"]["when"])
            assert interrupt_on["git_status"]["when"](MagicMock())

    async def test_build_subagents_with_zero_tool_mcp_server_still_wires_interrupt_on(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [
            SubAgentSettings(
                name="mcp_agent",
                description="Agent with failing/empty MCP server",
                system_prompt="MCP instructions.",
                mcp_servers=[SubAgentMCPServer(name="empty", command="npx")],
            )
        ]
        mock_model = MagicMock()

        with (
            patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)),
            patch("ollama_agent.agent.subagents.load_subagent_mcp_tools", AsyncMock(return_value=[])),
        ):
            specs = await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)
            assert len(specs) == 1
            assert specs[0]["tools"] == list(BUILTIN_TOOLS)
            assert "interrupt_on" in specs[0]
            assert "execute" in specs[0]["interrupt_on"]

    async def test_build_subagents_with_custom_should_interrupt(self) -> None:
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
        mock_tool.name = "git_commit"

        custom_policy = MagicMock(return_value=False)

        with (
            patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)),
            patch("ollama_agent.agent.subagents.load_subagent_mcp_tools", AsyncMock(return_value=[mock_tool])),
        ):
            specs = await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=custom_policy)
            assert len(specs) == 1
            interrupt_on = specs[0]["interrupt_on"]
            assert "git_commit" in interrupt_on
            when_fn = interrupt_on["git_commit"]["when"]
            mock_req = MagicMock()
            assert not when_fn(mock_req)
            custom_policy.assert_called_once_with(mock_req)

    async def test_build_subagents_without_mcp_servers_always_wires_tools_and_interrupt_on(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [
            SubAgentSettings(
                name="plain_agent",
                description="Agent without MCP tools",
                system_prompt="Plain instructions.",
                mcp_servers=[],
            )
        ]
        mock_model = MagicMock()

        with patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)):
            specs = await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)
            assert len(specs) == 1
            assert specs[0]["tools"] == list(BUILTIN_TOOLS)
            assert "interrupt_on" in specs[0]
            assert "execute" in specs[0]["interrupt_on"]
            assert "write_file" in specs[0]["interrupt_on"]
            assert "edit_file" in specs[0]["interrupt_on"]

    async def test_build_subagents_invalid_name_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [SubAgentSettings(name="", description="Missing name", system_prompt="prompt")]
        with pytest.raises(ValueError):
            await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)

    async def test_build_subagents_empty_description_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [SubAgentSettings(name="coder", description="", system_prompt="prompt")]
        with pytest.raises(ValueError):
            await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)

    async def test_build_subagents_missing_system_prompt_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [SubAgentSettings(name="coder", description="Writes code", system_prompt="")]
        with pytest.raises(ValueError):
            await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)

    def test_list_subagents_empty(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120)
        settings = Settings()
        list_subagents(console, settings)
        output = console.export_text()
        assert "No subagents configured." in output
        assert "Configure subagents in" in output

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
        assert "Configured Subagents" in output
        assert "researcher" in output
        assert "Web research" in output
        assert "gemma4:26b" in output
        assert "32768" in output
        assert "brave-search, fetch" in output

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
        assert "Configured Subagents" in output
        assert "coder" in output
        assert "qwen3.8:27b (inherited)" in output
        assert "10000 (inherited)" in output
        assert "-" in output

    async def test_build_subagents_with_mcp_includes_active_rag_tool(self) -> None:
        ms = ModelSettings(name="gemma4:26b")
        sa_list = [
            SubAgentSettings(
                name="researcher",
                description="Researcher",
                system_prompt="Prompt",
                mcp_servers=[SubAgentMCPServer(name="srv", command="npx")],
            )
        ]
        mock_rag = MagicMock()
        mock_rag.current_database = "docs"
        mock_mcp_tool = MagicMock(name="mcp_tool")
        set_rag_manager(mock_rag)
        try:
            with (
                patch(
                    "ollama_agent.agent.subagents.create_ollama_chat_model",
                    AsyncMock(return_value=MagicMock()),
                ),
                patch(
                    "ollama_agent.agent.subagents.load_subagent_mcp_tools",
                    AsyncMock(return_value=[mock_mcp_tool]),
                ),
            ):
                specs = await build_subagents(sa_list, model_settings=ms, should_interrupt_tool=lambda _: True)
                assert len(specs) == 1
                assert rag_search in specs[0]["tools"]
                assert mock_mcp_tool in specs[0]["tools"]
        finally:
            set_rag_manager(None)


if __name__ == "__main__":
    unittest.main()
