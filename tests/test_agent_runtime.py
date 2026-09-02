from __future__ import annotations

import asyncio
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessageChunk
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from rich.console import Console

from ollama_agent.agent.agent import AgentRuntime, _prepare_instructions
from ollama_agent.agent.builtin_tools import rag_search, set_rag_manager
from ollama_agent.agent.middleware import _stream_tool_events
from ollama_agent.agent.subagents import build_subagents, list_subagents
from ollama_agent.core.prompt_processor import PromptProcessingError
from ollama_agent.settings.config import (
    ModelSettings,
    RuntimeSettings,
    Settings,
    SubAgentMCPServer,
    SubAgentSettings,
    _default_instructions,
)
from ollama_agent.streaming.parsers import ThinkTagParser


class TestAgentRuntimeComponents(unittest.IsolatedAsyncioTestCase):
    """Unit tests for agent helpers, middleware, subagents, and builtin tools."""

    def setUp(self) -> None:
        set_rag_manager(None)

    def tearDown(self) -> None:
        set_rag_manager(None)

    def test_process_message_chunk_text(self) -> None:
        parser = ThinkTagParser()
        chunk = MagicMock(type="ai", content="Hello world", additional_kwargs={})
        events = parser.process_chunk(chunk)
        self.assertEqual(events, [{"type": "text_delta", "content": "Hello world"}])

    def test_process_message_chunk_reasoning(self) -> None:
        parser = ThinkTagParser()
        chunk = MagicMock(type="ai", content="", additional_kwargs={"reasoning_content": "thinking..."})
        events = parser.process_chunk(chunk)
        self.assertEqual(events, [{"type": "reasoning_delta", "content": "thinking..."}])

        # When reasoning is hidden
        parser2 = ThinkTagParser()
        self.assertEqual(parser2.process_chunk(chunk, hide_reasoning=True), [])

    def test_process_message_chunk_tool_chunk_ignored(self) -> None:
        parser = ThinkTagParser()
        chunk = MagicMock(type="tool", content="output", additional_kwargs={})
        self.assertEqual(parser.process_chunk(chunk), [])

        # Real ToolMessageChunk
        real_tool_chunk = ToolMessageChunk(content="output", tool_call_id="call-1")
        self.assertEqual(parser.process_chunk(real_tool_chunk), [])

    async def test_stream_tool_events_emits_events_and_result(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.stream_writer = MagicMock()

        async def dummy_handler(req: Any) -> Any:
            return MagicMock(content="done")

        req = ToolCallRequest(
            tool_call={"name": "web_search", "args": {"q": "python"}, "id": "call-1"},
            tool=None,
            state={},
            runtime=mock_runtime,
        )

        with patch("ollama_agent.agent.middleware.get_tool_timeout", return_value=5):
            res = await _stream_tool_events(req, dummy_handler)
            self.assertEqual(res.content, "done")
            # Verify stream_writer was called for tool_call and tool_output
            self.assertEqual(mock_runtime.stream_writer.call_count, 2)
            call_args_list = mock_runtime.stream_writer.call_args_list
            self.assertEqual(call_args_list[0][0][0]["type"], "tool_call")
            self.assertEqual(call_args_list[0][0][0]["name"], "web_search")
            self.assertEqual(call_args_list[1][0][0]["type"], "tool_output")

    async def test_stream_tool_events_task_agent_name_from_metadata(self) -> None:
        mock_runtime = MagicMock()

        async def dummy_handler(req: Any) -> Any:
            return MagicMock(content="ok")

        req = ToolCallRequest(
            tool_call={
                "name": "task",
                "args": {"name": "researcher"},
                "id": "call-2",
                "metadata": {"lc_agent_name": "meta_agent"},
            },
            tool=None,
            state={},
            runtime=mock_runtime,
        )

        with patch("ollama_agent.agent.middleware.get_tool_timeout", return_value=5):
            await _stream_tool_events(req, dummy_handler)

        call_event = mock_runtime.stream_writer.call_args_list[0][0][0]
        self.assertEqual(call_event["agent_name"], "researcher")
        out_event = mock_runtime.stream_writer.call_args_list[1][0][0]
        self.assertEqual(out_event["agent_name"], "researcher")

    async def test_stream_tool_events_timeout_returns_tool_message(self) -> None:
        async def slow_handler(req: Any) -> Any:
            await asyncio.sleep(0.5)
            return "done"

        mock_runtime = MagicMock()
        req = ToolCallRequest(
            tool_call={"name": "slow_tool", "args": {}, "id": "call-3"},
            tool=None,
            state={},
            runtime=mock_runtime,
        )

        with patch("ollama_agent.agent.middleware.get_tool_timeout", return_value=0.01):
            result = await _stream_tool_events(req, slow_handler)

        self.assertEqual(result.tool_call_id, "call-3")
        self.assertEqual(result.name, "slow_tool")
        self.assertEqual(result.status, "error")
        self.assertIn("timed out after 0.01s", result.content)
        self.assertEqual(mock_runtime.stream_writer.call_count, 2)
        call_events = [c[0][0]["type"] for c in mock_runtime.stream_writer.call_args_list]
        self.assertEqual(call_events, ["tool_call", "tool_output"])

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

        with patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)):
            specs = await build_subagents(sa_list, model_settings=ms)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0]["name"], "coder")
            self.assertEqual(specs[0]["description"], "Writes code")
            self.assertIn("You are an expert coder", specs[0]["system_prompt"])

    async def test_build_subagents_jinja2_rendering(self) -> None:
        ms = ModelSettings(name="qwen3:32b", base_url="http://localhost:11434")
        sa_list = [
            SubAgentSettings(
                name="jinja_coder",
                description="Writes code with template",
                system_prompt="Subagent {{ subagent.name }} uses {{ model_settings.name }}.",
            )
        ]
        mock_model = MagicMock()

        with patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)):
            specs = await build_subagents(sa_list, model_settings=ms)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0]["name"], "jinja_coder")
            self.assertIn("Subagent jinja_coder uses qwen3:32b.", specs[0]["system_prompt"])
            self.assertIn("Operating System:", specs[0]["system_prompt"])

    async def test_build_subagents_invalid_name_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b", base_url="http://localhost:11434")
        sa_list = [SubAgentSettings(name="", description="Missing name")]

        with self.assertRaises(ValueError):
            await build_subagents(sa_list, model_settings=ms)

    async def test_build_subagents_missing_system_prompt_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b", base_url="http://localhost:11434")
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

    async def test_rag_search_uninitialized(self) -> None:
        set_rag_manager(None)
        res = await rag_search.ainvoke({"query": "test"})
        self.assertFalse(res["success"])
        self.assertIn("not initialized", res["error"])

    async def test_rag_search_no_db_loaded(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.current_database = None
        set_rag_manager(mock_mgr)

        res = await rag_search.ainvoke({"query": "test"})
        self.assertFalse(res["success"])
        self.assertIn("No RAG database loaded", res["error"])

    async def test_rag_search_success(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.current_database = "my_docs"
        mock_mgr.search = AsyncMock(
            return_value=[
                {
                    "content": "Python is a programming language.",
                    "filename": "guide.md",
                    "source": "/docs/guide.md",
                    "score": 0.95,
                    "chunk_index": 0,
                }
            ]
        )
        set_rag_manager(mock_mgr)

        res = await rag_search.ainvoke({"query": "python"})
        self.assertTrue(res["success"])
        self.assertIn("Python is a programming language.", res["context"])
        self.assertIn("guide.md", res["context"])

    async def test_agent_runtime_lifecycle(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)

        with (
            patch.object(AgentRuntime, "_build_graph", AsyncMock(return_value=MagicMock())) as mock_bg,
            patch(
                "ollama_agent.agent.agent.load_instructions",
                return_value="instructions\n{% if runtime.allow_traversal %}fs_traversal{% else %}fs_sandboxed{% endif %}",
            ),
            patch("ollama_agent.agent.agent.ensure_memory_file"),
        ):
            set_rag_manager(None)
            await runtime.reload()
            self.assertIsNotNone(runtime.graph)
            mock_bg.assert_awaited_once()
            self.assertIn("fs_sandboxed", runtime._instructions)
            self.assertNotIn("fs_traversal", runtime._instructions)

        with patch("ollama_agent.agent.agent.save_settings"), patch.object(AgentRuntime, "reload", AsyncMock()):
            msg = await runtime.set_model("qwen3:32b")
            self.assertEqual(runtime.settings.model.name, "qwen3:32b")
            self.assertIn("qwen3:32b", msg)

            effort_msg = await runtime.set_reasoning_effort("high")
            self.assertEqual(runtime.settings.model.reasoning_effort, "high")
            self.assertIn("high", effort_msg)

            with self.assertRaises(ValueError):
                await runtime.set_reasoning_effort("invalid_effort")

        await runtime.aclose()

    async def test_agent_runtime_rag_active_instructions_and_tools(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)

        mock_mgr = MagicMock()
        mock_mgr.current_database = "active_docs"
        set_rag_manager(mock_mgr)

        with (
            patch.object(AgentRuntime, "_build_graph", AsyncMock(return_value=MagicMock())) as mock_bg,
            patch(
                "ollama_agent.agent.agent.load_instructions",
                return_value="base\n{% if runtime.allow_traversal %}fs_traversal{% else %}fs_sandbox{% endif %}\nrag_policy_content",
            ),
            patch("ollama_agent.agent.agent.ensure_memory_file"),
        ):
            await runtime.reload()
            self.assertIn("rag_policy_content", runtime._instructions)
            self.assertIn("fs_sandbox", runtime._instructions)
            mock_bg.assert_awaited_once()

        # Test tool injection in _build_graph
        with (
            patch("ollama_agent.agent.agent.ensure_model_supports_tools", AsyncMock()),
            patch("ollama_agent.agent.agent.create_ollama_chat_model", AsyncMock(return_value=MagicMock())),
            patch("ollama_agent.agent.agent.create_summarization_tool_middleware", return_value=MagicMock()),
            patch("ollama_agent.agent.agent.load_main_mcp_tools", AsyncMock(return_value=[])),
            patch("ollama_agent.agent.agent.create_deep_agent") as mock_cda,
            patch.object(AgentRuntime, "_sqlite_checkpointer", AsyncMock(return_value=MagicMock())),
        ):
            await runtime._build_graph()
            kwargs = mock_cda.call_args.kwargs
            self.assertIn(rag_search, kwargs["tools"])
            self.assertEqual(kwargs["skills"], (("/system_skills/", "Built-in"), ("/skills/", "User")))

        # When RAG is inactive, rag_search is not present
        mock_mgr.current_database = None
        with (
            patch("ollama_agent.agent.agent.ensure_model_supports_tools", AsyncMock()),
            patch("ollama_agent.agent.agent.create_ollama_chat_model", AsyncMock(return_value=MagicMock())),
            patch("ollama_agent.agent.agent.create_summarization_tool_middleware", return_value=MagicMock()),
            patch("ollama_agent.agent.agent.load_main_mcp_tools", AsyncMock(return_value=[])),
            patch("ollama_agent.agent.agent.create_deep_agent") as mock_cda,
            patch.object(AgentRuntime, "_sqlite_checkpointer", AsyncMock(return_value=MagicMock())),
        ):
            await runtime._build_graph()
            kwargs = mock_cda.call_args.kwargs
            self.assertNotIn(rag_search, kwargs["tools"])

        set_rag_manager(None)
        await runtime.aclose()

    def test_prepare_instructions_allow_traversal_true(self) -> None:
        settings = Settings()
        settings.runtime.allow_traversal = True
        with (
            patch("ollama_agent.agent.agent.load_instructions", side_effect=_default_instructions),
            patch("ollama_agent.agent.agent.ensure_memory_file"),
        ):
            instructions = _prepare_instructions(settings)
        self.assertIn("You have full access to the host filesystem", instructions)
        self.assertNotIn("operate on a virtual root", instructions)
        self.assertIn("Working Directory", instructions)

    def test_prepare_instructions_allow_traversal_false(self) -> None:
        settings = Settings()
        settings.runtime.allow_traversal = False
        with (
            patch("ollama_agent.agent.agent.load_instructions", side_effect=_default_instructions),
            patch("ollama_agent.agent.agent.ensure_memory_file"),
        ):
            instructions = _prepare_instructions(settings)
        self.assertIn("operate on a virtual root", instructions)
        self.assertNotIn("You have full access to the host filesystem", instructions)
        self.assertIn("Working Directory", instructions)

    def test_prepare_instructions_context_variables(self) -> None:
        settings = Settings(
            model=ModelSettings(name="test-model"),
            runtime=RuntimeSettings(allow_traversal=True),
        )
        custom_template = (
            "Model: {{ model.name }}\n"
            "Traversal: {{ runtime.allow_traversal }}\n"
            "RAG Top K: {{ rag.default_top_k }}\n"
            "Settings Model: {{ settings.model.name }}\n"
            "RAG Active: {{ rag_active }}\n"
            "RAG DB: {{ rag_database }}"
        )
        with (
            patch("ollama_agent.agent.agent.load_instructions", return_value=custom_template),
            patch("ollama_agent.agent.agent.ensure_memory_file"),
        ):
            instructions = _prepare_instructions(settings)
            self.assertIn("Model: test-model", instructions)
            self.assertIn("Traversal: True", instructions)
            self.assertIn("RAG Top K: 5", instructions)
            self.assertIn("Settings Model: test-model", instructions)
            self.assertIn("RAG Active: False", instructions)
            self.assertIn("RAG DB: ", instructions)

    def test_prepare_instructions_rag_inactive(self) -> None:
        settings = Settings()
        set_rag_manager(None)
        with (
            patch("ollama_agent.agent.agent.load_instructions", side_effect=_default_instructions),
            patch("ollama_agent.agent.agent.ensure_memory_file"),
        ):
            instructions = _prepare_instructions(settings)
        self.assertNotIn("# RAG POLICY", instructions)

    def test_prepare_instructions_rag_active(self) -> None:
        settings = Settings()
        mock_mgr = MagicMock()
        mock_mgr.current_database = "knowledge_base"
        set_rag_manager(mock_mgr)
        with (
            patch("ollama_agent.agent.agent.load_instructions", side_effect=_default_instructions),
            patch("ollama_agent.agent.agent.ensure_memory_file"),
        ):
            instructions = _prepare_instructions(settings)
        self.assertIn("# RAG POLICY", instructions)
        self.assertIn("('knowledge_base')", instructions)

    async def test_agent_runtime_run_streamed(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)

        async def mock_astream(*args: Any, **kwargs: Any):
            yield "custom", {"type": "tool_call", "name": "test_tool"}
            ai_chunk = MagicMock(type="ai", content="Hello response", additional_kwargs={})
            yield "messages", (ai_chunk,)

        mock_graph = MagicMock()
        mock_graph.astream = mock_astream
        mock_state = MagicMock(interrupts=[])
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        runtime.graph = mock_graph

        events = []
        async for event in runtime.run_streamed("Hello agent"):
            events.append(event)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"type": "tool_call", "name": "test_tool"})
        self.assertEqual(events[1], {"type": "text_delta", "content": "Hello response"})

    async def test_agent_runtime_run_streamed_interrupt(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)

        async def mock_astream(*args: Any, **kwargs: Any):
            ai_chunk = MagicMock(type="ai", content="Checking...", additional_kwargs={})
            yield "messages", (ai_chunk,)

        mock_graph = MagicMock()
        mock_graph.astream = mock_astream
        mock_state = MagicMock(interrupts=[MagicMock(value={"action_requests": [{"name": "write"}]})])
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        runtime.graph = mock_graph

        events = []
        async for event in runtime.run_streamed("Do write"):
            events.append(event)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"type": "text_delta", "content": "Checking..."})
        self.assertEqual(events[1]["type"], "interrupt")
        self.assertEqual(len(events[1]["interrupts"]), 1)

    async def test_agent_runtime_run_streamed_command_input(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)

        async def mock_astream(inputs: Any, *args: Any, **kwargs: Any):
            self.assertIsInstance(inputs, Command)
            ai_chunk = MagicMock(type="ai", content="Resumed", additional_kwargs={})
            yield "messages", (ai_chunk,)

        mock_graph = MagicMock()
        mock_graph.astream = mock_astream
        mock_state = MagicMock(interrupts=[])
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        runtime.graph = mock_graph

        cmd: Command[Any] = Command(resume={"decisions": [{"type": "approve"}]})
        events = []
        async for event in runtime.run_streamed(cmd):
            events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], {"type": "text_delta", "content": "Resumed"})

    async def test_agent_runtime_run_streamed_prompt_error(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)
        mock_graph = MagicMock()
        runtime.graph = mock_graph

        with patch(
            "ollama_agent.agent.agent.process_prompt_mentions",
            side_effect=PromptProcessingError("Invalid file mention"),
        ):
            events = []
            async for event in runtime.run_streamed("@nonexistent_mention"):
                events.append(event)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["type"], "error")
            self.assertIn("Invalid file mention", events[0]["content"])

    async def test_count_effective_tokens_without_and_with_summary(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)
        mock_graph = MagicMock()

        # Without summarization event
        msg1 = HumanMessage(content="Hello")
        msg2 = AIMessage(content="World")
        mock_graph.aget_state = AsyncMock(return_value=MagicMock(values={"messages": [msg1, msg2]}))
        runtime.graph = mock_graph

        count = await runtime.count_effective_tokens("thread-1")
        self.assertGreater(count, 0)

        # With summarization event
        summary_msg = HumanMessage(content="Summary of earlier conversation")
        mock_graph.aget_state = AsyncMock(
            return_value=MagicMock(
                values={
                    "messages": [msg1, msg2],
                    "_summarization_event": {
                        "cutoff_index": 1,
                        "summary_message": summary_msg,
                    },
                }
            )
        )
        count_with_summary = await runtime.count_effective_tokens("thread-1")
        self.assertGreater(count_with_summary, 0)

    async def test_sqlite_checkpointer_initializes_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "history.db"
            runtime = AgentRuntime(settings=Settings())
            with patch("ollama_agent.agent.agent.HISTORY_DB_PATH", db_path):
                saver = await runtime._sqlite_checkpointer()
                self.assertIsNotNone(saver)
                self.assertTrue(db_path.exists())

                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = {row[0] for row in cursor.fetchall()}
                conn.close()

                self.assertIn("writes", tables)
                self.assertIn("checkpoints", tables)
            await runtime.aclose()

    async def test_checkpointer_persists_across_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "history.db"
            runtime = AgentRuntime(settings=Settings())
            with (
                patch("ollama_agent.agent.agent.HISTORY_DB_PATH", db_path),
                patch.object(AgentRuntime, "_build_graph", AsyncMock(return_value=MagicMock())),
            ):
                cp1 = await runtime._sqlite_checkpointer()
                self.assertIsNotNone(cp1)
                await runtime.reload()
                cp2 = await runtime._sqlite_checkpointer()
                self.assertIs(cp1, cp2)
                self.assertIsNotNone(cp2.conn)
            await runtime.aclose()
            self.assertIsNone(runtime._checkpointer)

    async def test_agent_runtime_set_context_window(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)

        with (
            patch.object(AgentRuntime, "reload", AsyncMock()) as mock_reload,
            patch("ollama_agent.agent.agent.save_settings") as mock_save,
        ):
            res = await runtime.set_context_window(16384)
            self.assertEqual(runtime.settings.model.context_window, 16384)
            mock_save.assert_called_once_with(runtime.settings)
            mock_reload.assert_awaited_once()
            self.assertIn("16384", res)

            res2 = await runtime.set_context_window("max")
            self.assertEqual(runtime.settings.model.context_window, "max")
            self.assertIn("max", res2)
