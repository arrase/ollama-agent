from __future__ import annotations

import asyncio
import unittest
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.types import Command

from ollama_agent.agent.agent import AgentRuntime, _process_message_chunk
from ollama_agent.agent.builtin_tools import get_rag_manager, rag_search, set_rag_manager
from ollama_agent.agent.middleware import _extract_tool_name, _stream_tool_events
from ollama_agent.agent.subagents import build_subagents
from ollama_agent.core.prompt_processor import PromptProcessingError
from ollama_agent.settings.config import ModelSettings, Settings, SubAgentSettings


class TestAgentRuntimeComponents(unittest.IsolatedAsyncioTestCase):
    """Unit tests for agent helpers, middleware, subagents, and builtin tools."""

    def setUp(self) -> None:
        set_rag_manager(None)

    def tearDown(self) -> None:
        set_rag_manager(None)

    def test_process_message_chunk_text(self) -> None:
        chunk = MagicMock(type="ai", content="Hello world", additional_kwargs={})
        res = _process_message_chunk(chunk)
        self.assertEqual(res, {"type": "text_delta", "content": "Hello world"})

    def test_process_message_chunk_reasoning(self) -> None:
        chunk = MagicMock(type="ai", content="", additional_kwargs={"reasoning_content": "thinking..."})
        res = _process_message_chunk(chunk)
        self.assertEqual(res, {"type": "reasoning_delta", "content": "thinking..."})

        # When reasoning is hidden
        self.assertIsNone(_process_message_chunk(chunk, hide_reasoning=True))

    def test_process_message_chunk_tool_chunk_ignored(self) -> None:
        chunk = MagicMock(type="tool", content="output", additional_kwargs={})
        self.assertIsNone(_process_message_chunk(chunk))

    def test_extract_tool_name(self) -> None:
        # From request.name
        req1 = MagicMock(tool=None)
        req1.name = "search"
        self.assertEqual(_extract_tool_name(req1), "search")

        # From tool object
        tool_obj = MagicMock()
        tool_obj.name = "calc"
        req2 = MagicMock(spec=["tool"], tool=tool_obj)
        self.assertEqual(_extract_tool_name(req2), "calc")

    async def test_stream_tool_events_emits_events_and_result(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.stream_writer = MagicMock()

        async def dummy_handler(req: Any) -> Any:
            return MagicMock(content="done")

        req = MagicMock(
            runtime=mock_runtime,
            tool_call={"name": "web_search", "args": {"q": "python"}},
        )
        req.name = "web_search"

        with patch("ollama_agent.agent.middleware.get_tool_timeout", return_value=5):
            res = await _stream_tool_events(req, dummy_handler)
            self.assertEqual(res.content, "done")
            # Verify stream_writer was called for tool_call and tool_output
            self.assertEqual(mock_runtime.stream_writer.call_count, 2)
            call_args_list = mock_runtime.stream_writer.call_args_list
            self.assertEqual(call_args_list[0][0][0]["type"], "tool_call")
            self.assertEqual(call_args_list[0][0][0]["name"], "web_search")
            self.assertEqual(call_args_list[1][0][0]["type"], "tool_output")

    async def test_stream_tool_events_timeout_raises(self) -> None:
        async def slow_handler(req: Any) -> Any:
            await asyncio.sleep(0.5)
            return "done"

        req = MagicMock(runtime=None, tool_call=None)
        req.name = "slow_tool"

        with patch("ollama_agent.agent.middleware.get_tool_timeout", return_value=0.01):
            with self.assertRaises(TimeoutError):
                await _stream_tool_events(req, slow_handler)

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

        async with AsyncExitStack() as stack:
            with patch("ollama_agent.agent.subagents.create_ollama_chat_model", AsyncMock(return_value=mock_model)):
                specs = await build_subagents(sa_list, model_settings=ms, exit_stack=stack)
                self.assertEqual(len(specs), 1)
                self.assertEqual(specs[0]["name"], "coder")
                self.assertEqual(specs[0]["description"], "Writes code")
                self.assertIn("You are an expert coder", specs[0]["system_prompt"])

    async def test_build_subagents_invalid_name_raises(self) -> None:
        ms = ModelSettings(name="gemma4:26b", base_url="http://localhost:11434")
        sa_list = [SubAgentSettings(name="", description="Missing name")]

        async with AsyncExitStack() as stack:
            with self.assertRaises(ValueError):
                await build_subagents(sa_list, model_settings=ms, exit_stack=stack)

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

        with patch.object(AgentRuntime, "_build_graph", AsyncMock(return_value=MagicMock())) as mock_bg, \
             patch("ollama_agent.agent.agent.load_instructions", return_value="instructions\n{FILESYSTEM_POLICY}\n{RAG_POLICY}"), \
             patch("ollama_agent.agent.agent.load_fs_policy_sandboxed", return_value="fs_policy"), \
             patch("ollama_agent.agent.agent.ensure_memory_file"), \
             patch("ollama_agent.agent.agent.ensure_agents_file"):
            set_rag_manager(None)
            await runtime.reload()
            self.assertIsNotNone(runtime.graph)
            mock_bg.assert_awaited_once()
            self.assertIn("fs_policy", runtime._instructions)
            self.assertNotIn("{FILESYSTEM_POLICY}", runtime._instructions)
            self.assertNotIn("{RAG_POLICY}", runtime._instructions)

        with patch("ollama_agent.agent.agent.save_settings"), patch.object(AgentRuntime, "reload", AsyncMock()):
            msg = await runtime.set_model("qwen3:32b")
            self.assertEqual(runtime.settings.model.name, "qwen3:32b")
            self.assertIn("qwen3:32b", msg)

        await runtime.aclose()

    async def test_agent_runtime_rag_active_instructions_and_tools(self) -> None:
        settings = Settings()
        runtime = AgentRuntime(settings=settings)

        mock_mgr = MagicMock()
        mock_mgr.current_database = "active_docs"
        set_rag_manager(mock_mgr)

        with patch.object(AgentRuntime, "_build_graph", AsyncMock(return_value=MagicMock())) as mock_bg, \
             patch("ollama_agent.agent.agent.load_instructions", return_value="base\n{FILESYSTEM_POLICY}\n{RAG_POLICY}"), \
             patch("ollama_agent.agent.agent.load_fs_policy_sandboxed", return_value="fs_sandbox"), \
             patch("ollama_agent.agent.agent.load_rag_policy", return_value="rag_policy_content"), \
             patch("ollama_agent.agent.agent.ensure_memory_file"), \
             patch("ollama_agent.agent.agent.ensure_agents_file"):
            await runtime.reload()
            self.assertIn("rag_policy_content", runtime._instructions)
            self.assertIn("fs_sandbox", runtime._instructions)
            self.assertNotIn("{RAG_POLICY}", runtime._instructions)
            mock_bg.assert_awaited_once()

        # Test tool injection in _build_graph
        with patch("ollama_agent.agent.agent.ensure_model_supports_tools", AsyncMock()), \
             patch("ollama_agent.agent.agent.create_ollama_chat_model", AsyncMock(return_value=MagicMock())), \
             patch("ollama_agent.agent.agent.create_summarization_tool_middleware", return_value=MagicMock()), \
             patch("ollama_agent.agent.agent.load_main_mcp_tools", AsyncMock(return_value=[])), \
             patch("ollama_agent.agent.agent.create_deep_agent") as mock_cda, \
             patch.object(AgentRuntime, "_sqlite_checkpointer", AsyncMock(return_value=MagicMock())):
            await runtime._build_graph()
            kwargs = mock_cda.call_args.kwargs
            self.assertIn(rag_search, kwargs["tools"])

        # When RAG is inactive
        mock_mgr.current_database = None
        with patch("ollama_agent.agent.agent.ensure_model_supports_tools", AsyncMock()), \
             patch("ollama_agent.agent.agent.create_ollama_chat_model", AsyncMock(return_value=MagicMock())), \
             patch("ollama_agent.agent.agent.create_summarization_tool_middleware", return_value=MagicMock()), \
             patch("ollama_agent.agent.agent.load_main_mcp_tools", AsyncMock(return_value=[])), \
             patch("ollama_agent.agent.agent.create_deep_agent") as mock_cda, \
             patch.object(AgentRuntime, "_sqlite_checkpointer", AsyncMock(return_value=MagicMock())):
            await runtime._build_graph()
            kwargs = mock_cda.call_args.kwargs
            self.assertNotIn(rag_search, kwargs["tools"])

        set_rag_manager(None)
        await runtime.aclose()


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

        cmd = Command(resume={"decisions": [{"type": "approve"}]})
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

        with patch("ollama_agent.agent.agent.process_prompt_mentions", side_effect=PromptProcessingError("Invalid file mention")):
            events = []
            async for event in runtime.run_streamed("@nonexistent_mention"):
                events.append(event)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["type"], "error")
            self.assertIn("Invalid file mention", events[0]["content"])

