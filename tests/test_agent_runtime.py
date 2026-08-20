from __future__ import annotations

import asyncio
import unittest
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from ollama_agent.agent.agent import _process_message_chunk
from ollama_agent.agent.builtin_tools import get_rag_manager, rag_search, set_rag_manager
from ollama_agent.agent.middleware import _extract_tool_name, _stream_tool_events
from ollama_agent.agent.subagents import _build_spec, build_subagents
from ollama_agent.settings.config import ModelSettings, SubAgentSettings


class TestAgentRuntimeComponents(unittest.IsolatedAsyncioTestCase):
    """Unit tests for agent helpers, middleware, subagents, and builtin tools."""

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
