from __future__ import annotations

import io
import unittest
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

from rich.console import Console

from ollama_agent.streaming.base import StreamingRenderer
from ollama_agent.streaming.console_renderer import ConsoleStreamingRenderer
from ollama_agent.streaming.events import stream_agent_events


class DummyRenderer(StreamingRenderer):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.closed = False

    def on_event(self, event: dict[str, Any]) -> None:
        super().on_event(event)
        self.events.append(event)

    def on_text_delta(self, event: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class TestStreamingSystem(unittest.IsolatedAsyncioTestCase):
    """Unit tests for streaming base classes, console renderer, and event pipeline."""

    def test_base_streaming_renderer_dispatch(self) -> None:
        renderer = DummyRenderer()
        renderer.on_event({"type": "text_delta", "content": "hello"})
        self.assertEqual(len(renderer.events), 1)

    def test_console_streaming_renderer_deltas(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        renderer = ConsoleStreamingRenderer(console=console)

        renderer.on_event({"type": "text_delta", "content": "Hello world\n"})
        renderer.on_event({"type": "reasoning_delta", "content": "Thinking step\n"})
        renderer.on_event({"type": "tool_call", "name": "search"})
        renderer.on_event({"type": "tool_output", "name": "search", "output_len": 42})
        renderer.on_event({"type": "error", "content": "Network failure"})
        renderer.on_event({"type": "warning", "content": "Slow connection"})
        renderer.close()

        out = console.export_text()
        self.assertIn("Hello world", out)
        self.assertIn("Thinking step", out)
        self.assertIn("Calling tool", out)
        self.assertIn("Tool output received", out)
        self.assertIn("Network failure", out)
        self.assertIn("Slow connection", out)

    async def test_stream_agent_events_pipeline(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(prompt: str, thread_id: str = "") -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "text_delta", "content": "Hello"}
            yield {"type": "agent_update", "content": "state"}
            yield {"type": "text_delta", "content": "World"}

        mock_runtime.run_streamed = fake_stream

        renderer = DummyRenderer()
        await stream_agent_events(
            mock_runtime,
            "test prompt",
            renderer,
            ignore={"agent_update"},
            auto_close=True,
        )

        self.assertTrue(renderer.closed)
        types = [e["type"] for e in renderer.events]
        self.assertEqual(types, ["text_delta", "text_delta"])
        self.assertNotIn("agent_update", types)
