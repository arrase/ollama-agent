from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock, patch

from rich.console import Console

from ollama_agent.streaming import build_approval_decisions, extract_action_requests
from ollama_agent.streaming.base import StreamingRenderer
from ollama_agent.streaming.console_renderer import ConsoleStreamingRenderer
from ollama_agent.streaming.events import stream_agent_events


class DummyRenderer(StreamingRenderer):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.closed = False

    def on_event(self, event: dict[str, Any]) -> None:
        super().on_event(event)
        self.events.append(event)

    def on_text_delta(self, event: dict[str, Any]) -> None:
        pass

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        pass

    def on_tool_call(self, event: dict[str, Any]) -> None:
        pass

    def on_warning(self, event: dict[str, Any]) -> None:
        self.warnings.append(event)

    def on_error(self, event: dict[str, Any]) -> None:
        self.errors.append(event)

    def close(self) -> None:
        self.closed = True


class TestStreamingSystem(unittest.IsolatedAsyncioTestCase):
    """Unit tests for streaming base classes, console renderer, and event pipeline."""

    def test_base_streaming_renderer_dispatch(self) -> None:
        renderer = DummyRenderer()
        renderer.on_event({"type": "text_delta", "content": "hello"})
        self.assertEqual(len(renderer.events), 1)

    def test_console_streaming_renderer_skips_unknown_type(self) -> None:
        renderer = DummyRenderer()
        renderer.on_event({"type": "agent_update", "content": "state"})
        # The event reaches DummyRenderer.on_event which always appends,
        # but no type-specific handler (on_agent_update) was called — verify
        # it didn't crash and no warnings/errors were recorded.
        self.assertEqual(len(renderer.warnings), 0)
        self.assertEqual(len(renderer.errors), 0)

    def test_console_streaming_renderer_reasoning_is_pure_delta(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        renderer = ConsoleStreamingRenderer(console=console)

        renderer.on_event({"type": "reasoning_delta", "content": "the"})
        renderer.on_event({"type": "reasoning_delta", "content": "the"})
        renderer.close()

        out = console.export_text()
        self.assertIn("thethe", out)

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
            yield {"type": "tool_call", "name": "search"}
            yield {"type": "text_delta", "content": "World"}

        mock_runtime.run_streamed = fake_stream

        renderer = DummyRenderer()
        await stream_agent_events(mock_runtime, "test prompt", renderer)

        self.assertTrue(renderer.closed)
        types = [e["type"] for e in renderer.events]
        self.assertEqual(types, ["text_delta", "tool_call", "text_delta"])

    async def test_stream_agent_events_returns_completed_true(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(prompt: str, thread_id: str = "") -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "text_delta", "content": "Hi"}

        mock_runtime.run_streamed = fake_stream
        renderer = DummyRenderer()
        completed = await stream_agent_events(mock_runtime, "p", renderer)
        self.assertTrue(completed)
        self.assertTrue(renderer.closed)


class TestInterruptHandling(unittest.IsolatedAsyncioTestCase):
    """Tests for graceful interrupt handling (approval prompts and non-interactive mode)."""

    @staticmethod
    def _make_renderer() -> tuple[ConsoleStreamingRenderer, Console]:
        console = Console(file=io.StringIO(), record=True, force_terminal=False)
        return ConsoleStreamingRenderer(console=console), console

    @staticmethod
    def _interrupt_event() -> dict[str, Any]:
        return {
            "type": "interrupt",
            "interrupts": [
                SimpleNamespace(
                    value={
                        "action_requests": [
                            {"name": "execute", "args": {"command": "ls"}},
                        ]
                    }
                )
            ],
        }

    async def test_non_interactive_stdin_returns_none_with_hint(self) -> None:
        renderer, console = self._make_renderer()
        fake_stdin = SimpleNamespace(isatty=lambda: False)
        with patch("ollama_agent.streaming.console_renderer.sys.stdin", new=fake_stdin):
            result = await renderer.handle_interrupt(self._interrupt_event(), MagicMock())
        self.assertIsNone(result)
        self.assertIn("--yolo", console.export_text())

    async def test_approval_choices(self) -> None:
        runtime = MagicMock()
        runtime.auto_approved_tools = set()

        cases: list[tuple[list[str], str | None, bool]] = [
            (["y"], "approve", False),
            (["n"], "reject", False),
            (["a"], "approve", True),
            (["c"], None, False),
            (["w", "c"], None, False),
        ]
        for answers, expected, should_auto_approve in cases:
            with (
                self.subTest(answers=answers),
                patch("ollama_agent.streaming.console_renderer.sys.stdin", new=SimpleNamespace(isatty=lambda: True)),
                patch("builtins.input", side_effect=iter(answers)),
            ):
                renderer, _ = self._make_renderer()
                runtime.auto_approved_tools.clear()
                result = await renderer.handle_interrupt(self._interrupt_event(), runtime)
                if expected is None:
                    self.assertIsNone(result)
                else:
                    assert result is not None
                    self.assertEqual(result[0]["type"], expected)
                self.assertEqual("execute" in runtime.auto_approved_tools, should_auto_approve)

    async def test_eof_at_prompt_returns_none(self) -> None:
        renderer, console = self._make_renderer()
        fake_stdin = SimpleNamespace(isatty=lambda: True)
        with (
            patch("ollama_agent.streaming.console_renderer.sys.stdin", new=fake_stdin),
            patch("builtins.input", side_effect=EOFError),
        ):
            result = await renderer.handle_interrupt(self._interrupt_event(), MagicMock())
        self.assertIsNone(result)
        self.assertIn("Cancelled", console.export_text())

    async def test_stream_agent_events_propagates_keyboard_interrupt(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(prompt: str, thread_id: str = "") -> AsyncGenerator[dict[str, Any], None]:
            raise KeyboardInterrupt
            yield  # pragma: no cover - makes this an async generator

        mock_runtime.run_streamed = fake_stream
        renderer = DummyRenderer()
        with self.assertRaises(KeyboardInterrupt):
            await stream_agent_events(mock_runtime, "p", renderer)
        self.assertTrue(renderer.closed)

    async def test_stream_agent_events_aborted_interrupt_returns_false(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(prompt: str, thread_id: str = "") -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "interrupt", "interrupts": [SimpleNamespace(value={"action_requests": []})]}

        mock_runtime.run_streamed = fake_stream
        renderer = DummyRenderer()
        with patch.object(renderer, "handle_interrupt", return_value=None):
            completed = await stream_agent_events(mock_runtime, "p", renderer)
        self.assertFalse(completed)

    async def test_stream_agent_events_propagates_renderer_exceptions(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(prompt: str, thread_id: str = "") -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "interrupt", "interrupts": [SimpleNamespace(value={"action_requests": []})]}

        mock_runtime.run_streamed = fake_stream

        class BrokenRenderer(DummyRenderer):
            async def handle_interrupt(self, event: dict[str, Any], runtime: Any) -> list[dict[str, Any]] | None:
                raise ValueError("boom")

        renderer = BrokenRenderer()
        with self.assertRaises(ValueError):
            await stream_agent_events(mock_runtime, "p", renderer)
        self.assertTrue(renderer.closed)


class TestExtractActionRequests(unittest.TestCase):
    """Tests for direct interrupt payload extraction."""

    @staticmethod
    def _event(action_requests: Any) -> dict[str, Any]:
        return {
            "type": "interrupt",
            "interrupts": [SimpleNamespace(value={"action_requests": action_requests})],
        }

    def test_valid_payload(self) -> None:
        requests = [{"name": "execute", "args": {"command": "ls"}}]
        self.assertEqual(extract_action_requests(self._event(requests)), requests)

    def test_valid_tuple_payload(self) -> None:
        requests = [{"name": "write_file", "args": {"file_path": "/test"}}]
        event = {
            "type": "interrupt",
            "interrupts": (SimpleNamespace(value={"action_requests": requests}),),
        }
        self.assertEqual(extract_action_requests(event), requests)

    def test_missing_interrupts_raises(self) -> None:
        with self.assertRaises(KeyError):
            extract_action_requests({"type": "interrupt"})

    def test_empty_interrupts_raises(self) -> None:
        with self.assertRaises(IndexError):
            extract_action_requests({"type": "interrupt", "interrupts": []})

    def test_missing_value_attribute_raises(self) -> None:
        event = {"type": "interrupt", "interrupts": [SimpleNamespace()]}
        with self.assertRaises(AttributeError):
            extract_action_requests(event)

    def test_missing_action_requests_raises(self) -> None:
        event = {"type": "interrupt", "interrupts": [SimpleNamespace(value={})]}
        with self.assertRaises(KeyError):
            extract_action_requests(event)


class TestBuildApprovalDecisions(unittest.TestCase):
    """Tests for build_approval_decisions helper."""

    def setUp(self) -> None:
        self.requests = [{"name": "execute", "args": {"command": "ls"}}]

    def test_approve(self) -> None:
        decisions = build_approval_decisions(self.requests, "approve")
        self.assertEqual(decisions, [{"type": "approve"}])

    def test_allow(self) -> None:
        runtime = SimpleNamespace(auto_approved_tools=set())
        decisions = build_approval_decisions(self.requests, "allow", runtime=runtime)
        self.assertEqual(decisions, [{"type": "approve"}])
        self.assertIn("execute", runtime.auto_approved_tools)

    def test_reject_default_message(self) -> None:
        decisions = build_approval_decisions(self.requests, "reject")
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["type"], "reject")
        self.assertIn("execute", decisions[0]["message"])

    def test_reject_custom_message(self) -> None:
        decisions = build_approval_decisions(self.requests, "reject", reject_message="Custom rejection")
        self.assertEqual(decisions, [{"type": "reject", "message": "Custom rejection"}])

    def test_unknown_action_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_approval_decisions(self.requests, "unknown")
