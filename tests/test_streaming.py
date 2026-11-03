from __future__ import annotations

import io
import unittest
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
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

    def on_tool_output(self, event: dict[str, Any]) -> None:
        pass

    def on_warning(self, event: dict[str, Any]) -> None:
        self.warnings.append(event)

    def on_error(self, event: dict[str, Any]) -> None:
        self.errors.append(event)

    async def handle_interrupt(self, event: dict[str, Any], runtime: object) -> list[dict[str, Any]] | None:
        _ = event, runtime
        return None

    def close(self) -> None:
        self.closed = True


class TestStreamingSystem(unittest.IsolatedAsyncioTestCase):
    """Unit tests for streaming base classes, console renderer, and event pipeline."""

    def test_base_streaming_renderer_dispatch(self) -> None:
        renderer = DummyRenderer()
        renderer.on_event({"type": "text_delta", "content": "hello"})
        assert len(renderer.events) == 1

    def test_console_streaming_renderer_skips_unknown_type(self) -> None:
        renderer = DummyRenderer()
        renderer.on_event({"type": "agent_update", "content": "state"})
        # The event reaches DummyRenderer.on_event which always appends,
        # but no type-specific handler (on_agent_update) was called — verify
        # it didn't crash and no warnings/errors were recorded.
        assert len(renderer.warnings) == 0
        assert len(renderer.errors) == 0

    def test_console_streaming_renderer_reasoning_is_pure_delta(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        renderer = ConsoleStreamingRenderer(console=console)

        renderer.on_event({"type": "reasoning_delta", "content": "the"})
        renderer.on_event({"type": "reasoning_delta", "content": "the"})
        renderer.close()

        out = console.export_text()
        assert "thethe" in out

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
        assert "Hello world" in out
        assert "Thinking step" in out
        assert "Calling tool" in out
        assert "Tool output received" in out
        assert "Network failure" in out
        assert "Slow connection" in out

    def test_console_streaming_renderer_agent_prefix(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        renderer = ConsoleStreamingRenderer(console=console)
        # The prefix is escaped so an agent name carrying markup cannot break printing.
        assert renderer._agent_prefix({"agent_name": "worker"}) == "\\[worker] "
        assert renderer._agent_prefix({"agent_name": "a[/b]c"}) == "[a\\[/b]c] "
        assert renderer._agent_prefix({"agent_name": ""}) == ""
        assert renderer._agent_prefix({"agent_name": None}) == ""
        assert renderer._agent_prefix({}) == ""

    def test_reasoning_delta_with_closing_tag_does_not_crash(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        renderer = ConsoleStreamingRenderer(console=console)
        renderer.on_event({"type": "reasoning_delta", "content": "check [/bold] tag\nand [red]unclosed"})
        renderer.close()
        out = console.export_text()
        assert "[/bold]" in out
        assert "[red]unclosed" in out

    def test_error_and_warning_content_does_not_crash(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        renderer = ConsoleStreamingRenderer(console=console)
        renderer.on_event({"type": "error", "content": "bad [/red] markup [bold"})
        renderer.on_event({"type": "warning", "content": "warn [/yellow] markup [italic"})
        renderer.close()
        out = console.export_text()
        assert "[/red]" in out
        assert "[/yellow]" in out

    def test_tool_call_with_markup_in_name_does_not_crash(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        renderer = ConsoleStreamingRenderer(console=console)
        renderer.on_event({"type": "tool_call", "name": "tool[/yellow][bold]", "agent_name": "sub[/yellow]"})
        renderer.close()
        out = console.export_text()
        assert "tool[/yellow][bold]" in out

    async def test_stream_agent_events_pipeline(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(_prompt: str) -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "text_delta", "content": "Hello"}
            yield {"type": "tool_call", "name": "search"}
            yield {"type": "text_delta", "content": "World"}

        mock_runtime.run_streamed = fake_stream

        renderer = DummyRenderer()
        await stream_agent_events(mock_runtime, "test prompt", renderer)

        assert renderer.closed
        types = [e["type"] for e in renderer.events]
        assert types == ["text_delta", "tool_call", "text_delta"]

    async def test_stream_agent_events_returns_completed_true(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(_prompt: str) -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "text_delta", "content": "Hi"}

        mock_runtime.run_streamed = fake_stream
        renderer = DummyRenderer()
        completed = await stream_agent_events(mock_runtime, "p", renderer)
        assert completed
        assert renderer.closed


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
        assert result is None
        assert "--yolo" in console.export_text()

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
                    assert result is None
                else:
                    assert result is not None
                    assert result[0]["type"] == expected
                assert ("execute" in runtime.auto_approved_tools) == should_auto_approve

    async def test_eof_at_prompt_returns_none(self) -> None:
        renderer, console = self._make_renderer()
        fake_stdin = SimpleNamespace(isatty=lambda: True)
        with (
            patch("ollama_agent.streaming.console_renderer.sys.stdin", new=fake_stdin),
            patch("builtins.input", side_effect=EOFError),
        ):
            result = await renderer.handle_interrupt(self._interrupt_event(), MagicMock())
        assert result is None
        assert "Cancelled" in console.export_text()

    async def test_keyboard_interrupt_at_prompt_propagates(self) -> None:
        """Ctrl-C on the prompt propagates KeyboardInterrupt to the caller."""
        renderer, _console = self._make_renderer()
        fake_stdin = SimpleNamespace(isatty=lambda: True)
        event = self._interrupt_event()
        runtime = MagicMock()
        with (
            patch("ollama_agent.streaming.console_renderer.sys.stdin", new=fake_stdin),
            patch("builtins.input", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            await renderer.handle_interrupt(event, runtime)

    async def test_stream_agent_events_propagates_keyboard_interrupt(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(_prompt: str) -> AsyncGenerator[dict[str, Any], None]:
            for _ in ():
                yield {}
            raise KeyboardInterrupt

        mock_runtime.run_streamed = fake_stream
        renderer = DummyRenderer()
        with pytest.raises(KeyboardInterrupt):
            await stream_agent_events(mock_runtime, "p", renderer)
        assert renderer.closed

    async def test_stream_agent_events_aborted_interrupt_returns_false(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(_prompt: str) -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "interrupt", "interrupts": [SimpleNamespace(value={"action_requests": []})]}

        mock_runtime.run_streamed = fake_stream
        renderer = DummyRenderer()
        with patch.object(renderer, "handle_interrupt", return_value=None):
            completed = await stream_agent_events(mock_runtime, "p", renderer)
        assert not completed

    async def test_stream_agent_events_propagates_renderer_exceptions(self) -> None:
        mock_runtime = MagicMock()

        async def fake_stream(_prompt: str) -> AsyncGenerator[dict[str, Any], None]:
            yield {"type": "interrupt", "interrupts": [SimpleNamespace(value={"action_requests": []})]}

        mock_runtime.run_streamed = fake_stream

        class BrokenRenderer(DummyRenderer):
            async def handle_interrupt(self, _event: dict[str, Any], _runtime: object) -> list[dict[str, Any]] | None:
                raise ValueError("boom")

        renderer = BrokenRenderer()
        with pytest.raises(ValueError):
            await stream_agent_events(mock_runtime, "p", renderer)
        assert renderer.closed


class TestExtractActionRequests(unittest.TestCase):
    """Tests for direct interrupt payload extraction."""

    @staticmethod
    def _event(action_requests: object) -> dict[str, Any]:
        return {
            "type": "interrupt",
            "interrupts": [SimpleNamespace(value={"action_requests": action_requests})],
        }

    def test_valid_payload(self) -> None:
        requests = [{"name": "execute", "args": {"command": "ls"}}]
        assert extract_action_requests(self._event(requests)) == requests

    def test_valid_tuple_payload(self) -> None:
        requests = [{"name": "write_file", "args": {"file_path": "/test"}}]
        event = {
            "type": "interrupt",
            "interrupts": (SimpleNamespace(value={"action_requests": requests}),),
        }
        assert extract_action_requests(event) == requests

    def test_multiple_interrupts_concatenates_requests(self) -> None:
        req1 = {"name": "execute", "args": {"command": "ls"}}
        req2 = {"name": "write_file", "args": {"file_path": "/test"}}
        event = {
            "type": "interrupt",
            "interrupts": [
                SimpleNamespace(value={"action_requests": [req1]}),
                SimpleNamespace(value={"action_requests": [req2]}),
            ],
        }
        assert extract_action_requests(event) == [req1, req2]

    def test_empty_interrupts_returns_empty_list(self) -> None:
        assert extract_action_requests({"type": "interrupt", "interrupts": []}) == []


class TestBuildApprovalDecisions(unittest.TestCase):
    """Tests for build_approval_decisions helper."""

    def setUp(self) -> None:
        self.requests = [{"name": "execute", "args": {"command": "ls"}}]

    def test_approve(self) -> None:
        decisions = build_approval_decisions(self.requests, "approve")
        assert decisions == [{"type": "approve"}]

    def test_allow(self) -> None:
        runtime: Any = SimpleNamespace(auto_approved_tools=set())
        decisions = build_approval_decisions(self.requests, "allow", runtime=runtime)
        assert decisions == [{"type": "approve"}]
        assert "execute" in runtime.auto_approved_tools

    def test_reject_default_message(self) -> None:
        decisions = build_approval_decisions(self.requests, "reject")
        assert len(decisions) == 1
        assert decisions[0]["type"] == "reject"
        assert "execute" in decisions[0]["message"]

    def test_unknown_action_raises(self) -> None:
        with pytest.raises(ValueError):
            build_approval_decisions(self.requests, "unknown")
