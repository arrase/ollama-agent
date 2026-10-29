from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import make_mock_repl, recording_console
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ollama_agent.interfaces.tui import OllamaAgentApp
from ollama_agent.interfaces.tui.commands import IMMEDIATE_COMMANDS, _is_immediate_command
from ollama_agent.interfaces.tui.completion import slash_completions
from ollama_agent.interfaces.tui.widgets import (
    AgentFooter,
    AgentResponse,
    PromptQueueWidget,
    ReplInput,
    SystemOutputWidget,
    UserMessage,
)
from ollama_agent.streaming.base import StreamingRenderer


class TestPromptQueueWidget(unittest.TestCase):
    """Unit tests for PromptQueueWidget rendering and update behaviors."""

    def test_can_focus_is_false(self) -> None:
        widget = PromptQueueWidget()
        assert not widget.can_focus

    def test_update_queue_empty_hides_widget(self) -> None:
        widget = PromptQueueWidget()
        widget.update_queue([])
        assert not widget.display

    def test_update_queue_renders_items(self) -> None:
        widget = PromptQueueWidget()
        items = ["Prompt alpha", "Prompt beta"]
        widget.update_queue(items)
        assert widget.display
        rendered = str(widget.render())
        assert "Queued (2)" in rendered
        assert "#1 Prompt alpha" in rendered
        assert "#2 Prompt beta" in rendered

    def test_update_queue_truncates_long_items_and_escapes(self) -> None:
        widget = PromptQueueWidget()
        long_prompt = "A" * 70
        widget.update_queue([long_prompt])
        assert widget.display
        rendered = str(widget.render())
        assert "..." in rendered

    def test_update_queue_more_than_three_items(self) -> None:
        widget = PromptQueueWidget()
        items = [f"Item {i}" for i in range(1, 6)]
        widget.update_queue(items)
        assert widget.display
        rendered = str(widget.render())
        assert "Queued (5)" in rendered
        assert "#1 Item 1" in rendered
        assert "#2 Item 2" in rendered
        assert "#3 Item 3" in rendered
        assert "#4 Item 4" not in rendered
        assert "+2 more" in rendered


class TestIsImmediateCommand(unittest.TestCase):
    """Unit tests for immediate vs stateful/queued slash command classification."""

    def test_immediate_commands_set(self) -> None:
        assert ("/exit", "*") in IMMEDIATE_COMMANDS
        assert ("/quit", "*") in IMMEDIATE_COMMANDS
        assert ("/queue", "*") in IMMEDIATE_COMMANDS
        assert ("/yolo", "*") in IMMEDIATE_COMMANDS
        assert ("/stealth", "*") in IMMEDIATE_COMMANDS
        assert ("/model", "list") in IMMEDIATE_COMMANDS
        assert ("/effort", "") in IMMEDIATE_COMMANDS
        assert ("/context", "") in IMMEDIATE_COMMANDS
        assert ("/session", "list") in IMMEDIATE_COMMANDS

    def test_immediate_commands_bypass_queue(self) -> None:
        immediate_samples = [
            "/exit",
            "/quit",
            "/queue",
            "/queue list",
            "/queue clear",
            "/queue rm 1",
            "/queue remove 2",
            "/queue delete 3",
            "/yolo",
            "/yolo on",
            "/yolo off",
            "/stealth",
            "/stealth on",
            "/stealth off",
            "/model",
            "/model list",
            "/effort",
            "/context",
            "/params",
            "/params list",
            "/session",
            "/session list",
            "/session search memory",
            "/session delete sess_123",
            "/session export sess_123",
            "/task",
            "/task list",
            "/task delete task_abc",
            "/skill",
            "/skill list",
            "/skill show my_skill",
            "/skill delete my_skill",
            "/rag",
            "/rag status",
            "/rag list",
            "/rag create test_db",
            "/mcp",
            "/mcp list",
            "/mcp status",
            "/agents",
            "/agents list",
        ]
        for cmd in immediate_samples:
            with self.subTest(cmd=cmd):
                assert _is_immediate_command(cmd)

    def test_stateful_commands_and_prompts_are_not_immediate(self) -> None:
        stateful_samples = [
            "Normal prompt message",
            "What is the time complexity of dijkstra?",
            "",
            "/model set qwen2.5-coder:32b",
            "/model llama3:latest",
            "/effort high",
            "/effort set low",
            "/context 8192",
            "/context set 32768",
            "/params set temperature 0.7",
            "/session new",
            "/session resume sess_abc",
            "/session switch sess_abc",
            "/clear",
            "/new",
            "/task create write_unit_tests",
            "/task run task_123",
            "/skill create custom_skill",
            "/rag delete test_db",
            "/rag load test_db",
            "/rag unload",
            "/rag add /path/to/doc.pdf",
            "/rag add /path/to/docs --dir",
            "/mcp reload",
        ]
        for cmd in stateful_samples:
            with self.subTest(cmd=cmd):
                assert not _is_immediate_command(cmd)


class TestPromptQueueFIFO(unittest.IsolatedAsyncioTestCase):
    """Unit tests for FIFO queuing of multiple messages and slash commands."""

    async def test_fifo_queuing_of_multiple_prompts(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")

            app._is_generating = True

            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Prompt 1"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Prompt 2"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Prompt 3"))

            assert len(app._prompt_queue) == 3
            assert app._prompt_queue[0] == "Prompt 1"
            assert app._prompt_queue[1] == "Prompt 2"
            assert app._prompt_queue[2] == "Prompt 3"

            assert footer._queued_count == 3
            assert "3 queued" in str(footer.render())

            assert len(list(chat_scroll.query(".system-message"))) == 0
            queue_widget = app.query_one(PromptQueueWidget)
            assert queue_widget.display
            assert "Prompt 1" in str(queue_widget.render())
            assert "Prompt 2" in str(queue_widget.render())
            assert "Prompt 3" in str(queue_widget.render())
            sys_out = app.query_one(SystemOutputWidget)
            assert not sys_out.display

    async def test_fifo_queuing_stateful_slash_commands(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)

            app._is_generating = True

            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/task run task_123"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/model set llama3:latest"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/session new"))

            assert len(app._prompt_queue) == 3
            assert app._prompt_queue[0] == "/task run task_123"
            assert app._prompt_queue[1] == "/model set llama3:latest"
            assert app._prompt_queue[2] == "/session new"
            assert footer._queued_count == 3

    async def test_fifo_queuing_mixed_prompts_and_commands(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)

            app._is_generating = True

            app.on_repl_input_submitted(ReplInput.Submitted(inp, "First prompt"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/task run task_123"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Second prompt"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/session resume sess_xyz"))

            queue_items = list(app._prompt_queue)
            assert queue_items == ["First prompt", "/task run task_123", "Second prompt", "/session resume sess_xyz"]

    async def test_empty_or_whitespace_input_not_queued(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            app._is_generating = True

            app.on_repl_input_submitted(ReplInput.Submitted(inp, ""))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "   \n\t  "))

            assert len(app._prompt_queue) == 0


class TestImmediateCommandExecution(unittest.IsolatedAsyncioTestCase):
    """Unit tests for immediate commands executed while queue is non-empty or generation is active."""

    async def test_immediate_command_bypasses_queue_when_generating(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            inp = app.query_one(ReplInput)

            app._is_generating = True
            active_worker = MagicMock()
            app._current_worker = active_worker

            with patch("ollama_agent.interfaces.tui.app.run_slash_command", new_callable=AsyncMock):
                app.on_repl_input_submitted(ReplInput.Submitted(inp, "/yolo"))
                await pilot.pause()

                # Immediate command must not be added to queue
                assert len(app._prompt_queue) == 0
                # Active worker is preserved while generating
                assert app._current_worker == active_worker

    async def test_immediate_command_while_queue_has_items(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)

            app._is_generating = True
            app._prompt_queue.append("Queued prompt 1")
            app._prompt_queue.append("Queued prompt 2")
            app._update_queue_ui()

            assert footer._queued_count == 2

            with patch("ollama_agent.interfaces.tui.app.run_slash_command", new_callable=AsyncMock):
                app.on_repl_input_submitted(ReplInput.Submitted(inp, "/queue list"))
                await pilot.pause()

                # Queue remains unmodified
                assert len(app._prompt_queue) == 2
                assert footer._queued_count == 2

    async def test_immediate_command_when_idle_sets_current_worker(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            inp = app.query_one(ReplInput)
            app._is_generating = False

            with patch("ollama_agent.interfaces.tui.app.run_slash_command", new_callable=AsyncMock):
                app.on_repl_input_submitted(ReplInput.Submitted(inp, "/yolo"))
                assert app._current_worker is not None
                await pilot.pause()


class TestQueueDrainingBehavior(unittest.IsolatedAsyncioTestCase):
    """Unit tests for queue draining when generation or slash commands complete."""

    async def test_process_next_in_queue_noop_when_empty(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            with (
                patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream,
                patch("ollama_agent.interfaces.tui.app.run_slash_command", new_callable=AsyncMock) as mock_slash,
            ):
                app._process_next_in_queue()
                mock_stream.assert_not_called()
                mock_slash.assert_not_called()

    async def test_process_next_in_queue_blocked_when_busy(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("Pending item")

            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                # Blocked when generating
                app._is_generating = True
                app._is_approval_pending = False
                app._process_next_in_queue()
                assert len(app._prompt_queue) == 1
                mock_stream.assert_not_called()

                # Blocked when approval pending
                app._is_generating = False
                app._is_approval_pending = True
                app._process_next_in_queue()
                assert len(app._prompt_queue) == 1
                mock_stream.assert_not_called()

    async def test_drains_prompt_item_and_updates_badge(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")

            app._prompt_queue.append("Drainable prompt")
            app._update_queue_ui()
            assert footer._queued_count == 1

            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                app._process_next_in_queue()
                await pilot.pause()

                assert len(app._prompt_queue) == 0
                assert footer._queued_count == 0
                assert "queued" not in str(footer.render())

                user_msgs = list(chat_scroll.query(UserMessage))
                agent_msgs = list(chat_scroll.query(AgentResponse))
                assert len(user_msgs) == 1
                assert len(agent_msgs) == 1

                mock_stream.assert_called_once()
                assert mock_stream.call_args[0][0] == "Drainable prompt"

    async def test_drains_slash_command_item(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            footer = app.query_one(AgentFooter)

            app._prompt_queue.append("/task run task_123")
            app._update_queue_ui()
            assert footer._queued_count == 1

            with patch("ollama_agent.interfaces.tui.app.run_slash_command", new_callable=AsyncMock) as mock_slash:
                app._process_next_in_queue()
                await pilot.pause()

                assert len(app._prompt_queue) == 0
                assert footer._queued_count == 0
                mock_slash.assert_called_once_with(app, "/task run task_123")

    async def test_sequential_queue_draining_flow(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            app._prompt_queue.append("Prompt 1")
            app._prompt_queue.append("/session new")
            app._prompt_queue.append("Prompt 2")
            app._update_queue_ui()

            streamed_prompts: list[str] = []

            async def fake_stream_events(
                _rt: object, prompt: str | Command[object], _renderer: object, *, auto_close: bool = True
            ) -> None:
                _ = auto_close
                if isinstance(prompt, str):
                    streamed_prompts.append(prompt)

            with patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=fake_stream_events):
                app._process_next_in_queue()
                await pilot.pause()

            assert streamed_prompts == ["Prompt 1", "Prompt 2"]
            assert len(app._prompt_queue) == 0


class TestUserCancellationWithQueue(unittest.IsolatedAsyncioTestCase):
    """Unit tests for Esc / Ctrl+C cancellation behavior with the prompt queue."""

    async def test_cancel_generation_clears_queue_and_cancels_worker(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")

            app._prompt_queue.append("Queued prompt 1")
            app._prompt_queue.append("Queued prompt 2")
            app._update_queue_ui()

            mock_worker = MagicMock()
            app._is_generating = True
            app._current_worker = mock_worker

            app.action_cancel_generation()

            # Worker cancelled AND queue cleared
            mock_worker.cancel.assert_called_once()
            assert len(app._prompt_queue) == 0
            assert footer._queued_count == 0
            assert "queued" not in str(footer.render())

            assert len(list(chat_scroll.query(".system-message"))) == 0
            sys_out = app.query_one(SystemOutputWidget)
            assert sys_out.display
            assert "Prompt queue cleared" in str(sys_out.render())

    async def test_cancel_generation_cancels_worker_and_approval(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")

            # Active generating worker
            mock_worker = MagicMock()
            app._is_generating = True
            app._current_worker = mock_worker

            app.action_cancel_generation()
            mock_worker.cancel.assert_called_once()

            # Approval pending cancellation
            app._is_generating = False
            app._is_approval_pending = True
            footer.set_approval(True)

            app.action_cancel_generation()
            assert not app._is_approval_pending
            assert "Approval required" not in str(footer.render())

            assert len(list(chat_scroll.query(".system-message"))) == 0
            sys_out = app.query_one(SystemOutputWidget)
            assert sys_out.display
            assert "Approval cancelled" in str(sys_out.render())

    async def test_action_cancel_or_quit_branches(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            with patch.object(app, "action_cancel_generation") as mock_cancel, patch.object(app, "exit") as mock_exit:
                # 1. Generating cancels generation, does not exit
                app._is_generating = True
                app.action_cancel_or_quit()
                mock_cancel.assert_called_once()
                mock_exit.assert_not_called()

                # 2. Approval pending cancels generation, does not exit
                mock_cancel.reset_mock()
                app._is_generating = False
                app._is_approval_pending = True
                app.action_cancel_or_quit()
                mock_cancel.assert_called_once()
                mock_exit.assert_not_called()

                # 3. Non-empty queue cancels generation, does not exit
                mock_cancel.reset_mock()
                app._is_approval_pending = False
                app._prompt_queue.append("Task")
                app.action_cancel_or_quit()
                mock_cancel.assert_called_once()
                mock_exit.assert_not_called()

                # 4. Idle with empty queue exits cleanly
                mock_cancel.reset_mock()
                app._prompt_queue.clear()
                app.action_cancel_or_quit()
                mock_cancel.assert_not_called()
                mock_exit.assert_called_once()

    async def test_run_stream_cancelled_error_clears_queue(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            chat_scroll = app.query_one("#chat-scroll")
            footer = app.query_one(AgentFooter)
            inp = app.query_one(ReplInput)

            app._prompt_queue.append("Stale queued prompt")
            app._update_queue_ui()

            agent_msg = AgentResponse()

            with (
                patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=asyncio.CancelledError()),
                pytest.raises(asyncio.CancelledError),
            ):
                await app._run_stream("Trigger prompt", chat_scroll, agent_msg)

            assert len(app._prompt_queue) == 0
            assert footer._queued_count == 0
            assert not inp.disabled
            assert not app._is_approval_pending

            assert len(list(chat_scroll.query(".system-message"))) == 0
            sys_out = app.query_one(SystemOutputWidget)
            assert sys_out.display
            assert "Execution interrupted by user" in str(sys_out.render())

    async def test_queue_clear_and_rm_do_not_interrupt_active_generation(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            mock_worker = MagicMock()
            app._is_generating = True
            app._current_worker = mock_worker

            app._prompt_queue.append("Prompt 1")
            app._prompt_queue.append("Prompt 2")
            app._prompt_queue.append("Prompt 3")
            app._update_queue_ui()

            # /queue rm 2 removes item 2 without stopping worker
            repl._handle_queue_cmd(["rm", "2"])
            assert len(app._prompt_queue) == 2
            assert app._prompt_queue[0] == "Prompt 1"
            assert app._prompt_queue[1] == "Prompt 3"
            mock_worker.cancel.assert_not_called()
            assert app._is_generating

            # /queue clear clears queue without stopping worker
            repl._handle_queue_cmd(["clear"])
            assert len(app._prompt_queue) == 0
            mock_worker.cancel.assert_not_called()
            assert app._is_generating


class TestToolApprovalModalAndQueue(unittest.IsolatedAsyncioTestCase):
    """Unit tests for tool approval interaction with prompt queue and input widget."""

    async def test_approval_modal_keeps_repl_input_enabled(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")
            agent_msg = AgentResponse()
            await chat_scroll.mount(agent_msg)

            mock_interrupt = MagicMock()
            mock_interrupt.value = {"action_requests": [{"name": "read_file", "args": {"path": "main.py"}}]}

            async def fake_stream(
                rt: object, _prompt: object, renderer: StreamingRenderer, *, auto_close: bool = True
            ) -> None:
                _ = auto_close
                await renderer.handle_interrupt(
                    {"interrupts": [mock_interrupt]},
                    rt,
                )

            with patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=fake_stream):
                await app._run_stream("Read file prompt", chat_scroll, agent_msg)

            assert app._is_approval_pending
            assert not inp.disabled
            assert "Approval required" in str(footer.render())

    async def test_queuing_prompts_during_approval_pending(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)

            app._is_approval_pending = True
            footer.set_approval(True)

            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Prompt during approval"))

            assert len(app._prompt_queue) == 1
            assert app._prompt_queue[0] == "Prompt during approval"
            assert footer._queued_count == 1
            assert "1 queued" in str(footer.render())

    async def test_approval_decision_resumes_stream_and_drains_queue(self) -> None:
        repl = make_mock_repl()
        runtime = repl.runtime
        mock_state = MagicMock()
        mock_state.interrupts = []
        runtime.graph.aget_state = AsyncMock(return_value=mock_state)

        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")
            agent_msg = AgentResponse()
            chat_scroll.mount(agent_msg)

            app._is_approval_pending = True
            app._prompt_queue.append("Subsequent queued prompt")
            app._update_queue_ui()

            processed_calls: list[object] = []

            async def fake_stream_events(
                _rt: object, prompt: str | Command[object], _renderer: object, *, auto_close: bool = True
            ) -> None:
                _ = auto_close
                processed_calls.append(prompt)

            with patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=fake_stream_events):
                await app._handle_approval_decision([{"type": "approve"}], chat_scroll, agent_msg)
                await pilot.pause()

            assert not app._is_approval_pending
            assert len(processed_calls) == 2
            assert isinstance(processed_calls[0], Command)
            assert processed_calls[1] == "Subsequent queued prompt"
            assert len(app._prompt_queue) == 0


class TestQueueSlashCommandsHandler(unittest.IsolatedAsyncioTestCase):
    """Unit tests for /queue and /queue clear command handler."""

    def test_queue_list_when_empty(self) -> None:
        repl = make_mock_repl()
        repl.app = None
        repl._handle_queue_cmd([])
        output = repl.console.export_text()
        assert "Prompt queue is empty" in output

    async def test_queue_list_with_items(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("Prompt alpha")
            app._prompt_queue.append("Prompt beta")

            repl.console = recording_console()
            repl._handle_queue_cmd(["list"])
            output = repl.console.export_text()

            assert "Queued prompts (2)" in output
            assert "#1 Prompt alpha" in output
            assert "#2 Prompt beta" in output

    async def test_queue_clear_subcommand(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("Item 1")
            app._prompt_queue.append("Item 2")
            app._prompt_queue.append("Item 3")
            app._update_queue_ui()

            repl.console = recording_console()
            repl._handle_queue_cmd(["clear"])
            output = repl.console.export_text()

            assert "Prompt queue cleared (3 removed)" in output
            assert len(app._prompt_queue) == 0
            footer = app.query_one(AgentFooter)
            assert footer._queued_count == 0

    async def test_queue_rm_success(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("Item 1")
            app._prompt_queue.append("Item 2")
            app._prompt_queue.append("Item 3")
            app._update_queue_ui()

            repl.console = recording_console()
            repl._handle_queue_cmd(["rm", "2"])
            output = repl.console.export_text()

            assert "Removed #2 from prompt queue: Item 2" in output
            assert len(app._prompt_queue) == 2
            assert app._prompt_queue[0] == "Item 1"
            assert app._prompt_queue[1] == "Item 3"
            footer = app.query_one(AgentFooter)
            assert footer._queued_count == 2

    async def test_queue_rm_with_hash_prefix(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("Item 1")
            app._prompt_queue.append("Item 2")
            app._update_queue_ui()

            repl.console = recording_console()
            repl._handle_queue_cmd(["rm", "#1"])
            output = repl.console.export_text()

            assert "Removed #1 from prompt queue: Item 1" in output
            assert len(app._prompt_queue) == 1
            assert app._prompt_queue[0] == "Item 2"

    async def test_queue_rm_aliases_remove_and_delete(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("Alpha")
            app._prompt_queue.append("Beta")
            app._update_queue_ui()

            repl.console = recording_console()
            repl._handle_queue_cmd(["remove", "1"])
            assert "Removed #1 from prompt queue: Alpha" in repl.console.export_text()

            repl.console = recording_console()
            repl._handle_queue_cmd(["delete", "1"])
            assert "Removed #1 from prompt queue: Beta" in repl.console.export_text()
            assert len(app._prompt_queue) == 0

    def test_queue_rm_missing_position(self) -> None:
        repl = make_mock_repl()
        repl.app = None
        repl._handle_queue_cmd(["rm"])
        output = repl.console.export_text()
        assert "Usage: /queue rm <position>" in output

    def test_queue_rm_when_empty(self) -> None:
        repl = make_mock_repl()
        repl.app = None
        repl._handle_queue_cmd(["rm", "1"])
        output = repl.console.export_text()
        assert "Prompt queue is empty." in output

    async def test_queue_rm_invalid_position_non_digit(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("Item 1")
            repl.console = recording_console()
            repl._handle_queue_cmd(["rm", "abc"])
            output = repl.console.export_text()
            assert "Invalid queue position 'abc'. Usage: /queue rm <position>" in output

    async def test_queue_rm_out_of_range(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("Item 1")
            app._prompt_queue.append("Item 2")
            app._update_queue_ui()

            repl.console = recording_console()
            repl._handle_queue_cmd(["rm", "0"])
            assert "Queue position 0 out of range (queue has 2 items)." in repl.console.export_text()

            repl.console = recording_console()
            repl._handle_queue_cmd(["rm", "5"])
            assert "Queue position 5 out of range (queue has 2 items)." in repl.console.export_text()

    async def test_queue_rm_autocomplete(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append("First queued prompt")
            app._prompt_queue.append("Second queued prompt")

            # Level 2 completions for /queue rm
            comps = slash_completions("/queue rm ", app.repl, list(app._prompt_queue))
            assert len(comps) == 2
            assert comps[0][0] == "/queue rm 1"
            assert comps[1][0] == "/queue rm 2"
            assert "#1" in comps[0][1].plain
            assert "First queued prompt" in comps[0][1].plain

            # Filtered completion
            comps_filtered = slash_completions("/queue remove 2", app.repl, list(app._prompt_queue))
            assert len(comps_filtered) == 1
            assert comps_filtered[0][0] == "/queue remove 2"

            # Hash-prefixed filter
            comps_hash = slash_completions("/queue delete #1", app.repl, list(app._prompt_queue))
            assert len(comps_hash) == 1
            assert comps_hash[0][0] == "/queue delete 1"

    def test_queue_invalid_subcommand(self) -> None:
        repl = make_mock_repl()
        repl.app = None
        repl._handle_queue_cmd(["unknown_cmd"])
        output = repl.console.export_text()
        assert "Unknown queue subcommand 'unknown_cmd'" in output

    def test_queue_handler_when_app_is_none(self) -> None:
        repl = make_mock_repl()
        repl.app = None

        repl._handle_queue_cmd(["list"])
        assert "Prompt queue is empty" in repl.console.export_text()

        repl.console = recording_console()
        repl._handle_queue_cmd(["clear"])
        assert "Prompt queue cleared (0 removed)" in repl.console.export_text()


class TestSessionTransitionsWithQueue(unittest.IsolatedAsyncioTestCase):
    """Unit tests for session creation, switching, and resuming with queued items."""

    async def test_session_new_drains_subsequent_queue(self) -> None:
        repl = make_mock_repl()
        runtime = repl.runtime
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            app._prompt_queue.append("/session new")
            app._prompt_queue.append("First message in fresh session")
            app._update_queue_ui()

            streamed: list[str] = []

            async def fake_stream_events(
                _rt: object, prompt: str | Command[object], _renderer: object, *, auto_close: bool = True
            ) -> None:
                _ = auto_close
                if isinstance(prompt, str):
                    streamed.append(prompt)

            with (
                patch("ollama_agent.interfaces.tui.repl.new_session", return_value="sess_new_999"),
                patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=fake_stream_events),
            ):
                app._process_next_in_queue()
                await pilot.pause()

                assert runtime.thread_id == "sess_new_999"
                assert len(app._prompt_queue) == 0
                assert streamed == ["First message in fresh session"]

    async def test_session_resume_success_drains_subsequent_queue(self) -> None:
        repl = make_mock_repl()
        runtime = repl.runtime
        runtime.get_thread_messages = AsyncMock(
            return_value=[
                HumanMessage(content="Hello previous session"),
                AIMessage(content="Hello! How can I help you today?"),
            ]
        )
        runtime.count_effective_tokens = AsyncMock(return_value=128)

        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")

            app._prompt_queue.append("/session resume sess_target_123")
            app._prompt_queue.append("Followup question")
            app._update_queue_ui()

            streamed: list[str] = []

            async def fake_stream_events(
                _rt: object, prompt: str | Command[object], _renderer: object, *, auto_close: bool = True
            ) -> None:
                _ = auto_close
                if isinstance(prompt, str):
                    streamed.append(prompt)

            with (
                patch("ollama_agent.interfaces.tui.commands.resume_session", return_value="sess_target_123"),
                patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=fake_stream_events),
            ):
                app._process_next_in_queue()
                await pilot.pause()

                assert runtime.thread_id == "sess_target_123"
                assert runtime.last_context_tokens == 128

                user_msgs = list(chat_scroll.query(UserMessage))
                agent_msgs = list(chat_scroll.query(AgentResponse))
                assert len(user_msgs) == 2
                assert len(agent_msgs) == 2

                assert len(app._prompt_queue) == 0
                assert streamed == ["Followup question"]

    async def test_session_resume_not_found_drains_subsequent_queue(self) -> None:
        repl = make_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")

            app._prompt_queue.append("/session resume nonexistent_id")
            app._prompt_queue.append("Subsequent message")

            streamed: list[str] = []

            async def fake_stream_events(
                _rt: object, prompt: str | Command[object], _renderer: object, *, auto_close: bool = True
            ) -> None:
                _ = auto_close
                if isinstance(prompt, str):
                    streamed.append(prompt)

            with (
                patch("ollama_agent.interfaces.tui.commands.resume_session", return_value=None),
                patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=fake_stream_events),
            ):
                app._process_next_in_queue()
                await pilot.pause()

                assert len(list(chat_scroll.query(".system-message"))) == 0
                assert len(app._prompt_queue) == 0
                assert streamed == ["Subsequent message"]

    async def test_clear_and_new_aliases_drain_subsequent_queue(self) -> None:
        for alias in ("/clear", "/new"):
            with self.subTest(alias=alias):
                repl = make_mock_repl()
                runtime = repl.runtime
                app = OllamaAgentApp(repl)
                async with app.run_test() as pilot:
                    app._prompt_queue.append(alias)
                    app._prompt_queue.append("Next action")

                    streamed: list[str] = []

                    async def fake_stream_events(
                        _rt: object,
                        prompt: str | Command[object],
                        _renderer: object,
                        *,
                        auto_close: bool = True,
                        streamed: list[str] = streamed,
                    ) -> None:
                        _ = auto_close
                        if isinstance(prompt, str):
                            streamed.append(prompt)

                    with (
                        patch("ollama_agent.interfaces.tui.repl.new_session", return_value="sess_alias_001"),
                        patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=fake_stream_events),
                    ):
                        app._process_next_in_queue()
                        await pilot.pause()

                        assert runtime.thread_id == "sess_alias_001"
                        assert len(app._prompt_queue) == 0
                        assert streamed == ["Next action"]
