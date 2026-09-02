from __future__ import annotations

import asyncio
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ollama_agent.interfaces.repl import (
    OllamaAgentApp,
    OllamaREPL,
    QueuedItem,
    _is_immediate_command,
)
from ollama_agent.interfaces.tui_components import (
    AgentFooter,
    AgentResponse,
    PromptQueueWidget,
    ReplInput,
    SystemMessage,
    SystemOutputWidget,
    UserMessage,
)


def _create_mock_repl() -> tuple[OllamaREPL, MagicMock]:
    """Create a mock OllamaREPL instance with standard mock runtime."""
    runtime = MagicMock()
    runtime.settings.model.name = "qwen2.5-coder:32b"
    runtime.settings.model.reasoning_effort = "high"
    runtime.settings.model.context_window = 16384
    runtime.settings.runtime.collapse_thinking = True
    runtime.yolo_mode = False
    runtime.thread_id = "session_001"
    runtime.last_context_tokens = 512
    runtime.auto_approved_tools = set()
    runtime.get_thread_messages = AsyncMock(return_value=[])
    runtime.count_effective_tokens = AsyncMock(return_value=512)
    runtime.reload = AsyncMock()

    mock_state = MagicMock()
    mock_state.interrupts = []
    runtime.graph.aget_state = AsyncMock(return_value=mock_state)

    repl = OllamaREPL(runtime=runtime)
    repl.console = Console(file=io.StringIO())
    return repl, runtime


class TestQueuedItem(unittest.TestCase):
    """Unit tests for the QueuedItem data structure."""

    def test_queued_item_creation(self) -> None:
        item = QueuedItem(text="Explain quicksort in Python")
        self.assertEqual(item.text, "Explain quicksort in Python")


class TestPromptQueueWidget(unittest.TestCase):
    """Unit tests for PromptQueueWidget rendering and update behaviors."""

    def test_can_focus_is_false(self) -> None:
        widget = PromptQueueWidget()
        self.assertFalse(widget.can_focus)

    def test_update_queue_empty_hides_widget(self) -> None:
        widget = PromptQueueWidget()
        widget.update_queue([])
        self.assertFalse(widget.display)

    def test_update_queue_renders_items(self) -> None:
        widget = PromptQueueWidget()
        items = [QueuedItem("Prompt alpha"), QueuedItem("Prompt beta")]
        widget.update_queue(items)
        self.assertTrue(widget.display)
        rendered = str(widget.render())
        self.assertIn("Queued (2)", rendered)
        self.assertIn("#1 Prompt alpha", rendered)
        self.assertIn("#2 Prompt beta", rendered)

    def test_update_queue_truncates_long_items_and_escapes(self) -> None:
        widget = PromptQueueWidget()
        long_prompt = "A" * 70
        widget.update_queue([QueuedItem(long_prompt)])
        self.assertTrue(widget.display)
        rendered = str(widget.render())
        self.assertIn("...", rendered)

    def test_update_queue_more_than_three_items(self) -> None:
        widget = PromptQueueWidget()
        items = [QueuedItem(f"Item {i}") for i in range(1, 6)]
        widget.update_queue(items)
        self.assertTrue(widget.display)
        rendered = str(widget.render())
        self.assertIn("Queued (5)", rendered)
        self.assertIn("#1 Item 1", rendered)
        self.assertIn("#2 Item 2", rendered)
        self.assertIn("#3 Item 3", rendered)
        self.assertNotIn("#4 Item 4", rendered)
        self.assertIn("+2 more", rendered)


class TestIsImmediateCommand(unittest.TestCase):
    """Unit tests for immediate vs stateful/queued slash command classification."""

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
            "/rag delete test_db",
            "/rag load test_db",
            "/rag unload",
            "/mcp",
            "/mcp list",
            "/mcp status",
            "/agents",
            "/agents list",
        ]
        for cmd in immediate_samples:
            with self.subTest(cmd=cmd):
                self.assertTrue(_is_immediate_command(cmd))

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
            "/rag add /path/to/doc.pdf",
            "/rag add /path/to/docs --dir",
            "/mcp reload",
        ]
        for cmd in stateful_samples:
            with self.subTest(cmd=cmd):
                self.assertFalse(_is_immediate_command(cmd))


class TestPromptQueueFIFO(unittest.IsolatedAsyncioTestCase):
    """Unit tests for FIFO queuing of multiple messages and slash commands."""

    async def test_fifo_queuing_of_multiple_prompts(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")

            app._is_generating = True

            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Prompt 1"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Prompt 2"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Prompt 3"))

            self.assertEqual(len(app._prompt_queue), 3)
            self.assertEqual(app._prompt_queue[0].text, "Prompt 1")
            self.assertEqual(app._prompt_queue[1].text, "Prompt 2")
            self.assertEqual(app._prompt_queue[2].text, "Prompt 3")

            self.assertEqual(footer._queued_count, 3)
            self.assertIn("3 queued", str(footer.render()))

            self.assertEqual(len(list(chat_scroll.query(SystemMessage))), 0)
            queue_widget = app.query_one(PromptQueueWidget)
            self.assertTrue(queue_widget.display)
            self.assertIn("Prompt 1", str(queue_widget.render()))
            self.assertIn("Prompt 2", str(queue_widget.render()))
            self.assertIn("Prompt 3", str(queue_widget.render()))
            sys_out = app.query_one(SystemOutputWidget)
            self.assertFalse(sys_out.display)

    async def test_fifo_queuing_stateful_slash_commands(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)

            app._is_generating = True

            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/task run task_123"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/model set llama3:latest"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/session new"))

            self.assertEqual(len(app._prompt_queue), 3)
            self.assertEqual(app._prompt_queue[0].text, "/task run task_123")
            self.assertEqual(app._prompt_queue[1].text, "/model set llama3:latest")
            self.assertEqual(app._prompt_queue[2].text, "/session new")
            self.assertEqual(footer._queued_count, 3)

    async def test_fifo_queuing_mixed_prompts_and_commands(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)

            app._is_generating = True

            app.on_repl_input_submitted(ReplInput.Submitted(inp, "First prompt"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/task run task_123"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Second prompt"))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "/session resume sess_xyz"))

            queue_items = [item.text for item in app._prompt_queue]
            self.assertEqual(
                queue_items,
                ["First prompt", "/task run task_123", "Second prompt", "/session resume sess_xyz"],
            )

    async def test_empty_or_whitespace_input_not_queued(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            app._is_generating = True

            app.on_repl_input_submitted(ReplInput.Submitted(inp, ""))
            app.on_repl_input_submitted(ReplInput.Submitted(inp, "   \n\t  "))

            self.assertEqual(len(app._prompt_queue), 0)


class TestImmediateCommandExecution(unittest.IsolatedAsyncioTestCase):
    """Unit tests for immediate commands executed while queue is non-empty or generation is active."""

    async def test_immediate_command_bypasses_queue_when_generating(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)

            app._is_generating = True
            active_worker = MagicMock()
            app._current_worker = active_worker

            with patch.object(app, "_run_slash_command", new_callable=AsyncMock):
                app.on_repl_input_submitted(ReplInput.Submitted(inp, "/yolo"))

                # Immediate command must not be added to queue
                self.assertEqual(len(app._prompt_queue), 0)
                # Active worker is preserved while generating
                self.assertEqual(app._current_worker, active_worker)

    async def test_immediate_command_while_queue_has_items(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)

            app._is_generating = True
            app._prompt_queue.append(QueuedItem("Queued prompt 1"))
            app._prompt_queue.append(QueuedItem("Queued prompt 2"))
            app._update_queue_ui()

            self.assertEqual(footer._queued_count, 2)

            with patch.object(app, "_run_slash_command", new_callable=AsyncMock):
                app.on_repl_input_submitted(ReplInput.Submitted(inp, "/queue list"))

                # Queue remains unmodified
                self.assertEqual(len(app._prompt_queue), 2)
                self.assertEqual(footer._queued_count, 2)

    async def test_immediate_command_when_idle_sets_current_worker(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            app._is_generating = False

            with patch.object(app, "_run_slash_command", new_callable=AsyncMock):
                app.on_repl_input_submitted(ReplInput.Submitted(inp, "/yolo"))
                self.assertIsNotNone(app._current_worker)


class TestQueueDrainingBehavior(unittest.IsolatedAsyncioTestCase):
    """Unit tests for queue draining when generation or slash commands complete."""

    async def test_process_next_in_queue_noop_when_empty(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            with (
                patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream,
                patch.object(app, "_run_slash_command", new_callable=AsyncMock) as mock_slash,
            ):
                app._process_next_in_queue()
                mock_stream.assert_not_called()
                mock_slash.assert_not_called()

    async def test_process_next_in_queue_blocked_when_busy(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("Pending item"))

            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                # Blocked when generating
                app._is_generating = True
                app._is_approval_pending = False
                app._process_next_in_queue()
                self.assertEqual(len(app._prompt_queue), 1)
                mock_stream.assert_not_called()

                # Blocked when approval pending
                app._is_generating = False
                app._is_approval_pending = True
                app._process_next_in_queue()
                self.assertEqual(len(app._prompt_queue), 1)
                mock_stream.assert_not_called()

    async def test_drains_prompt_item_and_updates_badge(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")

            app._prompt_queue.append(QueuedItem("Drainable prompt"))
            app._update_queue_ui()
            self.assertEqual(footer._queued_count, 1)

            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                app._process_next_in_queue()

                self.assertEqual(len(app._prompt_queue), 0)
                self.assertEqual(footer._queued_count, 0)
                self.assertNotIn("queued", str(footer.render()))

                user_msgs = list(chat_scroll.query(UserMessage))
                agent_msgs = list(chat_scroll.query(AgentResponse))
                self.assertEqual(len(user_msgs), 1)
                self.assertEqual(len(agent_msgs), 1)

                mock_stream.assert_called_once()
                self.assertEqual(mock_stream.call_args[0][0], "Drainable prompt")

    async def test_drains_slash_command_item(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            footer = app.query_one(AgentFooter)

            app._prompt_queue.append(QueuedItem("/task run task_123"))
            app._update_queue_ui()
            self.assertEqual(footer._queued_count, 1)

            with patch.object(app, "_run_slash_command", new_callable=AsyncMock) as mock_slash:
                app._process_next_in_queue()

                self.assertEqual(len(app._prompt_queue), 0)
                self.assertEqual(footer._queued_count, 0)
                mock_slash.assert_called_once_with("/task run task_123")

    async def test_sequential_queue_draining_flow(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            app._prompt_queue.append(QueuedItem("Prompt 1"))
            app._prompt_queue.append(QueuedItem("/session new"))
            app._prompt_queue.append(QueuedItem("Prompt 2"))
            app._update_queue_ui()

            streamed_prompts: list[str] = []

            async def fake_stream_events(
                rt: object, prompt: str | Command[object], renderer: object, auto_close: bool = True
            ) -> None:
                if isinstance(prompt, str):
                    streamed_prompts.append(prompt)

            with patch("ollama_agent.interfaces.repl.stream_agent_events", side_effect=fake_stream_events):
                app._process_next_in_queue()
                await pilot.pause()

            self.assertEqual(streamed_prompts, ["Prompt 1", "Prompt 2"])
            self.assertEqual(len(app._prompt_queue), 0)


class TestUserCancellationWithQueue(unittest.IsolatedAsyncioTestCase):
    """Unit tests for Esc / Ctrl+C cancellation behavior with the prompt queue."""

    async def test_cancel_generation_clears_queue_and_cancels_worker(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")

            app._prompt_queue.append(QueuedItem("Queued prompt 1"))
            app._prompt_queue.append(QueuedItem("Queued prompt 2"))
            app._update_queue_ui()

            mock_worker = MagicMock()
            app._is_generating = True
            app._current_worker = mock_worker

            app.action_cancel_generation()

            # Worker cancelled AND queue cleared
            mock_worker.cancel.assert_called_once()
            self.assertEqual(len(app._prompt_queue), 0)
            self.assertEqual(footer._queued_count, 0)
            self.assertNotIn("queued", str(footer.render()))

            self.assertEqual(len(list(chat_scroll.query(SystemMessage))), 0)
            sys_out = app.query_one(SystemOutputWidget)
            self.assertTrue(sys_out.display)
            self.assertIn("Prompt queue cleared", str(sys_out.render()))

    async def test_cancel_generation_cancels_worker_and_approval(self) -> None:
        repl, _ = _create_mock_repl()
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
            self.assertFalse(app._is_approval_pending)
            self.assertNotIn("Approval required", str(footer.render()))

            self.assertEqual(len(list(chat_scroll.query(SystemMessage))), 0)
            sys_out = app.query_one(SystemOutputWidget)
            self.assertTrue(sys_out.display)
            self.assertIn("Approval cancelled", str(sys_out.render()))

    async def test_action_cancel_or_quit_branches(self) -> None:
        repl, _ = _create_mock_repl()
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
                app._prompt_queue.append(QueuedItem("Task"))
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
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            chat_scroll = app.query_one("#chat-scroll")
            footer = app.query_one(AgentFooter)
            inp = app.query_one(ReplInput)

            app._prompt_queue.append(QueuedItem("Stale queued prompt"))
            app._update_queue_ui()

            agent_msg = AgentResponse()

            with patch("ollama_agent.interfaces.repl.stream_agent_events", side_effect=asyncio.CancelledError()):
                with self.assertRaises(asyncio.CancelledError):
                    await app._run_stream("Trigger prompt", chat_scroll, agent_msg)

            self.assertEqual(len(app._prompt_queue), 0)
            self.assertEqual(footer._queued_count, 0)
            self.assertFalse(inp.disabled)
            self.assertFalse(app._is_approval_pending)

            self.assertEqual(len(list(chat_scroll.query(SystemMessage))), 0)
            sys_out = app.query_one(SystemOutputWidget)
            self.assertTrue(sys_out.display)
            self.assertIn("Execution interrupted by user", str(sys_out.render()))

    async def test_queue_clear_and_rm_do_not_interrupt_active_generation(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            mock_worker = MagicMock()
            app._is_generating = True
            app._current_worker = mock_worker

            app._prompt_queue.append(QueuedItem("Prompt 1"))
            app._prompt_queue.append(QueuedItem("Prompt 2"))
            app._prompt_queue.append(QueuedItem("Prompt 3"))
            app._update_queue_ui()

            # /queue rm 2 removes item 2 without stopping worker
            repl._handle_queue_cmd(["rm", "2"])
            self.assertEqual(len(app._prompt_queue), 2)
            self.assertEqual(app._prompt_queue[0].text, "Prompt 1")
            self.assertEqual(app._prompt_queue[1].text, "Prompt 3")
            mock_worker.cancel.assert_not_called()
            self.assertTrue(app._is_generating)

            # /queue clear clears queue without stopping worker
            repl._handle_queue_cmd(["clear"])
            self.assertEqual(len(app._prompt_queue), 0)
            mock_worker.cancel.assert_not_called()
            self.assertTrue(app._is_generating)


class TestToolApprovalModalAndQueue(unittest.IsolatedAsyncioTestCase):
    """Unit tests for tool approval interaction with prompt queue and input widget."""

    async def test_approval_modal_keeps_repl_input_enabled(self) -> None:
        repl, runtime = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)
            chat_scroll = app.query_one("#chat-scroll")
            agent_msg = AgentResponse()
            chat_scroll.mount(agent_msg)

            # Mock graph state to return interrupts requiring approval
            mock_state = MagicMock()
            mock_interrupt = MagicMock()
            mock_interrupt.value = {"action_requests": [{"name": "read_file", "args": {"path": "main.py"}}]}
            mock_state.interrupts = [mock_interrupt]
            runtime.graph.aget_state = AsyncMock(return_value=mock_state)

            with patch("ollama_agent.interfaces.repl.stream_agent_events", new_callable=AsyncMock):
                await app._run_stream("Read file prompt", chat_scroll, agent_msg)

            self.assertTrue(app._is_approval_pending)
            self.assertFalse(inp.disabled)
            self.assertIn("Approval required", str(footer.render()))

    async def test_queuing_prompts_during_approval_pending(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            footer = app.query_one(AgentFooter)

            app._is_approval_pending = True
            footer.set_approval(True)

            app.on_repl_input_submitted(ReplInput.Submitted(inp, "Prompt during approval"))

            self.assertEqual(len(app._prompt_queue), 1)
            self.assertEqual(app._prompt_queue[0].text, "Prompt during approval")
            self.assertEqual(footer._queued_count, 1)
            self.assertIn("1 queued", str(footer.render()))

    async def test_approval_decision_resumes_stream_and_drains_queue(self) -> None:
        repl, runtime = _create_mock_repl()
        mock_state = MagicMock()
        mock_state.interrupts = []
        runtime.graph.aget_state = AsyncMock(return_value=mock_state)

        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")
            agent_msg = AgentResponse()
            chat_scroll.mount(agent_msg)

            app._is_approval_pending = True
            app._prompt_queue.append(QueuedItem("Subsequent queued prompt"))
            app._update_queue_ui()

            processed_calls: list[object] = []

            async def fake_stream_events(
                rt: object, prompt: str | Command[object], renderer: object, auto_close: bool = True
            ) -> None:
                processed_calls.append(prompt)

            with patch("ollama_agent.interfaces.repl.stream_agent_events", side_effect=fake_stream_events):
                await app._handle_approval_decision([{"type": "approve"}], chat_scroll, agent_msg)
                await pilot.pause()

            self.assertFalse(app._is_approval_pending)
            self.assertEqual(len(processed_calls), 2)
            self.assertIsInstance(processed_calls[0], Command)
            self.assertEqual(processed_calls[1], "Subsequent queued prompt")
            self.assertEqual(len(app._prompt_queue), 0)


class TestQueueSlashCommandsHandler(unittest.IsolatedAsyncioTestCase):
    """Unit tests for /queue and /queue clear command handler."""

    def test_queue_list_when_empty(self) -> None:
        repl, _ = _create_mock_repl()
        repl._handle_queue_cmd([])
        output = repl.console.file.getvalue()
        self.assertIn("Prompt queue is empty", output)

    async def test_queue_list_with_items(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("Prompt alpha"))
            app._prompt_queue.append(QueuedItem("Prompt beta"))

            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["list"])
            output = repl.console.file.getvalue()

            self.assertIn("Queued prompts (2)", output)
            self.assertIn("#1 Prompt alpha", output)
            self.assertIn("#2 Prompt beta", output)

    async def test_queue_clear_subcommand(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("Item 1"))
            app._prompt_queue.append(QueuedItem("Item 2"))
            app._prompt_queue.append(QueuedItem("Item 3"))
            app._update_queue_ui()

            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["clear"])
            output = repl.console.file.getvalue()

            self.assertIn("Prompt queue cleared (3 removed)", output)
            self.assertEqual(len(app._prompt_queue), 0)
            footer = app.query_one(AgentFooter)
            self.assertEqual(footer._queued_count, 0)

    async def test_queue_rm_success(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("Item 1"))
            app._prompt_queue.append(QueuedItem("Item 2"))
            app._prompt_queue.append(QueuedItem("Item 3"))
            app._update_queue_ui()

            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["rm", "2"])
            output = repl.console.file.getvalue()

            self.assertIn("Removed #2 from prompt queue: Item 2", output)
            self.assertEqual(len(app._prompt_queue), 2)
            self.assertEqual(app._prompt_queue[0].text, "Item 1")
            self.assertEqual(app._prompt_queue[1].text, "Item 3")
            footer = app.query_one(AgentFooter)
            self.assertEqual(footer._queued_count, 2)

    async def test_queue_rm_with_hash_prefix(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("Item 1"))
            app._prompt_queue.append(QueuedItem("Item 2"))
            app._update_queue_ui()

            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["rm", "#1"])
            output = repl.console.file.getvalue()

            self.assertIn("Removed #1 from prompt queue: Item 1", output)
            self.assertEqual(len(app._prompt_queue), 1)
            self.assertEqual(app._prompt_queue[0].text, "Item 2")

    async def test_queue_rm_aliases_remove_and_delete(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("Alpha"))
            app._prompt_queue.append(QueuedItem("Beta"))
            app._update_queue_ui()

            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["remove", "1"])
            self.assertIn("Removed #1 from prompt queue: Alpha", repl.console.file.getvalue())

            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["delete", "1"])
            self.assertIn("Removed #1 from prompt queue: Beta", repl.console.file.getvalue())
            self.assertEqual(len(app._prompt_queue), 0)

    def test_queue_rm_missing_position(self) -> None:
        repl, _ = _create_mock_repl()
        repl._handle_queue_cmd(["rm"])
        output = repl.console.file.getvalue()
        self.assertIn("Usage: /queue rm <position>", output)

    def test_queue_rm_when_empty(self) -> None:
        repl, _ = _create_mock_repl()
        repl._handle_queue_cmd(["rm", "1"])
        output = repl.console.file.getvalue()
        self.assertIn("Prompt queue is empty.", output)

    async def test_queue_rm_invalid_position_non_digit(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("Item 1"))
            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["rm", "abc"])
            output = repl.console.file.getvalue()
            self.assertIn("Invalid queue position 'abc'. Usage: /queue rm <position>", output)

    async def test_queue_rm_out_of_range(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("Item 1"))
            app._prompt_queue.append(QueuedItem("Item 2"))
            app._update_queue_ui()

            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["rm", "0"])
            self.assertIn("Queue position 0 out of range (queue has 2 items).", repl.console.file.getvalue())

            repl.console = Console(file=io.StringIO())
            repl._handle_queue_cmd(["rm", "5"])
            self.assertIn("Queue position 5 out of range (queue has 2 items).", repl.console.file.getvalue())

    async def test_queue_rm_autocomplete(self) -> None:
        repl, _ = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test():
            app._prompt_queue.append(QueuedItem("First queued prompt"))
            app._prompt_queue.append(QueuedItem("Second queued prompt"))

            # Level 2 completions for /queue rm
            comps = app._slash_completions("/queue rm ")
            self.assertEqual(len(comps), 2)
            self.assertEqual(comps[0][0], "/queue rm 1")
            self.assertEqual(comps[1][0], "/queue rm 2")
            self.assertIn("#1", comps[0][1].plain)
            self.assertIn("First queued prompt", comps[0][1].plain)

            # Filtered completion
            comps_filtered = app._slash_completions("/queue remove 2")
            self.assertEqual(len(comps_filtered), 1)
            self.assertEqual(comps_filtered[0][0], "/queue remove 2")

            # Hash-prefixed filter
            comps_hash = app._slash_completions("/queue delete #1")
            self.assertEqual(len(comps_hash), 1)
            self.assertEqual(comps_hash[0][0], "/queue delete 1")

    def test_queue_invalid_subcommand(self) -> None:
        repl, _ = _create_mock_repl()
        repl._handle_queue_cmd(["unknown_cmd"])
        output = repl.console.file.getvalue()
        self.assertIn("Unknown queue subcommand 'unknown_cmd'", output)

    def test_queue_handler_when_app_is_none(self) -> None:
        repl, _ = _create_mock_repl()
        repl.app = None

        repl._handle_queue_cmd(["list"])
        self.assertIn("Prompt queue is empty", repl.console.file.getvalue())

        repl.console = Console(file=io.StringIO())
        repl._handle_queue_cmd(["clear"])
        self.assertIn("Prompt queue cleared (0 removed)", repl.console.file.getvalue())


class TestSessionTransitionsWithQueue(unittest.IsolatedAsyncioTestCase):
    """Unit tests for session creation, switching, and resuming with queued items."""

    async def test_session_new_drains_subsequent_queue(self) -> None:
        repl, runtime = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            app._prompt_queue.append(QueuedItem("/session new"))
            app._prompt_queue.append(QueuedItem("First message in fresh session"))
            app._update_queue_ui()

            streamed: list[str] = []

            async def fake_stream_events(
                rt: object, prompt: str | Command[object], renderer: object, auto_close: bool = True
            ) -> None:
                if isinstance(prompt, str):
                    streamed.append(prompt)

            with (
                patch("ollama_agent.interfaces.repl.new_session", return_value="sess_new_999"),
                patch("ollama_agent.interfaces.repl.stream_agent_events", side_effect=fake_stream_events),
            ):
                app._process_next_in_queue()
                await pilot.pause()

                self.assertEqual(runtime.thread_id, "sess_new_999")
                self.assertEqual(len(app._prompt_queue), 0)
                self.assertEqual(streamed, ["First message in fresh session"])

    async def test_session_resume_success_drains_subsequent_queue(self) -> None:
        repl, runtime = _create_mock_repl()
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

            app._prompt_queue.append(QueuedItem("/session resume sess_target_123"))
            app._prompt_queue.append(QueuedItem("Followup question"))
            app._update_queue_ui()

            streamed: list[str] = []

            async def fake_stream_events(
                rt: object, prompt: str | Command[object], renderer: object, auto_close: bool = True
            ) -> None:
                if isinstance(prompt, str):
                    streamed.append(prompt)

            with (
                patch("ollama_agent.interfaces.repl.resume_session", return_value="sess_target_123"),
                patch("ollama_agent.interfaces.repl.stream_agent_events", side_effect=fake_stream_events),
            ):
                app._process_next_in_queue()
                await pilot.pause()

                self.assertEqual(runtime.thread_id, "sess_target_123")
                self.assertEqual(runtime.last_context_tokens, 128)

                user_msgs = list(chat_scroll.query(UserMessage))
                agent_msgs = list(chat_scroll.query(AgentResponse))
                self.assertEqual(len(user_msgs), 2)
                self.assertEqual(len(agent_msgs), 2)

                self.assertEqual(len(app._prompt_queue), 0)
                self.assertEqual(streamed, ["Followup question"])

    async def test_session_resume_not_found_drains_subsequent_queue(self) -> None:
        repl, runtime = _create_mock_repl()
        app = OllamaAgentApp(repl)
        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")

            app._prompt_queue.append(QueuedItem("/session resume nonexistent_id"))
            app._prompt_queue.append(QueuedItem("Subsequent message"))

            streamed: list[str] = []

            async def fake_stream_events(
                rt: object, prompt: str | Command[object], renderer: object, auto_close: bool = True
            ) -> None:
                if isinstance(prompt, str):
                    streamed.append(prompt)

            with (
                patch("ollama_agent.interfaces.repl.resume_session", return_value=None),
                patch("ollama_agent.interfaces.repl.stream_agent_events", side_effect=fake_stream_events),
            ):
                app._process_next_in_queue()
                await pilot.pause()

                self.assertEqual(len(list(chat_scroll.query(SystemMessage))), 0)
                self.assertEqual(len(app._prompt_queue), 0)
                self.assertEqual(streamed, ["Subsequent message"])

    async def test_clear_and_new_aliases_drain_subsequent_queue(self) -> None:
        for alias in ("/clear", "/new"):
            with self.subTest(alias=alias):
                repl, runtime = _create_mock_repl()
                app = OllamaAgentApp(repl)
                async with app.run_test() as pilot:
                    app._prompt_queue.append(QueuedItem(alias))
                    app._prompt_queue.append(QueuedItem("Next action"))

                    streamed: list[str] = []

                    async def fake_stream_events(
                        rt: object, prompt: str | Command[object], renderer: object, auto_close: bool = True
                    ) -> None:
                        if isinstance(prompt, str):
                            streamed.append(prompt)

                    with (
                        patch("ollama_agent.interfaces.repl.new_session", return_value="sess_alias_001"),
                        patch("ollama_agent.interfaces.repl.stream_agent_events", side_effect=fake_stream_events),
                    ):
                        app._process_next_in_queue()
                        await pilot.pause()

                        self.assertEqual(runtime.thread_id, "sess_alias_001")
                        self.assertEqual(len(app._prompt_queue), 0)
                        self.assertEqual(streamed, ["Next action"])
