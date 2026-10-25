from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import OptionList

from rich.console import Console

from ollama_agent.agent.episodic_memory import HistoryError
from ollama_agent.interfaces.tui import OllamaAgentApp, OllamaREPL
from ollama_agent.interfaces.tui.commands import run_slash_command
from ollama_agent.interfaces.tui.completion import _get_cached_models, _model_cache, file_completions
from ollama_agent.interfaces.tui.renderer import _TUIStreamingRenderer
from ollama_agent.interfaces.tui.screens import MainScreen
from ollama_agent.settings.config import Settings, load_settings, save_settings
from ollama_agent.tasks.commands import apply_task_settings
from ollama_agent.tasks.manager import Task
from ollama_agent.interfaces.tui.widgets import (
    AgentFooter,
    AgentHeader,
    AgentResponse,
    PromptQueueWidget,
    ReplInput,
    SystemOutputWidget,
    ToolApprovalWidget,
    ToolCallMessage,
    ToolOutputMessage,
    UserMessage,
)

from conftest import make_mock_repl, make_mock_runtime


class TestTUIComponents(unittest.IsolatedAsyncioTestCase):
    """Unit tests for modern minimalist TUI components and widgets."""

    def setUp(self) -> None:
        self.repl_mock = make_mock_repl()
        self.repl_mock.runtime.settings.model.name = "qwen2.5-coder:32b"
        self.repl_mock.runtime.settings.model.reasoning_effort = "high"
        self.repl_mock.runtime.settings.runtime.collapse_thinking = True
        self.repl_mock.runtime.yolo_mode = False
        self.repl_mock.runtime.stealth_mode = False
        self.repl_mock._rag_ctx = None

    def test_agent_header_format_default(self) -> None:
        self.repl_mock.runtime.settings.model.context_window = 16384
        self.repl_mock.runtime.last_context_tokens = 2048
        header = AgentHeader(self.repl_mock)
        header.update_header()
        rendered = header.render()
        self.assertIn("ollama-agent", str(rendered))
        self.assertIn("qwen2.5-coder:32b", str(rendered))
        self.assertIn("high", str(rendered))
        self.assertIn("Context:", str(rendered))
        self.assertIn("2.0k/16.4k", str(rendered))
        self.assertIn("YOLO: OFF", str(rendered))

    def test_agent_header_format_zero_tokens(self) -> None:
        self.repl_mock.runtime.settings.model.context_window = 16384
        self.repl_mock.runtime.last_context_tokens = 0
        header = AgentHeader(self.repl_mock)
        header.update_header()
        rendered = header.render()
        self.assertIn("Context:", str(rendered))
        self.assertIn("0/16.4k (0%)", str(rendered))

    def test_agent_header_format_yolo_and_rag(self) -> None:
        self.repl_mock.runtime.settings.model.context_window = 8192
        self.repl_mock.runtime.last_context_tokens = 7500
        self.repl_mock.runtime.yolo_mode = True
        rag_ctx_mock = MagicMock()
        rag_ctx_mock.rag_manager.current_database = "docs_db"
        self.repl_mock._rag_ctx = rag_ctx_mock

        header = AgentHeader(self.repl_mock)
        header.update_header()
        rendered = header.render()
        self.assertIn("YOLO", str(rendered))
        self.assertIn("docs_db", str(rendered))
        self.assertIn("Context:", str(rendered))

    def test_agent_footer_idle_and_generating(self) -> None:
        footer = AgentFooter()
        footer.update_footer()
        idle_text = str(footer.render())
        self.assertIn("interrupt", idle_text)
        self.assertIn("commands", idle_text)

        footer.set_generating(True)
        gen_text = str(footer.render())
        self.assertIn("Generating response", gen_text)

        footer.set_approval(True)
        appr_text = str(footer.render())
        self.assertIn("Approval required", appr_text)
        self.assertIn("approve", appr_text)
        self.assertIn("reject", appr_text)

        footer.set_approval(False)
        footer.set_generating(False)
        self.assertIn("interrupt", str(footer.render()))

    def test_agent_footer_with_queue(self) -> None:
        footer = AgentFooter()
        footer.set_queued_count(3)
        self.assertIn("3 queued", str(footer.render()))

        footer.set_generating(True)
        self.assertIn("3 queued", str(footer.render()))

        footer.set_approval(True)
        self.assertIn("3 queued", str(footer.render()))

        footer.set_queued_count(0)
        self.assertNotIn("queued", str(footer.render()))

    def test_user_message_compose(self) -> None:
        msg = UserMessage("How do I build a minimal CLI?")
        composed = list(msg.compose())
        self.assertEqual(len(composed), 2)
        self.assertIn("you", str(composed[0].render()))
        self.assertEqual(str(composed[1].render()), "How do I build a minimal CLI?")

    def test_tool_call_and_output_messages(self) -> None:
        tc = ToolCallMessage("read_file", agent_name="researcher")
        self.assertIn("read_file", str(tc.render()))
        self.assertIn("researcher", str(tc.render()))

        to = ToolOutputMessage(agent_name="researcher", output_len=256)
        self.assertIn("256 chars", str(to.render()))
        self.assertIn("researcher", str(to.render()))

    async def test_agent_response_workflow(self) -> None:
        class AgentResponseTestApp(App):
            def __init__(self) -> None:
                super().__init__()
                self.repl = MagicMock()
                self.repl.runtime.settings.runtime.collapse_thinking = True

            def compose(self) -> ComposeResult:
                yield AgentResponse()

        app = AgentResponseTestApp()
        async with app.run_test():
            agent_msg = app.query_one(AgentResponse)
            self.assertIsNotNone(agent_msg)

            # Test appending thinking
            agent_msg.append_thinking("Thinking step 1...")
            agent_msg.append_thinking(" step 2...")
            self.assertIsNotNone(agent_msg.current_thinking)
            assert agent_msg.current_thinking is not None
            agent_msg._animate_thinking()
            self.assertIn("Thinking", agent_msg.current_thinking.title)

            # Test appending text (stops thinking animation and flushes)
            agent_msg.append_text("# Heading\nHello world")
            agent_msg.flush_text()
            self.assertIsNotNone(agent_msg.current_text_widget)

            # Tool call
            agent_msg.add_tool_call("web_search")
            agent_msg.add_tool_output(output_len=120)
            agent_msg.add_error("Network timeout")
            agent_msg.add_warning("High latency")
            agent_msg.finish_generation()

            # Verify mounted widgets
            tool_calls = agent_msg.query(ToolCallMessage)
            self.assertEqual(len(tool_calls), 1)
            tool_outputs = agent_msg.query(ToolOutputMessage)
            self.assertEqual(len(tool_outputs), 1)
            sys_msgs = agent_msg.query(".system-message")
            self.assertEqual(len(sys_msgs), 2)

    def test_system_output_widget_lifecycle(self) -> None:
        widget = SystemOutputWidget()
        self.assertFalse(widget.display)

        # Show notice
        widget.show_notice("🛑 Prompt queue cleared.")
        self.assertTrue(widget.display)
        self.assertIn("Prompt queue cleared", str(widget.render()))

        # Show output with title
        widget.show_output("Model table content", title="/model list")
        self.assertTrue(widget.display)
        rendered = str(widget.render())
        self.assertIn("/model list", rendered)
        self.assertIn("Model table content", rendered)

        # Clear output
        widget.clear_output()
        self.assertFalse(widget.display)

    async def test_repl_input_history_and_keys(self) -> None:
        inp = ReplInput()
        inp.add_history_entry("first message")
        inp.add_history_entry("second message")
        inp.add_history_entry("second message")  # duplicate should not append

        self.assertEqual(len(inp._history), 2)
        self.assertEqual(inp._history[-1], "second message")

        # Test history navigation keys
        up_event = events.Key("up", "up")
        handled_up = inp._handle_history_key(up_event)
        self.assertTrue(handled_up)
        self.assertEqual(inp.text, "second message")

        handled_up2 = inp._handle_history_key(up_event)
        self.assertTrue(handled_up2)
        self.assertEqual(inp.text, "first message")

        down_event = events.Key("down", "down")
        handled_down = inp._handle_history_key(down_event)
        self.assertTrue(handled_down)
        self.assertEqual(inp.text, "second message")

    async def test_repl_input_multiline_and_keys(self) -> None:
        class InputApp(App):
            def compose(self) -> ComposeResult:
                yield OptionList(id="autocomplete-list")
                yield ReplInput(id="repl-input")

            def on_mount(self) -> None:
                self.query_one("#autocomplete-list", OptionList).display = False

        app = InputApp()
        with patch("ollama_agent.interfaces.tui.widgets.input.load_past_user_prompts", return_value=[]):
            async with app.run_test() as pilot:
                await pilot.pause()
                inp = app.query_one(ReplInput)
                inp.add_history_entry("prev command")

                # 1. Height adjusts on multiline text
                inp.text = "line 1\nline 2\nline 3"
                await pilot.pause()
                assert inp.styles.height is not None
                self.assertEqual(inp.styles.height.value, 3)

                # 2. Text reset resets height to default minimum (1)
                inp.text = ""
                await pilot.pause()
                assert inp.styles.height is not None
                self.assertEqual(inp.styles.height.value, 1)

                # 3. Backslash continuation (\ + Enter) inserts newline
                inp.text = "SELECT * FROM test \\"
                inp.action_cursor_line_end()
                inp.on_key(events.Key("enter", "\r"))
                await pilot.pause()
                self.assertEqual(inp.text, "SELECT * FROM test \n")
                assert inp.styles.height is not None
                self.assertEqual(inp.styles.height.value, 2)

                # 4. Multiline navigation vs history
                inp.text = "first line\nsecond line"
                await pilot.pause()
                inp.cursor_location = (1, 0)
                # Up on second line should NOT trigger history
                self.assertFalse(inp._handle_history_key(events.Key("up", "up")))

                # Up on first line but col > 0 should move cursor to (0, 0)
                inp.cursor_location = (0, 5)
                self.assertTrue(inp._handle_history_key(events.Key("up", "up")))
                self.assertEqual(inp.cursor_location, (0, 0))

                # Up on (0, 0) should trigger history
                self.assertTrue(inp._handle_history_key(events.Key("up", "up")))
                self.assertEqual(inp.text, "prev command")

    async def test_tool_approval_widget_keyboard_and_decisions(self) -> None:
        app_ref_mock = MagicMock()
        app_ref_mock.repl.runtime.auto_approved_tools = set()
        scroll_mock = MagicMock()
        agent_msg = AgentResponse()

        reqs = [{"name": "write_file", "args": {"path": "src/main.py"}}]
        widget = ToolApprovalWidget(
            action_requests=reqs,
            app_ref=app_ref_mock,
            scroll=scroll_mock,
            agent_msg=agent_msg,
        )

        class ApprovalApp(App):
            def compose(self) -> ComposeResult:
                yield widget

        app = ApprovalApp()
        async with app.run_test():
            # Auto-focused on approve-btn
            assert app.focused is not None
            self.assertEqual(app.focused.id, "approve-btn")

            # Test keypress 'y' for Approve
            key_event = events.Key("y", "y")
            widget.on_key(key_event)
            self.assertIsNone(widget.buttons_container)
            app_ref_mock._handle_approval_decision.assert_called_once()
            decisions = app_ref_mock._handle_approval_decision.call_args[0][0]
            self.assertEqual(decisions, [{"type": "approve"}])

    async def test_tool_approval_widget_navigation_and_shortcuts(self) -> None:
        # Test single direct shortcuts (y, n, a)
        for shortcut, expected_type in [
            ("y", "approve"),
            ("n", "reject"),
            ("a", "approve"),
        ]:
            app_ref_mock = MagicMock()
            app_ref_mock.repl.runtime.auto_approved_tools = set()
            widget = ToolApprovalWidget(
                action_requests=[{"name": "execute", "args": {"cmd": "ls"}}],
                app_ref=app_ref_mock,
                scroll=MagicMock(),
                agent_msg=AgentResponse(),
            )

            class ShortcutApp(App):
                def compose(self) -> ComposeResult:
                    yield widget

            app = ShortcutApp()
            async with app.run_test():
                widget.on_key(events.Key(shortcut, shortcut))
                app_ref_mock._handle_approval_decision.assert_called_once()
                decisions = app_ref_mock._handle_approval_decision.call_args[0][0]
                self.assertEqual(decisions[0]["type"], expected_type)
                if shortcut == "a":
                    self.assertIn("execute", app_ref_mock.repl.runtime.auto_approved_tools)

        # Test shortcut 'c' (cancel) calls cancel_approval directly without a reject decision
        app_ref_mock = MagicMock()
        widget = ToolApprovalWidget(
            action_requests=[{"name": "execute", "args": {"cmd": "ls"}}],
            app_ref=app_ref_mock,
            scroll=MagicMock(),
            agent_msg=AgentResponse(),
        )

        class CancelShortcutApp(App):
            def compose(self) -> ComposeResult:
                yield widget

        cancel_app = CancelShortcutApp()
        async with cancel_app.run_test():
            widget.on_key(events.Key("c", "c"))
            app_ref_mock._handle_approval_decision.assert_not_called()
            app_ref_mock.cancel_approval.assert_called_once()
            app_ref_mock.run_worker.assert_called_once()

        # Test arrow key navigation between buttons
        app_ref_mock = MagicMock()
        widget = ToolApprovalWidget(
            action_requests=[{"name": "execute", "args": {"cmd": "ls"}}],
            app_ref=app_ref_mock,
            scroll=MagicMock(),
            agent_msg=AgentResponse(),
        )

        class NavApp(App):
            def compose(self) -> ComposeResult:
                yield widget

        nav_app = NavApp()
        async with nav_app.run_test() as pilot:
            assert nav_app.focused is not None
            self.assertEqual(nav_app.focused.id, "approve-btn")

            # Press Right arrow -> reject-btn
            widget.on_key(events.Key("right", "right"))
            await pilot.pause()
            assert nav_app.focused is not None
            self.assertEqual(nav_app.focused.id, "reject-btn")

            # Press Right arrow -> allow-btn
            widget.on_key(events.Key("right", "right"))
            await pilot.pause()
            assert nav_app.focused is not None
            self.assertEqual(nav_app.focused.id, "allow-btn")

            # Press Right arrow -> cancel-btn
            widget.on_key(events.Key("right", "right"))
            await pilot.pause()
            assert nav_app.focused is not None
            self.assertEqual(nav_app.focused.id, "cancel-btn")

            # Press Right arrow (wrap around) -> approve-btn
            widget.on_key(events.Key("right", "right"))
            await pilot.pause()
            assert nav_app.focused is not None
            self.assertEqual(nav_app.focused.id, "approve-btn")

            # Press Left arrow (wrap around) -> cancel-btn
            widget.on_key(events.Key("left", "left"))
            await pilot.pause()
            assert nav_app.focused is not None
            self.assertEqual(nav_app.focused.id, "cancel-btn")

            # Press Enter on focused button (cancel-btn)
            widget.on_key(events.Key("enter", "\r"))
            app_ref_mock._handle_approval_decision.assert_not_called()
            app_ref_mock.cancel_approval.assert_called_once()
            app_ref_mock.run_worker.assert_called_once()


class TestOllamaAgentApp(unittest.IsolatedAsyncioTestCase):
    """Headless integration tests for OllamaAgentApp."""

    async def test_app_composition_and_yolo_toggle(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.stealth_mode = False
        repl_mock._get_commands.return_value = {}

        app = OllamaAgentApp(repl_mock)
        async with app.run_test():
            # Check widgets present
            self.assertIsNotNone(app.query_one(AgentHeader))
            self.assertIsNotNone(app.query_one(AgentFooter))
            self.assertIsNotNone(app.query_one(ReplInput))
            self.assertIsNotNone(app.query_one(OptionList))

            # Toggle YOLO mode and update UI
            repl_mock.runtime.yolo_mode = True
            app.update_mode_ui()
            prompt_char = app.query_one("#prompt-char")
            input_container = app.query_one("#input-container")
            self.assertEqual(prompt_char.styles.color.hex, "#F87171")
            self.assertTrue(input_container.has_class("yolo-mode"))

            repl_mock.runtime.yolo_mode = False
            app.update_mode_ui()
            self.assertEqual(prompt_char.styles.color.hex, "#38BDF8")
            self.assertFalse(input_container.has_class("yolo-mode"))

    async def test_autocomplete_trigger(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.stealth_mode = False

        cmd_spec = MagicMock()
        cmd_spec.summary = "Switch model"
        repl_mock._get_commands.return_value = {"/model": cmd_spec}

        app = OllamaAgentApp(repl_mock)
        async with app.run_test():
            autolist = app.query_one("#autocomplete-list", OptionList)
            self.assertFalse(autolist.display)

            # Trigger autocomplete with /
            app.update_autocomplete("/m")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 2)  # /model, /mcp

            # Aliases are documented for autocomplete
            app.update_autocomplete("/")
            all_ids = {autolist.get_option_at_index(i).id for i in range(autolist.option_count)}
            self.assertIn("/clear", all_ids)
            self.assertIn("/quit", all_ids)

            # Hide autocomplete
            app.hide_autocomplete()
            self.assertFalse(autolist.display)

    async def test_accept_completion_slash_command(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.stealth_mode = False

        cmd_spec = MagicMock()
        cmd_spec.summary = "Switch model"
        repl_mock._get_commands.return_value = {"/model": cmd_spec}

        app = OllamaAgentApp(repl_mock)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            inp.text = "/mo"
            app.update_autocomplete("/mo")
            app.accept_completion(0)
            self.assertEqual(inp.text, "/model ")

    async def test_autocomplete_subcommands_and_entities(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.stealth_mode = False

        # Mock tasks and skills
        mock_task = MagicMock(title="My Test Task")
        repl_mock._task_ctx.task_manager.list_all.return_value = [("test-task", mock_task)]

        mock_skill = MagicMock()
        mock_skill.name = "My Test Skill"
        repl_mock._skills_ctx.skill_manager.list_all.return_value = [("test-skill", mock_skill)]

        mock_rag_mgr = MagicMock()
        mock_rag_mgr.list_database_names.return_value = ["docs-db"]
        repl_mock._get_rag_ctx.return_value.rag_manager = mock_rag_mgr

        app = OllamaAgentApp(repl_mock)
        async with app.run_test():
            autolist = app.query_one("#autocomplete-list", OptionList)

            # 1. Level 1: Subcommands for /task
            app.update_autocomplete("/task ")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 4)  # list, create, run, delete

            # 1b. Level 1: Subcommands for /session including search and switch
            app.update_autocomplete("/session ")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 7)  # list, search, resume, switch, new, export, delete

            # 1c. Level 1: Subcommands for /agents
            app.update_autocomplete("/agents ")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 1)  # list
            app.accept_completion(0)
            inp = app.query_one(ReplInput)
            self.assertEqual(inp.text, "/agents list ")

            # 1d. Level 1: Subcommands for /queue
            app.update_autocomplete("/queue ")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 4)  # clear, rm, remove, delete
            app.accept_completion(0)
            inp = app.query_one(ReplInput)
            self.assertEqual(inp.text, "/queue clear ")

            app.update_autocomplete("/session sea")
            self.assertEqual(autolist.option_count, 1)
            app.accept_completion(0)
            inp = app.query_one(ReplInput)
            self.assertEqual(inp.text, "/session search ")

            # Filter subcommands
            app.update_autocomplete("/task r")
            self.assertEqual(autolist.option_count, 1)
            app.accept_completion(0)
            inp = app.query_one(ReplInput)
            self.assertEqual(inp.text, "/task run ")

            # 2. Level 2: Dynamic Task IDs for /task run
            app.update_autocomplete("/task run ")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 1)
            app.accept_completion(0)
            self.assertEqual(inp.text, "/task run test-task ")

            # 3. Level 2: Dynamic Skill IDs for /skill show
            app.update_autocomplete("/skill show ")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 1)
            app.accept_completion(0)
            self.assertEqual(inp.text, "/skill show test-skill ")

            # 4. Level 2: Dynamic RAG DBs for /rag load
            app.update_autocomplete("/rag load ")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 1)
            app.accept_completion(0)
            self.assertEqual(inp.text, "/rag load docs-db ")

            # 5. Level 2: Dynamic Session IDs for /session resume
            with patch(
                "ollama_agent.interfaces.tui.completion.get_available_sessions",
                return_value=[{"thread_id": "session-12345678", "steps": 5}],
            ):
                app.update_autocomplete("/session resume ")
                self.assertTrue(autolist.display)
                self.assertEqual(autolist.option_count, 1)
                app.accept_completion(0)
                self.assertEqual(inp.text, "/session resume session-12345678 ")

                # 6. Non-matching entity or extra tokens hides autocomplete
                app.update_autocomplete("/session resume betybetryj......")
                self.assertFalse(autolist.display)
                self.assertEqual(autolist.option_count, 0)

            # 7. Level 2: Dynamic Models for /model set
            mock_m1 = MagicMock(model="llama3.2:latest", size=4 * 1024**3)
            mock_m2 = MagicMock(model="mistral:latest", size=5 * 1024**3)
            with patch("ollama_agent.interfaces.tui.completion._list_models_sync", return_value=[mock_m1, mock_m2]):
                app.update_autocomplete("/model set ")
                self.assertTrue(autolist.display)
                self.assertEqual(autolist.option_count, 2)

                app.update_autocomplete("/model set llam")
                self.assertEqual(autolist.option_count, 1)
                app.accept_completion(0)
                self.assertEqual(inp.text, "/model set llama3.2:latest ")

            with patch(
                "ollama_agent.interfaces.tui.completion._list_models_sync",
                side_effect=OSError("Connection refused"),
            ):
                app.update_autocomplete("/model set ")
                self.assertFalse(autolist.display)
                self.assertEqual(autolist.option_count, 0)

    async def test_accept_completion_file_mention(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False
        repl_mock._get_commands.return_value = {}

        app = OllamaAgentApp(repl_mock)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            inp.text = "Look at @"
            with patch("ollama_agent.interfaces.tui.app.file_completions", return_value=[("src/main.py", "1.2 KB")]):
                app.update_autocomplete("Look at @")
                app.accept_completion(0)
                self.assertEqual(inp.text, "Look at @src/main.py ")

    async def test_run_slash_commands_dispatch(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.console = Console(file=io.StringIO())
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False

        def _toggle_yolo(args):
            repl_mock.runtime.yolo_mode = not repl_mock.runtime.yolo_mode

        def _set_effort(args):
            if args:
                repl_mock.runtime.settings.model.reasoning_effort = args[0]

        repl_mock._get_commands.return_value = {
            "/yolo": _toggle_yolo,
            "/effort": _set_effort,
        }

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            # 1. /clear
            chat_scroll = app.query_one("#chat-scroll")
            await chat_scroll.mount(UserMessage("clear me"))
            await pilot.pause()
            with patch("ollama_agent.interfaces.tui.repl.new_session", return_value="newsess12345678") as mock_new:
                await run_slash_command(app, "/clear")
                await pilot.pause()
                self.assertEqual(len(list(chat_scroll.query(UserMessage))), 0)
                mock_new.assert_called_once()
                self.assertEqual(repl_mock.runtime.thread_id, "newsess12345678")

            # 1b. /new
            await chat_scroll.mount(UserMessage("new me"))
            await pilot.pause()
            with patch("ollama_agent.interfaces.tui.repl.new_session", return_value="newsess87654321") as mock_new2:
                await run_slash_command(app, "/new")
                await pilot.pause()
                self.assertEqual(len(list(chat_scroll.query(UserMessage))), 0)
                mock_new2.assert_called_once()
                self.assertEqual(repl_mock.runtime.thread_id, "newsess87654321")

            # 2. Unknown command
            await run_slash_command(app, "/unknown-cmd")
            self.assertEqual(len(list(chat_scroll.query(".system-message"))), 0)
            sys_out = app.query_one(SystemOutputWidget)
            self.assertTrue(sys_out.display)
            self.assertIn("Unknown command", str(sys_out.render()))

            # 3. /yolo command
            await run_slash_command(app, "/yolo")
            self.assertTrue(repl_mock.runtime.yolo_mode)

            # 4. /session resume command (mounting previous conversation)
            human_msg = MagicMock(type="human", content="Past question")
            ai_msg = MagicMock(type="ai", content="Past answer")
            repl_mock.runtime.get_thread_messages = AsyncMock(return_value=[human_msg, ai_msg])
            repl_mock.runtime.count_effective_tokens = AsyncMock(return_value=42)

            with patch("ollama_agent.interfaces.tui.commands.resume_session", return_value="sess12345678"):
                await run_slash_command(app, "/session resume sess12345678")
                await pilot.pause()
                self.assertEqual(len(list(chat_scroll.query(UserMessage))), 1)
                self.assertEqual(len(list(chat_scroll.query(AgentResponse))), 1)
                self.assertEqual(len(list(chat_scroll.query(".system-message"))), 0)
                self.assertIn("Resumed session", str(sys_out.render()))

            # 5. /effort command updates header
            with patch.object(app.query_one(AgentHeader), "update_header") as mock_update_header:
                await run_slash_command(app, "/effort high")
                mock_update_header.assert_called_once()
                self.assertEqual(repl_mock.runtime.settings.model.reasoning_effort, "high")

            # 6. /task create conversational dispatch
            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                await run_slash_command(app, "/task create my-new-task")
                mock_stream.assert_called_once()
                prompt_arg = mock_stream.call_args[0][0]
                self.assertIn("task-creator", prompt_arg)
                self.assertIn("my-new-task", prompt_arg)

            # 7. /skill create conversational dispatch
            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                await run_slash_command(app, "/skill create git-commit-helper")
                mock_stream.assert_called_once()
                prompt_arg = mock_stream.call_args[0][0]
                self.assertIn("skill-creator", prompt_arg)
                self.assertIn("git-commit-helper", prompt_arg)

    async def test_task_run_restores_runtime_settings(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.console = Console(file=io.StringIO())
        repl_mock.runtime.settings.model.name = "chat-model"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.reload = AsyncMock()
        repl_mock._rag_ctx = None
        repl_mock._get_commands.return_value = {}

        task = MagicMock(title="My Task", model="task-model", reasoning_effort="high", prompt="Do things")
        task.render.return_value = "Do things"
        repl_mock._task_ctx.resolve_task.return_value = ("my-task", task)

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            with (
                patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream,
                patch("ollama_agent.interfaces.tui.commands.apply_task_settings", wraps=apply_task_settings) as mock_apply,
            ):
                await run_slash_command(app, "/task run my-task -y")
                await pilot.pause()

            mock_apply.assert_called_once_with(repl_mock.runtime.settings, task)
            task.render.assert_called_once_with({})
            mock_stream.assert_awaited_once()
            self.assertEqual(mock_stream.call_args[0][0], "Do things")

            # Runtime values are restored after the task finished (nothing persists).
            self.assertEqual(repl_mock.runtime.settings.model.name, "chat-model")
            self.assertEqual(repl_mock.runtime.settings.model.reasoning_effort, "medium")
            self.assertFalse(repl_mock.runtime.yolo_mode)
            self.assertEqual(repl_mock.runtime.reload.await_count, 2)

    async def test_task_run_with_variables_in_repl(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.console = Console(file=io.StringIO())
        repl_mock.runtime.settings.model.name = "chat-model"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.reload = AsyncMock()
        repl_mock._rag_ctx = None
        repl_mock._get_commands.return_value = {}

        task = MagicMock(title="Param Task", model="task-model", reasoning_effort="high", prompt="P")
        task.render.return_value = "Rendered: file=src/app.py mode=strict"
        repl_mock._task_ctx.resolve_task.return_value = ("param-task", task)

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                await run_slash_command(app, "/task run param-task file=src/app.py mode=strict")
                await pilot.pause()

            task.render.assert_called_once_with({"file": "src/app.py", "mode": "strict"})
            mock_stream.assert_awaited_once()
            self.assertEqual(mock_stream.call_args[0][0], "Rendered: file=src/app.py mode=strict")

    async def test_task_run_invalid_var_in_repl_shows_notice(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.console = Console(file=io.StringIO())
        repl_mock.runtime.settings.model.name = "chat-model"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.reload = AsyncMock()
        repl_mock._rag_ctx = None
        repl_mock._get_commands.return_value = {}

        task = MagicMock(title="Param Task", model="task-model", reasoning_effort="high")
        repl_mock._task_ctx.resolve_task.return_value = ("param-task", task)

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            with (
                patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream,
                patch.object(app, "show_system_notice") as mock_notice,
            ):
                await run_slash_command(app, "/task run param-task invalid_var")
                await pilot.pause()

            mock_stream.assert_not_called()
            mock_notice.assert_called_once()
            self.assertIn("invalid_var", mock_notice.call_args[0][0])

    async def test_task_run_template_error_in_repl_shows_notice(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.console = Console(file=io.StringIO())
        repl_mock.runtime.settings.model.name = "chat-model"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.reload = AsyncMock()
        repl_mock._rag_ctx = None
        repl_mock._get_commands.return_value = {}

        task = MagicMock(title="Param Task", model="task-model", reasoning_effort="high")
        task.render.side_effect = ValueError("Missing required input: file")
        repl_mock._task_ctx.resolve_task.return_value = ("param-task", task)

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            with (
                patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream,
                patch.object(app, "show_system_notice") as mock_notice,
            ):
                await run_slash_command(app, "/task run param-task")
                await pilot.pause()

            mock_stream.assert_not_called()
            mock_notice.assert_called_once()
            self.assertIn("Missing required input: file", mock_notice.call_args[0][0])

    async def test_tui_streaming_renderer_events(self) -> None:
        repl_mock = make_mock_repl()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False
        repl_mock._get_commands.return_value = {}

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")
            agent_msg = AgentResponse()
            await chat_scroll.mount(agent_msg)
            await pilot.pause()

            renderer = _TUIStreamingRenderer(app, chat_scroll, agent_msg)
            renderer.on_reasoning_delta({"type": "reasoning_delta", "content": "Let me think..."})
            renderer.on_text_delta({"type": "text_delta", "content": "Done response"})
            renderer.on_tool_call({"type": "tool_call", "name": "file_search", "agent_name": "worker"})
            renderer.on_tool_output({"type": "tool_output", "agent_name": "worker", "output_len": 64})
            renderer.on_warning({"type": "warning", "content": "Rate limit warning"})
            renderer.on_error({"type": "error", "content": "Something failed"})
            renderer.close()

            tool_calls = agent_msg.query(ToolCallMessage)
            self.assertEqual(len(tool_calls), 1)
            tool_outputs = agent_msg.query(ToolOutputMessage)
            self.assertEqual(len(tool_outputs), 1)
            sys_msgs = agent_msg.query(".system-message")
            self.assertEqual(len(sys_msgs), 2)


class TestOllamaREPLUnit(unittest.IsolatedAsyncioTestCase):
    """Unit tests for OllamaREPL helper methods and lifecycle."""

    async def test_repl_yolo_handler(self) -> None:
        runtime_mock = MagicMock()
        runtime_mock.yolo_mode = False
        runtime_mock.settings.rag = MagicMock()
        runtime_mock.settings.model.name = "gemma4:26b"
        runtime_mock.settings.model.base_url = "http://localhost:11434"

        repl = OllamaREPL(runtime=runtime_mock)
        repl.console = Console(file=io.StringIO())

        # Toggle on
        repl._handle_yolo_cmd([])
        self.assertTrue(runtime_mock.yolo_mode)

        # Explicit off
        repl._handle_yolo_cmd(["off"])
        self.assertFalse(runtime_mock.yolo_mode)

        # Explicit on
        repl._handle_yolo_cmd(["on"])
        self.assertTrue(runtime_mock.yolo_mode)

    async def test_repl_cleanup_closes_rag_and_runtime(self) -> None:
        runtime_mock = make_mock_runtime(aclose=AsyncMock())
        repl = OllamaREPL(runtime=runtime_mock)

        rag_ctx_mock = MagicMock()
        rag_ctx_mock.rag_manager.aclose = AsyncMock()
        repl._rag_ctx = rag_ctx_mock

        await repl.cleanup()
        rag_ctx_mock.rag_manager.aclose.assert_awaited_once()
        runtime_mock.aclose.assert_awaited_once()

    async def test_repl_cleanup_without_rag_still_closes_runtime(self) -> None:
        runtime_mock = make_mock_runtime(aclose=AsyncMock())
        repl = OllamaREPL(runtime=runtime_mock)
        repl._rag_ctx = None

        await repl.cleanup()
        runtime_mock.aclose.assert_awaited_once()

    async def test_slash_command_output_renders_in_system_output_not_in_chat(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")
            sys_out = app.query_one(SystemOutputWidget)

            async def fake_handler(args: list[str]) -> None:
                repl.console.print("Models: llama3, mistral")

            with patch.dict(app.repl._get_commands(), {"/model": fake_handler}):
                await run_slash_command(app, "/model list")
                await pilot.pause()

            # Chat scroll MUST remain completely clean of system messages
            self.assertEqual(len(list(chat_scroll.query(".system-message"))), 0)
            self.assertEqual(len(list(chat_scroll.query(UserMessage))), 0)
            self.assertEqual(len(list(chat_scroll.query(AgentResponse))), 0)

            # System output widget MUST be visible and contain the captured output
            self.assertTrue(sys_out.display)
            rendered = str(sys_out.render())
            self.assertIn("/model list", rendered)
            self.assertIn("Models: llama3, mistral", rendered)

    async def test_esc_dismisses_system_output_when_idle(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test():
            sys_out = app.query_one(SystemOutputWidget)
            app.show_system_notice("A notice to dismiss")
            self.assertTrue(sys_out.display)

            # Press Escape when idle
            app.action_cancel_generation()
            self.assertFalse(sys_out.display)

    async def test_submitting_new_prompt_dismisses_system_output(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test() as pilot:
            inp = app.query_one(ReplInput)
            sys_out = app.query_one(SystemOutputWidget)

            app.show_system_notice("Previous notice")
            self.assertTrue(sys_out.display)

            with patch.object(app, "_run_stream", new_callable=AsyncMock):
                app.on_repl_input_submitted(ReplInput.Submitted(inp, "Hello agent"))
                await pilot.pause()

            self.assertFalse(sys_out.display)

    async def test_esc_dismisses_system_output_without_cancelling_active_worker(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test():
            sys_out = app.query_one(SystemOutputWidget)
            mock_worker = MagicMock()
            app._is_generating = True
            app._current_worker = mock_worker

            # Show system output (e.g. from /model list while generating)
            app.show_system_output("Models list...", title="/model list")
            self.assertTrue(sys_out.display)

            # First Escape: dismisses SystemOutputWidget, worker remains untouched
            app.action_cancel_generation()
            self.assertFalse(sys_out.display)
            mock_worker.cancel.assert_not_called()

            # Second Escape: now that SystemOutputWidget is closed, cancels worker
            app.action_cancel_generation()
            mock_worker.cancel.assert_called_once()

    async def test_widgets_adapt_height_without_limits(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test(size=(120, 50)) as pilot:
            sys_out = app.query_one(SystemOutputWidget)
            content = "\n".join([f"line {i}" for i in range(20)])
            app.show_system_output(content, title="/mcp list")
            await pilot.pause()
            self.assertGreaterEqual(sys_out.outer_size.height, 22)

            q = app.query_one(PromptQueueWidget)
            for i in range(4):
                app._prompt_queue.append(f"Item {i}")
            app._update_queue_ui()
            await pilot.pause()
            self.assertGreaterEqual(q.outer_size.height, 7)

    async def test_repl_input_load_history_handles_history_error(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        with (
            patch(
                "ollama_agent.interfaces.tui.widgets.input.load_past_user_prompts",
                side_effect=HistoryError("Corrupt DB"),
            ),
            patch.object(app, "show_system_notice") as mock_notice,
        ):
            async with app.run_test() as pilot:
                await pilot.pause()
                input_widget = app.query_one(ReplInput)
                mock_notice.assert_called_once()
                notice_text = mock_notice.call_args[0][0]
                self.assertIn("Prompt history unavailable", notice_text)
                self.assertEqual(input_widget._history, [])

    async def test_immediate_command_does_not_drain_queue(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test():
            app._prompt_queue.append("Pending prompt")
            with patch.object(app, "_process_next_in_queue") as mock_drain:
                await run_slash_command(app, "/yolo")
                mock_drain.assert_not_called()

    async def test_esc_cancels_mounted_approval_widget_buttons(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")
            agent_msg = AgentResponse()
            await chat_scroll.mount(agent_msg)
            widget = ToolApprovalWidget(
                action_requests=[{"name": "execute", "args": {"cmd": "rm -rf"}}],
                app_ref=app,
                scroll=chat_scroll,
                agent_msg=agent_msg,
            )
            await chat_scroll.mount(widget)
            await pilot.pause()

            app._is_approval_pending = True
            app.action_cancel_generation()
            await pilot.pause()
            self.assertTrue(widget._decided)
            self.assertIsNone(widget.buttons_container)
            self.assertFalse(app._is_approval_pending)

    async def test_enter_submits_when_input_already_matches_highlighted_completion(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test() as pilot:
            inp = app.query_one(ReplInput)
            autolist = app.query_one("#autocomplete-list", OptionList)

            # Complete a no-arg command -> popup should stay hidden
            inp.text = "/cl"
            app.update_autocomplete("/cl")
            self.assertTrue(autolist.display)
            app.accept_completion(0)
            await pilot.pause()
            self.assertEqual(inp.text, "/clear")
            self.assertFalse(autolist.display)

            # When popup is open and text matches highlighted option, Enter submits directly
            inp.text = "/model"
            app.update_autocomplete("/model")
            self.assertTrue(autolist.display)
            with patch.object(inp, "post_message") as mock_post:
                inp.on_key(events.Key("enter", "\r"))
                self.assertFalse(autolist.display)
                mock_post.assert_called_once()
                submitted_msg = mock_post.call_args[0][0]
                self.assertIsInstance(submitted_msg, ReplInput.Submitted)
                self.assertEqual(submitted_msg.value, "/model")

    def test_file_completions_subdirectory_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subdir = root / "src"
            subdir.mkdir()
            (subdir / "main.py").write_text("print(1)", encoding="utf-8")
            (subdir / "other.py").write_text("print(2)", encoding="utf-8")

            with patch("pathlib.Path.cwd", return_value=root):
                results = list(file_completions("src/m"))
                paths = [p for p, _ in results]
                self.assertIn("src/main.py", paths)
                self.assertNotIn("src/other.py", paths)

    async def test_thinking_after_text_flushes_text_in_order(self) -> None:
        class OrderTestApp(App):
            def __init__(self) -> None:
                super().__init__()
                self.repl = MagicMock()
                self.repl.runtime.settings.runtime.collapse_thinking = True

            def compose(self) -> ComposeResult:
                yield AgentResponse()

        app = OrderTestApp()
        async with app.run_test():
            agent_msg = app.query_one(AgentResponse)
            agent_msg.append_text("First text block")
            text_widget = agent_msg.current_text_widget
            self.assertIsNotNone(text_widget)
            # Appending thinking after text must flush the buffered text first
            with patch.object(agent_msg, "flush_text", wraps=agent_msg.flush_text) as mock_flush:
                agent_msg.append_thinking("Second thinking block")
                mock_flush.assert_called_once()
            self.assertIsNone(agent_msg.current_text_widget)
            self.assertIsNotNone(agent_msg.current_thinking)

    async def test_multiline_history_navigation_places_cursor_at_end(self) -> None:
        class HistoryCursorApp(App):
            def compose(self) -> ComposeResult:
                yield OptionList(id="autocomplete-list")
                yield ReplInput(id="repl-input")

        app = HistoryCursorApp()
        with patch("ollama_agent.interfaces.tui.widgets.input.load_past_user_prompts", return_value=[]):
            async with app.run_test() as pilot:
                await pilot.pause()
                inp = app.query_one(ReplInput)
                inp.add_history_entry("line one\nline two longer")
                inp._handle_history_key(events.Key("up", "up"))
                self.assertEqual(inp.cursor_location, (1, len("line two longer")))

    def test_main_screen_forward_event_catches_detached_widget_region_error(self) -> None:
        screen = MainScreen()
        screen._select_state = MagicMock(end=None)
        with (
            patch(
                "textual.screen.Screen._forward_event",
                side_effect=AttributeError("'NoneType' object has no attribute 'region'"),
            ),
            patch("ollama_agent.interfaces.tui.screens.logging.warning") as mock_warn,
        ):
            screen._forward_event(MagicMock())
            self.assertIsNone(screen._select_state)
            mock_warn.assert_called_once()

    def test_main_screen_forward_event_reraises_other_attribute_errors(self) -> None:
        screen = MainScreen()
        event = MagicMock()
        with patch(
            "textual.screen.Screen._forward_event",
            side_effect=AttributeError("some other attribute error"),
        ):
            with self.assertRaises(AttributeError):
                screen._forward_event(event)

    async def test_approval_cancel_aborts_without_resuming_stream_and_runs_cleanup(self) -> None:
        runtime_mock = make_mock_runtime()
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")
            agent_msg = AgentResponse()
            await chat_scroll.mount(agent_msg)
            await pilot.pause()

            cleanup_mock = AsyncMock()
            app._post_stream_cleanup = cleanup_mock
            app._is_approval_pending = True

            with (
                patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_run_stream,
                patch.object(app, "show_system_notice") as mock_notice,
            ):
                await app.cancel_approval()
                mock_run_stream.assert_not_awaited()
                cleanup_mock.assert_awaited_once()
                self.assertFalse(app._is_approval_pending)
                self.assertIsNone(app._post_stream_cleanup)
                mock_notice.assert_called_once()
                self.assertIn("Approval cancelled", str(mock_notice.call_args[0][0]))

    async def test_slash_session_delete_current_resets_session(self) -> None:
        runtime_mock = make_mock_runtime()
        runtime_mock.thread_id = "abcd1234"
        repl = OllamaREPL(runtime=runtime_mock)
        app = OllamaAgentApp(repl)

        async with app.run_test() as pilot:
            chat_scroll = app.query_one("#chat-scroll")
            await chat_scroll.mount(UserMessage("Hello"))
            await pilot.pause()

            def fake_delete(args: list[str]) -> None:
                repl.console.print(f"[green]✓ Deleted session: {args[1]}[/green]")

            repl._commands = {"/session": fake_delete}
            await run_slash_command(app, "/session delete abcd1234")
            self.assertNotEqual(runtime_mock.thread_id, "abcd1234")
            self.assertEqual(len(chat_scroll.children), 0)

    async def test_task_run_hitl_approval_save_settings_and_cancel_restores_settings(self) -> None:
        runtime_mock = make_mock_runtime()
        runtime_mock.settings = Settings()
        runtime_mock.settings.model.name = "base-model:latest"
        runtime_mock.settings.model.reasoning_effort = "low"
        repl = OllamaREPL(runtime=runtime_mock)
        task = Task(title="Deploy", prompt="Run deploy", model="task-model:70b", reasoning_effort="high")
        repl._task_ctx.task_manager.save("deploy", task)
        app = OllamaAgentApp(repl)

        async def fake_stream_events(rt: object, _prompt: object, renderer: object, **_kw: object) -> None:
            interrupt_obj = MagicMock()
            interrupt_obj.value = {"action_requests": [{"name": "execute", "args": {"cmd": "deploy.sh"}}]}
            await renderer.handle_interrupt({"interrupts": [interrupt_obj]}, rt)  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_file = Path(tmp_dir) / "settings.yaml"
            save_settings(runtime_mock.settings, settings_file)

            async with app.run_test() as pilot:
                with patch("ollama_agent.interfaces.tui.app.stream_agent_events", side_effect=fake_stream_events):
                    await run_slash_command(app, "/task run deploy")
                    await pilot.pause()

                # Task override is active in runtime while paused at HITL approval
                self.assertTrue(app._is_approval_pending)
                self.assertEqual(runtime_mock.settings.model.name, "task-model:70b")
                self.assertEqual(runtime_mock.settings.model.reasoning_effort, "high")

                # Concurrent save_settings (e.g. /params set) must not persist temporary task model/effort
                runtime_mock.settings.model.temperature = 0.4
                save_settings(runtime_mock.settings, settings_file)
                persisted = load_settings(settings_file)
                self.assertEqual(persisted.model.name, "base-model:latest")
                self.assertEqual(persisted.model.reasoning_effort, "low")
                self.assertEqual(persisted.model.temperature, 0.4)

                # Cancel approval via Escape -> restores runtime model/effort and calls _begin_cancel_approval once
                with patch.object(app, "_begin_cancel_approval", wraps=app._begin_cancel_approval) as spy_begin:
                    app.action_cancel_generation()
                    await pilot.pause()
                    self.assertEqual(spy_begin.call_count, 1)

                self.assertFalse(app._is_approval_pending)
                self.assertEqual(runtime_mock.settings.model.name, "base-model:latest")
                self.assertEqual(runtime_mock.settings.model.reasoning_effort, "low")

    def test_model_autocomplete_negative_cache_and_background_refresh(self) -> None:
        mock_m1 = MagicMock(model="qwen3:8b", size=4 * 1024**3)
        mock_m2 = MagicMock(model="gemma4:26b", size=8 * 1024**3)

        # 1. Negative caching: connection error is cached so subsequent keystrokes don't re-fetch
        with patch(
            "ollama_agent.interfaces.tui.completion._list_models_sync",
            side_effect=OSError("Connection refused"),
        ) as mock_fetch_err:
            self.assertEqual(_get_cached_models("http://unreachable:11434"), [])
            self.assertEqual(_get_cached_models("http://unreachable:11434"), [])
            self.assertEqual(mock_fetch_err.call_count, 1)

        # 2. Expired TTL returns cached value immediately and refreshes in background thread
        with patch(
            "ollama_agent.interfaces.tui.completion._list_models_sync",
            return_value=[mock_m2],
        ) as mock_fetch_refresh:
            key = ("http://localhost:11434", mock_fetch_refresh)
            _model_cache[key] = (time.monotonic() - 20.0, [mock_m1])
            immediate = _get_cached_models("http://localhost:11434")
            self.assertEqual(immediate, [mock_m1])
            for _ in range(20):
                if _model_cache[key][1] == [mock_m2]:
                    break
                time.sleep(0.005)
            self.assertEqual(_model_cache[key][1], [mock_m2])


