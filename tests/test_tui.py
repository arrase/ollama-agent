from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Markdown, OptionList, Static, TextArea

from ollama_agent.interfaces.repl import OllamaAgentApp, OllamaREPL, _TUIStreamingRenderer
from ollama_agent.interfaces.tui_components import (
    AgentFooter,
    AgentHeader,
    AgentResponse,
    ReplInput,
    SkillCreateModal,
    SystemMessage,
    TaskCreateModal,
    ToolApprovalWidget,
    ToolCallMessage,
    ToolOutputMessage,
    UserMessage,
)


class TestTUIComponents(unittest.IsolatedAsyncioTestCase):
    """Unit tests for modern minimalist TUI components and widgets."""

    def setUp(self) -> None:
        self.repl_mock = MagicMock()
        self.repl_mock.runtime.settings.model.name = "qwen2.5-coder:32b"
        self.repl_mock.runtime.settings.model.reasoning_effort = "high"
        self.repl_mock.runtime.settings.runtime.collapse_thinking = True
        self.repl_mock.runtime.yolo_mode = False
        self.repl_mock._rag_ctx = None

    def test_agent_header_format_default(self) -> None:
        header = AgentHeader(self.repl_mock)
        header.update_header()
        rendered = header.render()
        self.assertIn("ollama-agent", str(rendered))
        self.assertIn("qwen2.5-coder:32b", str(rendered))
        self.assertIn("high", str(rendered))
        self.assertIn("YOLO: OFF", str(rendered))

    def test_agent_header_format_yolo_and_rag(self) -> None:
        self.repl_mock.runtime.yolo_mode = True
        rag_ctx_mock = MagicMock()
        rag_ctx_mock.rag_manager.current_database = "docs_db"
        self.repl_mock._rag_ctx = rag_ctx_mock

        header = AgentHeader(self.repl_mock)
        header.update_header()
        rendered = header.render()
        self.assertIn("YOLO", str(rendered))
        self.assertIn("docs_db", str(rendered))

    def test_agent_footer_idle_and_generating(self) -> None:
        footer = AgentFooter()
        footer.update_footer()
        idle_text = str(footer.render())
        self.assertIn("interrupt", idle_text)
        self.assertIn("commands", idle_text)

        footer.set_generating(True)
        gen_text = str(footer.render())
        self.assertIn("Generating response", gen_text)

        footer.set_generating(False)
        self.assertIn("interrupt", str(footer.render()))

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
            def compose(self) -> ComposeResult:
                yield AgentResponse()

        app = AgentResponseTestApp()
        async with app.run_test() as pilot:
            agent_msg = app.query_one(AgentResponse)
            self.assertIsNotNone(agent_msg)

            # Test appending thinking
            agent_msg.append_thinking("Thinking step 1...")
            agent_msg.append_thinking(" step 2...")
            self.assertIsNotNone(agent_msg.current_thinking)
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
            sys_msgs = agent_msg.query(SystemMessage)
            self.assertEqual(len(sys_msgs), 2)

    async def test_repl_input_history_and_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_file = Path(tmpdir) / "tui_history.txt"
            with patch("ollama_agent.interfaces.tui_components.APP_DIR", Path(tmpdir)):
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
        async with app.run_test() as pilot:
            inp = app.query_one(ReplInput)
            inp.add_history_entry("prev command")

            # 1. Height adjusts on multiline text
            inp.text = "line 1\nline 2\nline 3"
            await pilot.pause()
            self.assertEqual(inp.styles.height.value, 3)

            # 2. Text reset resets height to default minimum (2)
            inp.text = ""
            await pilot.pause()
            self.assertEqual(inp.styles.height.value, 2)

            # 3. Backslash continuation (\ + Enter) inserts newline
            inp.text = "SELECT * FROM test \\"
            inp.action_cursor_line_end()
            inp.on_key(events.Key("enter", "\r"))
            await pilot.pause()
            self.assertEqual(inp.text, "SELECT * FROM test \n")
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
        async with app.run_test() as pilot:
            # Test keypress 'y' for Approve
            key_event = events.Key("y", "y")
            widget.on_key(key_event)
            self.assertIsNone(widget.buttons_container)
            app_ref_mock._handle_approval_decision.assert_called_once()
            decisions = app_ref_mock._handle_approval_decision.call_args[0][0]
            self.assertEqual(decisions, [{"type": "approve"}])


    async def test_task_create_modal(self) -> None:
        app_mock = MagicMock()
        app_mock.repl.runtime.settings.model.name = "qwen2.5-coder:32b"
        app_mock.repl.runtime.settings.model.reasoning_effort = "high"

        modal = TaskCreateModal(app_mock, "test-task", False)

        class ModalHostApp(App):
            def compose(self) -> ComposeResult:
                yield Static("main screen")

        app = ModalHostApp()
        async with app.run_test() as pilot:
            result_holder = []
            def on_dismiss(res):
                result_holder.append(res)
            app.push_screen(modal, on_dismiss)
            await pilot.pause()

            # Verify inputs are present
            title_inp = modal.query_one("#title-input", Input)
            title_inp.value = "My Test Task"
            prompt_area = modal.query_one("#prompt-area", TextArea)
            prompt_area.text = "Do something cool"

            # Press create button
            create_btn = modal.query_one("#create-btn", Button)
            modal.on_button_pressed(Button.Pressed(create_btn))
            await pilot.pause()

            self.assertEqual(len(result_holder), 1)
            self.assertEqual(result_holder[0][0], "test-task")
            self.assertEqual(result_holder[0][1], "My Test Task")
            self.assertEqual(result_holder[0][4], "Do something cool")

    async def test_skill_create_modal(self) -> None:
        app_mock = MagicMock()
        modal = SkillCreateModal(app_mock, "test-skill", False)

        class ModalHostApp(App):
            def compose(self) -> ComposeResult:
                yield Static("main screen")

        app = ModalHostApp()
        async with app.run_test() as pilot:
            result_holder = []
            def on_dismiss(res):
                result_holder.append(res)
            app.push_screen(modal, on_dismiss)
            await pilot.pause()

            # Verify inputs are present
            name_inp = modal.query_one("#name-input", Input)
            name_inp.value = "Skill Name"
            desc_inp = modal.query_one("#desc-input", Input)
            desc_inp.value = "Skill Description"
            inst_area = modal.query_one("#instructions-area", TextArea)
            inst_area.text = "Run tests"

            # Press create button
            create_btn = modal.query_one("#create-btn", Button)
            modal.on_button_pressed(Button.Pressed(create_btn))
            await pilot.pause()

            self.assertEqual(len(result_holder), 1)
            self.assertEqual(result_holder[0][0], "test-skill")
            self.assertEqual(result_holder[0][1], "Skill Name")
            self.assertEqual(result_holder[0][2], "Skill Description")
            self.assertEqual(result_holder[0][3], "Run tests")


class TestOllamaAgentApp(unittest.IsolatedAsyncioTestCase):
    """Headless integration tests for OllamaAgentApp."""

    async def test_app_composition_and_yolo_toggle(self) -> None:
        repl_mock = MagicMock()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False
        repl_mock._get_commands.return_value = {}

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            # Check widgets present
            self.assertIsNotNone(app.query_one(AgentHeader))
            self.assertIsNotNone(app.query_one(AgentFooter))
            self.assertIsNotNone(app.query_one(ReplInput))
            self.assertIsNotNone(app.query_one(OptionList))

            # Toggle YOLO mode and update UI
            repl_mock.runtime.yolo_mode = True
            app.update_yolo_ui()
            prompt_char = app.query_one("#prompt-char")
            self.assertEqual(prompt_char.styles.color.hex, "#F87171")

            repl_mock.runtime.yolo_mode = False
            app.update_yolo_ui()
            self.assertEqual(prompt_char.styles.color.hex, "#38BDF8")

    async def test_autocomplete_trigger(self) -> None:
        repl_mock = MagicMock()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False

        cmd_spec = MagicMock()
        cmd_spec.summary = "Show help message"
        repl_mock._get_commands.return_value = {"/help": cmd_spec}

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            autolist = app.query_one("#autocomplete-list", OptionList)
            self.assertFalse(autolist.display)

            # Trigger autocomplete with /
            app.update_autocomplete("/h")
            self.assertTrue(autolist.display)
            self.assertEqual(autolist.option_count, 1)

            # Hide autocomplete
            app.hide_autocomplete()
            self.assertFalse(autolist.display)

