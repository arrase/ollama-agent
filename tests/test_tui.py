from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import OptionList

from rich.console import Console

from ollama_agent.interfaces.dispatch import REPLCommand
from ollama_agent.interfaces.repl import OllamaAgentApp, OllamaREPL, _TUIStreamingRenderer
from ollama_agent.interfaces.tui_components import (
    AgentFooter,
    AgentHeader,
    AgentResponse,
    ReplInput,
    SystemMessage,
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
            sys_msgs = agent_msg.query(SystemMessage)
            self.assertEqual(len(sys_msgs), 2)

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
        with patch("ollama_agent.interfaces.tui_components.load_past_user_prompts", return_value=[]):
            async with app.run_test() as pilot:
                await pilot.pause()
                inp = app.query_one(ReplInput)
                inp.add_history_entry("prev command")

                # 1. Height adjusts on multiline text
                inp.text = "line 1\nline 2\nline 3"
                await pilot.pause()
                assert inp.styles.height is not None
                self.assertEqual(inp.styles.height.value, 3)

                # 2. Text reset resets height to default minimum (2)
                inp.text = ""
                await pilot.pause()
                assert inp.styles.height is not None
                self.assertEqual(inp.styles.height.value, 2)

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
        # Test single direct shortcuts (y, n, a, c)
        for shortcut, expected_type in [
            ("y", "approve"),
            ("n", "reject"),
            ("a", "approve"),
            ("c", "reject"),
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
            app_ref_mock._handle_approval_decision.assert_called_once()
            decisions = app_ref_mock._handle_approval_decision.call_args[0][0]
            self.assertEqual(decisions[0]["type"], "reject")
            self.assertEqual(decisions[0]["message"], "User cancelled the execution.")




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
        async with app.run_test():
            # Check widgets present
            self.assertIsNotNone(app.query_one(AgentHeader))
            self.assertIsNotNone(app.query_one(AgentFooter))
            self.assertIsNotNone(app.query_one(ReplInput))
            self.assertIsNotNone(app.query_one(OptionList))

            # Toggle YOLO mode and update UI
            repl_mock.runtime.yolo_mode = True
            app.update_yolo_ui()
            prompt_char = app.query_one("#prompt-char")
            input_container = app.query_one("#input-container")
            self.assertEqual(prompt_char.styles.color.hex, "#F87171")
            self.assertTrue(input_container.has_class("yolo-mode"))

            repl_mock.runtime.yolo_mode = False
            app.update_yolo_ui()
            self.assertEqual(prompt_char.styles.color.hex, "#38BDF8")
            self.assertFalse(input_container.has_class("yolo-mode"))

    async def test_autocomplete_trigger(self) -> None:
        repl_mock = MagicMock()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False

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
            self.assertIn("/compress", all_ids)
            self.assertIn("/quit", all_ids)

            # Hide autocomplete
            app.hide_autocomplete()
            self.assertFalse(autolist.display)

    async def test_accept_completion_slash_command(self) -> None:
        repl_mock = MagicMock()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False

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
        repl_mock = MagicMock()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False

        # Mock tasks and skills
        mock_task = MagicMock(title="My Test Task")
        repl_mock._task_ctx.task_manager.list_all.return_value = [("test-task", mock_task)]

        mock_skill = MagicMock()
        mock_skill.name = "My Test Skill"
        repl_mock._skills_ctx.skill_manager.list_all.return_value = [("test-skill", mock_skill)]

        mock_rag_mgr = MagicMock()
        mock_rag_mgr.list_databases.return_value = [{"name": "docs-db", "chunks": 5}]
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
            with patch("ollama_agent.interfaces.repl.get_available_sessions", return_value=[{"thread_id": "session-12345678", "steps": 5}]):
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
            with patch("ollama_agent.interfaces.repl._list_models_sync", return_value=[mock_m1, mock_m2]):
                app.update_autocomplete("/model set ")
                self.assertTrue(autolist.display)
                self.assertEqual(autolist.option_count, 2)

                app.update_autocomplete("/model set llam")
                self.assertEqual(autolist.option_count, 1)
                app.accept_completion(0)
                self.assertEqual(inp.text, "/model set llama3.2:latest ")

            with patch("ollama_agent.interfaces.repl._list_models_sync", side_effect=OSError("Connection refused")):
                app.update_autocomplete("/model set ")
                self.assertFalse(autolist.display)
                self.assertEqual(autolist.option_count, 0)

    async def test_accept_completion_file_mention(self) -> None:
        repl_mock = MagicMock()
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False
        repl_mock._get_commands.return_value = {}

        app = OllamaAgentApp(repl_mock)
        async with app.run_test():
            inp = app.query_one(ReplInput)
            inp.text = "Look at @"
            with patch.object(app, "_file_completions", return_value=[("src/main.py", "1.2 KB")]):
                app.update_autocomplete("Look at @")
                app.accept_completion(0)
                self.assertEqual(inp.text, "Look at @src/main.py ")

    async def test_run_slash_commands_dispatch(self) -> None:
        repl_mock = MagicMock()
        repl_mock.console = Console(file=io.StringIO())
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock._rag_ctx = None
        repl_mock.runtime.yolo_mode = False

        async def _handle_new_session(args: list[str]) -> None:
            pass

        repl_mock._handle_new_session = _handle_new_session

        def _toggle_yolo(args):
            repl_mock.runtime.yolo_mode = not repl_mock.runtime.yolo_mode

        def _set_effort(args):
            if args:
                repl_mock.runtime.settings.model.reasoning_effort = args[0]

        repl_mock._get_commands.return_value = {
            "/yolo": REPLCommand(summary="toggle yolo", section="General", usage=None, handler=_toggle_yolo),
            "/effort": REPLCommand(summary="manage effort", section="Model Management", usage=None, handler=_set_effort),
        }

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            # 1. /clear
            chat_scroll = app.query_one("#chat-scroll")
            await chat_scroll.mount(UserMessage("clear me"))
            await pilot.pause()
            with patch("ollama_agent.interfaces.repl.new_session", return_value="newsess12345678"):
                async def _handle_new(args: list[str], _tid: str = "newsess12345678") -> None:
                    repl_mock.runtime.thread_id = _tid
                repl_mock._handle_new_session = _handle_new
                await app._run_slash_command("/clear")
                await pilot.pause()
                self.assertEqual(len(list(chat_scroll.query(UserMessage))), 0)
                self.assertEqual(repl_mock.runtime.thread_id, "newsess12345678")

            # 1b. /new
            await chat_scroll.mount(UserMessage("new me"))
            await pilot.pause()
            with patch("ollama_agent.interfaces.repl.new_session", return_value="newsess87654321"):
                async def _handle_new2(args: list[str], _tid: str = "newsess87654321") -> None:
                    repl_mock.runtime.thread_id = _tid
                repl_mock._handle_new_session = _handle_new2
                await app._run_slash_command("/new")
                await pilot.pause()
                self.assertEqual(len(list(chat_scroll.query(UserMessage))), 0)
                self.assertEqual(repl_mock.runtime.thread_id, "newsess87654321")

            # 2. Unknown command
            await app._run_slash_command("/unknown-cmd")
            sys_msgs = list(chat_scroll.query(SystemMessage))
            self.assertTrue(len(sys_msgs) > 0)
            self.assertIn("Unknown command", str(sys_msgs[-1].render()))

            # 3. /yolo command
            await app._run_slash_command("/yolo")
            self.assertTrue(repl_mock.runtime.yolo_mode)

            # 4. /session resume command (mounting previous conversation)
            human_msg = MagicMock(type="human", content="Past question")
            ai_msg = MagicMock(type="ai", content="Past answer")
            repl_mock.runtime.get_thread_messages = AsyncMock(return_value=[human_msg, ai_msg])
            repl_mock.runtime.count_effective_tokens = AsyncMock(return_value=42)

            with patch("ollama_agent.interfaces.repl.resume_session", return_value="sess12345678"):
                await app._run_slash_command("/session resume sess12345678")
                await pilot.pause()
                self.assertEqual(len(list(chat_scroll.query(UserMessage))), 1)
                self.assertEqual(len(list(chat_scroll.query(AgentResponse))), 1)

            # 5. /effort command updates header
            with patch.object(app.query_one(AgentHeader), "update_header") as mock_update_header:
                await app._run_slash_command("/effort high")
                mock_update_header.assert_called_once()
                self.assertEqual(repl_mock.runtime.settings.model.reasoning_effort, "high")

            # 6. /task create conversational dispatch
            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                await app._run_slash_command("/task create my-new-task")
                mock_stream.assert_called_once()
                prompt_arg = mock_stream.call_args[0][0]
                self.assertIn("task-creator", prompt_arg)
                self.assertIn("my-new-task", prompt_arg)

            # 7. /skill create conversational dispatch
            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream:
                await app._run_slash_command("/skill create git-commit-helper")
                mock_stream.assert_called_once()
                prompt_arg = mock_stream.call_args[0][0]
                self.assertIn("skill-creator", prompt_arg)
                self.assertIn("git-commit-helper", prompt_arg)

    async def test_task_run_restores_runtime_settings(self) -> None:
        repl_mock = MagicMock()
        repl_mock.console = Console(file=io.StringIO())
        repl_mock.runtime.settings.model.name = "chat-model"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"
        repl_mock.runtime.yolo_mode = False
        repl_mock.runtime.reload = AsyncMock()
        repl_mock._rag_ctx = None
        repl_mock._get_commands.return_value = {}

        task = MagicMock(title="My Task", model="task-model", reasoning_effort="high", prompt="Do things")
        repl_mock._task_ctx._find_or_exit.return_value = ("my-task", task)

        app = OllamaAgentApp(repl_mock)
        async with app.run_test() as pilot:
            with patch.object(app, "_run_stream", new_callable=AsyncMock) as mock_stream, \
                 patch("ollama_agent.interfaces.repl.apply_task_settings") as mock_apply:
                await app._run_slash_command("/task run my-task -y")
                await pilot.pause()

            mock_apply.assert_called_once_with(repl_mock.runtime.settings, task)
            mock_stream.assert_awaited_once()
            self.assertEqual(mock_stream.call_args[0][0], "Do things")

            # Runtime values are restored after the task finished (nothing persists).
            self.assertEqual(repl_mock.runtime.settings.model.name, "chat-model")
            self.assertEqual(repl_mock.runtime.settings.model.reasoning_effort, "medium")
            self.assertFalse(repl_mock.runtime.yolo_mode)
            self.assertEqual(repl_mock.runtime.reload.await_count, 2)

    async def test_tui_streaming_renderer_events(self) -> None:
        repl_mock = MagicMock()
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
            sys_msgs = agent_msg.query(SystemMessage)
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

    async def test_repl_cleanup_unloads_rag(self) -> None:
        runtime_mock = MagicMock()
        runtime_mock.aclose = AsyncMock()
        repl = OllamaREPL(runtime=runtime_mock)

        rag_ctx_mock = MagicMock()
        repl._rag_ctx = rag_ctx_mock

        await repl.cleanup()
        rag_ctx_mock.rag_manager.unload.assert_called_once()
        runtime_mock.aclose.assert_awaited_once()

