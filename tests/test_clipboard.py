from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from textual import events

from ollama_agent.interfaces.clipboard import copy_to_system_clipboard, get_system_clipboard
from ollama_agent.interfaces.repl import OllamaAgentApp, OllamaREPL
from ollama_agent.interfaces.tui_components import ReplInput, UserMessage


class TestClipboard(unittest.TestCase):
    """Unit tests for system clipboard utilities."""

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_copy_darwin(self, mock_run: MagicMock) -> None:
        copy_to_system_clipboard("hello mac")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], ["pbcopy"])
        self.assertEqual(kwargs["input"], b"hello mac")

    @patch("sys.platform", "linux")
    @patch("shutil.which", side_effect=lambda cmd: "/usr/bin/wl-copy" if cmd == "wl-copy" else None)
    @patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"})
    @patch("subprocess.run")
    def test_copy_linux_wayland(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        copy_to_system_clipboard("hello wayland")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], ["wl-copy"])
        self.assertEqual(kwargs["input"], b"hello wayland")

    @patch("sys.platform", "linux")
    @patch("shutil.which", side_effect=lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None)
    @patch.dict("os.environ", {}, clear=True)
    @patch("subprocess.run")
    def test_copy_linux_xclip(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        copy_to_system_clipboard("hello xclip")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], ["xclip", "-selection", "clipboard"])
        self.assertEqual(kwargs["input"], b"hello xclip")

    @patch("sys.platform", "win32")
    @patch("ctypes.windll", create=True)
    def test_copy_win32(self, mock_windll: MagicMock) -> None:
        mock_u32 = MagicMock()
        mock_k32 = MagicMock()
        mock_windll.user32 = mock_u32
        mock_windll.kernel32 = mock_k32
        mock_u32.OpenClipboard.return_value = 1
        mock_k32.GlobalAlloc.return_value = 12345
        mock_k32.GlobalLock.return_value = 67890

        copy_to_system_clipboard("hello windows")
        mock_u32.OpenClipboard.assert_called_once_with(None)
        mock_u32.EmptyClipboard.assert_called_once()
        mock_u32.SetClipboardData.assert_called_once_with(13, 12345)
        mock_u32.CloseClipboard.assert_called_once()

    @patch("sys.platform", "win32")
    @patch("ctypes.windll", create=True)
    def test_get_clipboard_win32(self, mock_windll: MagicMock) -> None:
        mock_u32 = MagicMock()
        mock_k32 = MagicMock()
        mock_windll.user32 = mock_u32
        mock_windll.kernel32 = mock_k32
        mock_u32.OpenClipboard.return_value = 1
        mock_u32.GetClipboardData.return_value = 12345
        mock_k32.GlobalLock.return_value = 67890

        with patch("ctypes.c_wchar_p") as mock_wchar:
            mock_wchar.return_value.value = "windows clipboard text"
            res = get_system_clipboard()
            self.assertEqual(res, "windows clipboard text")
        mock_u32.OpenClipboard.assert_called_once_with(None)
        mock_u32.CloseClipboard.assert_called_once()

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_get_clipboard_darwin(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="mac clipboard content")
        res = get_system_clipboard()
        self.assertEqual(res, "mac clipboard content")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], ["pbpaste"])
        self.assertEqual(kwargs["encoding"], "utf-8")


class TestAppClipboardIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for OllamaAgentApp selection and clipboard features."""

    async def test_app_copy_to_clipboard_invokes_system_copy(self) -> None:
        repl_mock = MagicMock(spec=OllamaREPL)
        repl_mock.runtime = MagicMock()
        repl_mock.runtime.yolo_mode = False
        repl_mock._rag_ctx = None
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"

        app = OllamaAgentApp(repl_mock)
        with patch("ollama_agent.interfaces.repl.copy_to_system_clipboard") as mock_sys_copy:
            async with app.run_test() as pilot:
                app.copy_to_clipboard("test copy content")
                mock_sys_copy.assert_called_once_with("test copy content")
                self.assertEqual(app._clipboard, "test copy content")

    async def test_text_selected_auto_copies(self) -> None:
        repl_mock = MagicMock(spec=OllamaREPL)
        repl_mock.runtime = MagicMock()
        repl_mock.runtime.yolo_mode = False
        repl_mock._rag_ctx = None
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"

        app = OllamaAgentApp(repl_mock)
        with patch("ollama_agent.interfaces.repl.copy_to_system_clipboard") as mock_sys_copy:
            async with app.run_test() as pilot:
                chat_scroll = app.query_one("#chat-scroll")
                msg = UserMessage("Selectable text here")
                await chat_scroll.mount(msg)
                await pilot.pause()

                app.screen.text_select_all()
                app.on_text_selected(events.TextSelected())
                self.assertTrue(mock_sys_copy.called)
                copied_arg = mock_sys_copy.call_args[0][0]
                self.assertIn("Selectable text here", copied_arg)

    async def test_action_copy_selection(self) -> None:
        repl_mock = MagicMock(spec=OllamaREPL)
        repl_mock.runtime = MagicMock()
        repl_mock.runtime.yolo_mode = False
        repl_mock._rag_ctx = None
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"

        app = OllamaAgentApp(repl_mock)
        with patch("ollama_agent.interfaces.repl.copy_to_system_clipboard") as mock_sys_copy:
            async with app.run_test() as pilot:
                chat_scroll = app.query_one("#chat-scroll")
                msg = UserMessage("Copy action message")
                await chat_scroll.mount(msg)
                await pilot.pause()

                app.screen.text_select_all()
                app.action_copy_selection()
                mock_sys_copy.assert_called()
                copied_arg = mock_sys_copy.call_args[0][0]
                self.assertIn("Copy action message", copied_arg)
