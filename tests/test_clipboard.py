from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from textual import events

from ollama_agent.interfaces.clipboard import (
    ClipboardError,
    copy_to_system_clipboard,
    get_system_clipboard,
)
from ollama_agent.interfaces.repl import OllamaAgentApp, OllamaREPL
from ollama_agent.interfaces.tui_components import UserMessage


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


class TestClipboard(unittest.TestCase):
    """Unit tests for system clipboard utilities."""

    @patch("sys.platform", "darwin")
    @patch("subprocess.run", return_value=_proc())
    def test_copy_darwin(self, mock_run: MagicMock) -> None:
        copy_to_system_clipboard("hello mac")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], ["pbcopy"])
        self.assertEqual(kwargs["input"], b"hello mac")
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)

    @patch("sys.platform", "darwin")
    @patch("subprocess.run", return_value=_proc(returncode=1))
    def test_copy_darwin_failure_raises(self, mock_run: MagicMock) -> None:
        with self.assertRaises(ClipboardError):
            copy_to_system_clipboard("hello mac")

    @patch("sys.platform", "darwin")
    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["pbcopy"], timeout=3.0))
    def test_copy_timeout_raises_clipboard_error(self, mock_run: MagicMock) -> None:
        with self.assertRaises(ClipboardError):
            copy_to_system_clipboard("hello mac")

    @patch("sys.platform", "darwin")
    @patch("subprocess.run", side_effect=OSError("command failed"))
    def test_copy_oserror_raises_clipboard_error(self, mock_run: MagicMock) -> None:
        with self.assertRaises(ClipboardError):
            copy_to_system_clipboard("hello mac")

    @patch("sys.platform", "linux")
    @patch("shutil.which", side_effect=lambda cmd: "/usr/bin/wl-copy" if cmd == "wl-copy" else None)
    @patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"})
    @patch("subprocess.run", return_value=_proc())
    def test_copy_linux_wayland(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        copy_to_system_clipboard("hello wayland")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], ["wl-copy"])
        self.assertEqual(kwargs["input"], b"hello wayland")

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch.dict("os.environ", {}, clear=True)
    def test_copy_linux_without_tool_raises(self, mock_which: MagicMock) -> None:
        with self.assertRaises(ClipboardError):
            copy_to_system_clipboard("no tools")

    @patch("sys.platform", "linux")
    @patch("shutil.which", side_effect=lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None)
    @patch.dict("os.environ", {}, clear=True)
    @patch("subprocess.run", return_value=_proc())
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

        # memmove must be mocked: the mocked GlobalLock returns a plain int,
        # and writing to it as a pointer would segfault the interpreter.
        with patch("ctypes.memmove") as mock_memmove:
            copy_to_system_clipboard("hello windows")
            mock_memmove.assert_called_once()
        mock_u32.OpenClipboard.assert_called_once_with(None)
        mock_u32.EmptyClipboard.assert_called_once()
        mock_u32.SetClipboardData.assert_called_once_with(13, 12345)
        mock_u32.CloseClipboard.assert_called_once()

    @patch("sys.platform", "win32")
    @patch("ctypes.windll", create=True)
    def test_copy_win32_global_lock_failure_raises(self, mock_windll: MagicMock) -> None:
        mock_u32 = MagicMock()
        mock_k32 = MagicMock()
        mock_windll.user32 = mock_u32
        mock_windll.kernel32 = mock_k32
        mock_u32.OpenClipboard.return_value = 1
        mock_k32.GlobalAlloc.return_value = 12345
        mock_k32.GlobalLock.return_value = 0

        with patch("ctypes.memmove") as mock_memmove:
            with self.assertRaises(ClipboardError):
                copy_to_system_clipboard("hello windows")
            mock_memmove.assert_not_called()
        mock_k32.GlobalFree.assert_called_once_with(12345)

    @patch("sys.platform", "win32")
    @patch("ctypes.windll", create=True)
    def test_copy_win32_set_data_failure_raises(self, mock_windll: MagicMock) -> None:
        mock_u32 = MagicMock()
        mock_k32 = MagicMock()
        mock_windll.user32 = mock_u32
        mock_windll.kernel32 = mock_k32
        mock_u32.OpenClipboard.return_value = 1
        mock_k32.GlobalAlloc.return_value = 12345
        mock_k32.GlobalLock.return_value = 67890
        mock_u32.SetClipboardData.return_value = 0

        with patch("ctypes.memmove"):
            with self.assertRaises(ClipboardError):
                copy_to_system_clipboard("hello windows")
        mock_k32.GlobalFree.assert_called_once_with(12345)
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
        mock_run.return_value = _proc(stdout="mac clipboard content")
        res = get_system_clipboard()
        self.assertEqual(res, "mac clipboard content")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], ["pbpaste"])
        self.assertEqual(kwargs["encoding"], "utf-8")

    @patch("sys.platform", "darwin")
    @patch("subprocess.run", return_value=_proc(returncode=1, stderr="boom"))
    def test_get_clipboard_failure_raises(self, mock_run: MagicMock) -> None:
        with self.assertRaises(ClipboardError):
            get_system_clipboard()

    @patch("sys.platform", "darwin")
    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["pbpaste"], timeout=3.0))
    def test_get_clipboard_timeout_raises(self, mock_run: MagicMock) -> None:
        with self.assertRaises(ClipboardError):
            get_system_clipboard()

    @patch("sys.platform", "darwin")
    @patch("subprocess.run", side_effect=OSError("command failed"))
    def test_get_clipboard_oserror_raises(self, mock_run: MagicMock) -> None:
        with self.assertRaises(ClipboardError):
            get_system_clipboard()


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
            async with app.run_test():
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

    async def test_app_copy_failure_notifies_user(self) -> None:
        repl_mock = MagicMock(spec=OllamaREPL)
        repl_mock.runtime = MagicMock()
        repl_mock.runtime.yolo_mode = False
        repl_mock._rag_ctx = None
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"

        app = OllamaAgentApp(repl_mock)
        with patch("ollama_agent.interfaces.repl.copy_to_system_clipboard", side_effect=ClipboardError("no xclip")), \
             patch.object(OllamaAgentApp, "notify") as mock_notify:
            async with app.run_test():
                app.copy_to_clipboard("test copy content")
                self.assertEqual(app._clipboard, "test copy content")
                mock_notify.assert_called_once()
                self.assertIn("no xclip", mock_notify.call_args[0][0])

    async def test_screen_forward_event_handles_detached_widget_gracefully(self) -> None:
        repl_mock = MagicMock(spec=OllamaREPL)
        repl_mock.runtime = MagicMock()
        repl_mock.runtime.yolo_mode = False
        repl_mock._rag_ctx = None
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"

        app = OllamaAgentApp(repl_mock)
        async with app.run_test():
            event = events.MouseDown(None, 5, 5, 0, 0, 1, False, False, False)
            with (
                patch("textual.screen.Screen._forward_event", side_effect=AttributeError("'NoneType' object has no attribute 'region'")),
                patch("logging.warning"),
            ):
                app.screen._forward_event(event)
                self.assertIsNone(app.screen._select_state)

    async def test_screen_forward_event_reraises_other_attribute_error(self) -> None:
        repl_mock = MagicMock(spec=OllamaREPL)
        repl_mock.runtime = MagicMock()
        repl_mock.runtime.yolo_mode = False
        repl_mock._rag_ctx = None
        repl_mock.runtime.settings.model.name = "gemma4:26b"
        repl_mock.runtime.settings.model.reasoning_effort = "medium"

        app = OllamaAgentApp(repl_mock)
        async with app.run_test():
            event = events.MouseDown(None, 5, 5, 0, 0, 1, False, False, False)
            with patch("textual.screen.Screen._forward_event", side_effect=AttributeError("'CustomType' object has no attribute 'something_else'")):
                with self.assertRaises(AttributeError):
                    app.screen._forward_event(event)

