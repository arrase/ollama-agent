"""System clipboard integration utilities for macOS, Linux, and Windows."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys


class ClipboardError(Exception):
    """Raised when a system clipboard operation fails."""


def _configure_win32_clipboard(u32: ctypes.CDLL, k32: ctypes.CDLL) -> None:
    u32.OpenClipboard.argtypes = [ctypes.c_void_p]
    u32.OpenClipboard.restype = ctypes.c_int
    u32.CloseClipboard.argtypes = []
    u32.CloseClipboard.restype = ctypes.c_int
    u32.EmptyClipboard.argtypes = []
    u32.EmptyClipboard.restype = ctypes.c_int
    u32.GetClipboardData.argtypes = [ctypes.c_uint]
    u32.GetClipboardData.restype = ctypes.c_void_p
    u32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    u32.SetClipboardData.restype = ctypes.c_void_p

    k32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    k32.GlobalAlloc.restype = ctypes.c_void_p
    k32.GlobalLock.argtypes = [ctypes.c_void_p]
    k32.GlobalLock.restype = ctypes.c_void_p
    k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    k32.GlobalUnlock.restype = ctypes.c_int
    k32.GlobalFree.argtypes = [ctypes.c_void_p]
    k32.GlobalFree.restype = ctypes.c_void_p


def _detect_linux_clipboard_cmd(
    wayland_cmd: list[str],
    xclip_cmd: list[str],
    xsel_cmd: list[str],
) -> list[str]:
    if (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET")) and shutil.which(wayland_cmd[0]):
        return wayland_cmd
    if shutil.which("xclip"):
        return xclip_cmd
    if shutil.which("xsel"):
        return xsel_cmd
    raise ClipboardError(f"No clipboard command found (install {wayland_cmd[0]}, xclip or xsel)")


def _get_linux_copy_cmd() -> list[str]:
    """Detect available Linux clipboard copy command."""
    return _detect_linux_clipboard_cmd(
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    )


def _get_linux_paste_cmd() -> list[str]:
    """Detect available Linux clipboard paste command."""
    return _detect_linux_clipboard_cmd(
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "--clipboard", "--output"],
    )


def _copy_via_command(cmd: list[str], text: str) -> None:
    try:
        proc = subprocess.run(  # noqa: S603 - argv from hardcoded clipboard tool list
            cmd,
            check=False,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise ClipboardError(f"'{cmd[0]}' failed: {exc}") from exc
    if proc.returncode != 0:
        raise ClipboardError(f"'{cmd[0]}' exited with code {proc.returncode}")


def _paste_via_command(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(  # noqa: S603 - argv from hardcoded clipboard tool list
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise ClipboardError(f"'{cmd[0]}' failed: {exc}") from exc
    if proc.returncode != 0:
        raise ClipboardError(f"'{cmd[0]}' exited with code {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def _copy_win32(text: str) -> None:
    u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32  # type: ignore[attr-defined]
    _configure_win32_clipboard(u32, k32)
    if not u32.OpenClipboard(None):
        raise ClipboardError("Could not open the Windows clipboard")
    try:
        u32.EmptyClipboard()
        encoded = text.encode("utf-16-le") + b"\x00\x00"
        h_mem = k32.GlobalAlloc(0x0042, len(encoded))
        if not h_mem:
            raise ClipboardError("GlobalAlloc failed")
        if not (p_mem := k32.GlobalLock(h_mem)):
            k32.GlobalFree(h_mem)
            raise ClipboardError("GlobalLock failed while copying to the clipboard")
        try:
            ctypes.memmove(p_mem, encoded, len(encoded))
        finally:
            k32.GlobalUnlock(h_mem)
        if not u32.SetClipboardData(13, h_mem):
            k32.GlobalFree(h_mem)
            raise ClipboardError("SetClipboardData failed while copying to the clipboard")
    finally:
        u32.CloseClipboard()


def _paste_win32() -> str:
    u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32  # type: ignore[attr-defined]
    _configure_win32_clipboard(u32, k32)
    if not u32.OpenClipboard(None):
        raise ClipboardError("Could not open the Windows clipboard")
    try:
        if not (h_mem := u32.GetClipboardData(13)):
            raise ClipboardError("GetClipboardData failed while reading the clipboard")
        if not (p_mem := k32.GlobalLock(h_mem)):
            raise ClipboardError("GlobalLock failed while reading the clipboard")
        try:
            text = ctypes.c_wchar_p(p_mem).value
        finally:
            k32.GlobalUnlock(h_mem)
        if text is None:
            raise ClipboardError("Failed to decode text from clipboard")
        return text
    finally:
        u32.CloseClipboard()


def copy_to_system_clipboard(text: str) -> None:
    """Copy text to the OS system clipboard using native platform tools.

    Raises:
        ClipboardError: If the platform is unsupported or the platform tool fails.
    """
    if sys.platform == "darwin":
        _copy_via_command(["pbcopy"], text)
    elif sys.platform.startswith(("linux", "freebsd", "openbsd")):
        _copy_via_command(_get_linux_copy_cmd(), text)
    elif sys.platform == "win32":
        _copy_win32(text)
    else:
        raise ClipboardError(f"Unsupported platform: {sys.platform}")


def get_system_clipboard() -> str:
    """Retrieve text from the OS system clipboard.

    Raises:
        ClipboardError: If the platform is unsupported or the platform tool fails.
    """
    if sys.platform == "darwin":
        return _paste_via_command(["pbpaste"])
    if sys.platform.startswith(("linux", "freebsd", "openbsd")):
        return _paste_via_command(_get_linux_paste_cmd())
    if sys.platform == "win32":
        return _paste_win32()
    raise ClipboardError(f"Unsupported platform: {sys.platform}")
