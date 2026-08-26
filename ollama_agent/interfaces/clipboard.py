"""System clipboard integration utilities for macOS, Linux, and Windows."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys


class ClipboardError(Exception):
    """Raised when a system clipboard operation fails."""


def _copy_via_command(cmd: list[str], text: str) -> None:
    proc = subprocess.run(cmd, input=text.encode("utf-8"), check=False)
    if proc.returncode != 0:
        raise ClipboardError(f"'{cmd[0]}' exited with code {proc.returncode}")


def _paste_via_command(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise ClipboardError(f"'{cmd[0]}' exited with code {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def copy_to_system_clipboard(text: str) -> None:
    """Copy text to the OS system clipboard using native platform tools.

    Raises:
        ClipboardError: If the platform is unsupported or the platform tool fails.
    """
    if sys.platform == "darwin":
        _copy_via_command(["pbcopy"], text)
    elif sys.platform.startswith(("linux", "freebsd", "openbsd")):
        if (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET")) and shutil.which("wl-copy"):
            _copy_via_command(["wl-copy"], text)
        elif shutil.which("xclip"):
            _copy_via_command(["xclip", "-selection", "clipboard"], text)
        elif shutil.which("xsel"):
            _copy_via_command(["xsel", "--clipboard", "--input"], text)
        else:
            raise ClipboardError("No clipboard command found (install wl-copy, xclip or xsel)")
    elif sys.platform == "win32":
        u32 = ctypes.windll.user32
        k32 = ctypes.windll.kernel32
        if not u32.OpenClipboard(None):
            raise ClipboardError("Could not open the Windows clipboard")
        try:
            u32.EmptyClipboard()
            encoded = text.encode("utf-16-le") + b"\x00\x00"
            h_mem = k32.GlobalAlloc(0x0042, len(encoded))  # GMEM_MOVEABLE | GMEM_ZEROINIT
            if not h_mem:
                raise ClipboardError("GlobalAlloc failed while copying to the clipboard")
            p_mem = k32.GlobalLock(h_mem)
            if not p_mem:
                k32.GlobalFree(h_mem)
                raise ClipboardError("GlobalLock failed while copying to the clipboard")
            ctypes.memmove(p_mem, encoded, len(encoded))
            k32.GlobalUnlock(h_mem)
            if not u32.SetClipboardData(13, h_mem):  # CF_UNICODETEXT
                k32.GlobalFree(h_mem)
                raise ClipboardError("SetClipboardData failed while copying to the clipboard")
        finally:
            u32.CloseClipboard()
    else:
        raise ClipboardError(f"Unsupported platform: {sys.platform}")


def get_system_clipboard() -> str:
    """Retrieve text from the OS system clipboard.

    Raises:
        ClipboardError: If the platform is unsupported or the platform tool fails.
    """
    if sys.platform == "darwin":
        return _paste_via_command(["pbpaste"])
    elif sys.platform.startswith(("linux", "freebsd", "openbsd")):
        if (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET")) and shutil.which("wl-paste"):
            return _paste_via_command(["wl-paste", "--no-newline"])
        elif shutil.which("xclip"):
            return _paste_via_command(["xclip", "-selection", "clipboard", "-o"])
        elif shutil.which("xsel"):
            return _paste_via_command(["xsel", "--clipboard", "--output"])
        else:
            raise ClipboardError("No clipboard command found (install wl-paste, xclip or xsel)")
    elif sys.platform == "win32":
        u32 = ctypes.windll.user32
        k32 = ctypes.windll.kernel32
        if not u32.OpenClipboard(None):
            raise ClipboardError("Could not open the Windows clipboard")
        try:
            h_mem = u32.GetClipboardData(13)  # CF_UNICODETEXT
            if not h_mem:
                return ""
            p_mem = k32.GlobalLock(h_mem)
            if not p_mem:
                raise ClipboardError("GlobalLock failed while reading the clipboard")
            try:
                val = ctypes.c_wchar_p(p_mem).value
                return val if val is not None else ""
            finally:
                k32.GlobalUnlock(h_mem)
        finally:
            u32.CloseClipboard()
    else:
        raise ClipboardError(f"Unsupported platform: {sys.platform}")
