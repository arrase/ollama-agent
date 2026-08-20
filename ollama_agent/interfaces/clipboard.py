"""System clipboard integration utilities for macOS, Linux, and Windows."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys


def copy_to_system_clipboard(text: str) -> None:
    """Copy text to the OS system clipboard using native platform tools."""
    if sys.platform == "darwin":
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif sys.platform.startswith(("linux", "freebsd", "openbsd")):
        if (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET")) and shutil.which("wl-copy"):
            subprocess.run(
                ["wl-copy"],
                input=text.encode("utf-8"),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif shutil.which("xclip"):
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif shutil.which("xsel"):
            subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=text.encode("utf-8"),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif sys.platform == "win32":
        u32 = ctypes.windll.user32
        k32 = ctypes.windll.kernel32
        if u32.OpenClipboard(None):
            try:
                u32.EmptyClipboard()
                encoded = text.encode("utf-16-le") + b"\x00\x00"
                h_mem = k32.GlobalAlloc(0x0042, len(encoded))  # GMEM_MOVEABLE | GMEM_ZEROINIT
                if h_mem:
                    p_mem = k32.GlobalLock(h_mem)
                    ctypes.memmove(p_mem, encoded, len(encoded))
                    k32.GlobalUnlock(h_mem)
                    u32.SetClipboardData(13, h_mem)  # CF_UNICODETEXT
            finally:
                u32.CloseClipboard()


def get_system_clipboard() -> str:
    """Retrieve text from the OS system clipboard."""
    if sys.platform == "darwin":
        proc = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return proc.stdout
    elif sys.platform.startswith(("linux", "freebsd", "openbsd")):
        if (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET")) and shutil.which("wl-paste"):
            proc = subprocess.run(
                ["wl-paste", "--no-newline"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            return proc.stdout
        elif shutil.which("xclip"):
            proc = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            return proc.stdout
        elif shutil.which("xsel"):
            proc = subprocess.run(
                ["xsel", "--clipboard", "--output"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            return proc.stdout
    elif sys.platform == "win32":
        u32 = ctypes.windll.user32
        k32 = ctypes.windll.kernel32
        if u32.OpenClipboard(None):
            try:
                h_mem = u32.GetClipboardData(13)  # CF_UNICODETEXT
                if h_mem:
                    p_mem = k32.GlobalLock(h_mem)
                    if p_mem:
                        try:
                            return ctypes.c_wchar_p(p_mem).value or ""
                        finally:
                            k32.GlobalUnlock(h_mem)
            finally:
                u32.CloseClipboard()
    return ""
