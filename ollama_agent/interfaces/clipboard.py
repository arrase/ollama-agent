"""System clipboard integration utilities for macOS, Linux, and Windows."""

from __future__ import annotations

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
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
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
        subprocess.run(
            ["clip"],
            input=text.encode("utf-16"),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def get_system_clipboard() -> str:
    """Retrieve text from the OS system clipboard."""
    if sys.platform == "darwin":
        proc = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout
    elif sys.platform.startswith(("linux", "freebsd", "openbsd")):
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
            proc = subprocess.run(
                ["wl-paste", "--no-newline"],
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.stdout
        elif shutil.which("xclip"):
            proc = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.stdout
        elif shutil.which("xsel"):
            proc = subprocess.run(
                ["xsel", "--clipboard", "--output"],
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.stdout
    elif sys.platform == "win32":
        proc = subprocess.run(
            ["powershell", "-command", "Get-Clipboard"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.rstrip("\r\n")
    return ""
