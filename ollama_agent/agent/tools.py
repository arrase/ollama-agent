"""Built-in tools for the agent."""

from __future__ import annotations

import subprocess
from typing import TypedDict

from agents import function_tool


class CommandResult(TypedDict):
    success: bool
    stdout: str
    stderr: str
    exit_code: int


_BUILTIN_TOOL_TIMEOUT = 30


def set_builtin_tool_timeout(timeout: int) -> None:
    global _BUILTIN_TOOL_TIMEOUT
    _BUILTIN_TOOL_TIMEOUT = timeout


def get_builtin_tool_timeout() -> int:
    return _BUILTIN_TOOL_TIMEOUT


def _error(stderr: str, *, exit_code: int = -1) -> CommandResult:
    return {"success": False, "stdout": "", "stderr": stderr, "exit_code": exit_code}


@function_tool
def execute_command(command: str) -> CommandResult:
    timeout = get_builtin_tool_timeout()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return _error(f"Error: The command exceeded the {timeout} second time limit")
    except Exception as exc:  # noqa: BLE001
        return _error(f"Error executing command: {exc}")
