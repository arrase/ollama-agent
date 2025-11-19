"""Built-in tools for the agent."""

from __future__ import annotations

import subprocess
from typing import Any, Dict, Optional, TypedDict
from functools import wraps
from agents import function_tool

from ..memory import (
    Mem0InitializationError,
    Mem0NotConfiguredError,
    add_memory_entry,
    search_memories,
)


class CommandResult(TypedDict):
    success: bool
    stdout: str
    stderr: str
    exit_code: int


class Mem0ToolResult(TypedDict, total=False):
    success: bool
    data: Dict[str, Any]
    error: str


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
    """Execute a shell command and return the result.

    Args:
        command: The shell command to execute.

    Returns:
        A CommandResult containing success status, stdout, stderr, and exit code.
    """
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


def _mem0_error(message: str) -> Mem0ToolResult:
    return {"success": False, "error": message}

def mem0_tool(func):
    """Decorator to handle Mem0 errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return {"success": True, "data": func(*args, **kwargs)}
        except Mem0NotConfiguredError:
            return _mem0_error("Mem0 integration is not initialized")
        except Mem0InitializationError as exc:
            return _mem0_error(f"Mem0 initialization failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            return _mem0_error(f"Failed to {func.__name__.replace('mem0_', '').replace('_', ' ')}: {exc}")
    return wrapper


@function_tool
@mem0_tool
def mem0_add_memory(memory: str) -> Any:
    """Persist a new memory for the active user.

    Args:
        memory: The memory content to store.

    Returns:
        A Mem0ToolResult indicating success or failure, with stored data or error message.
    """
    return add_memory_entry(memory)


@function_tool
@mem0_tool
def mem0_search_memory(query: str, limit: Optional[int] = None) -> Any:
    """Search stored memories relevant to the provided query.

    Args:
        query: The search query to find relevant memories.
        limit: Optional maximum number of memories to return.

    Returns:
        A Mem0ToolResult containing search results or error message.
    """
    return search_memories(query, limit=limit)
