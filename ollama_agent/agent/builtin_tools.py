"""Built-in tools for the agent."""

from __future__ import annotations

import subprocess
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Optional

from agents import function_tool

from ..core import CommandResult, Mem0ToolResult
from ..memory import Mem0InitializationError

if TYPE_CHECKING:
    from ..memory import MemoryManager

# Context variables for thread-safe access
_tool_timeout: ContextVar[int] = ContextVar("tool_timeout", default=30)
_memory_manager: ContextVar[Optional["MemoryManager"]] = ContextVar("memory_manager", default=None)


def set_tool_timeout(timeout: int) -> None:
    """Set timeout for command execution."""
    _tool_timeout.set(timeout)


def get_tool_timeout() -> int:
    """Get current command timeout."""
    return _tool_timeout.get()


def set_memory_manager(manager: Optional["MemoryManager"]) -> None:
    """Set the active MemoryManager."""
    _memory_manager.set(manager)


def get_memory_manager() -> Optional["MemoryManager"]:
    """Get the current MemoryManager."""
    return _memory_manager.get()


@function_tool
def execute_command(command: str) -> CommandResult:
    """Execute a shell command and return the result.

    Args:
        command: The shell command to execute.

    Returns:
        A CommandResult with success status, stdout, stderr, and exit code.
    """
    timeout = get_tool_timeout()
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "exit_code": -1}


def _mem0_call(fn: str, *args: Any, **kwargs: Any) -> Mem0ToolResult:
    """Execute a memory manager operation safely."""
    manager = get_memory_manager()
    if not manager:
        return {"success": False, "error": "Mem0 not initialized"}
    try:
        data = getattr(manager, fn)(*args, **kwargs)
        return {"success": True, "data": data}
    except Mem0InitializationError as e:
        return {"success": False, "error": f"Mem0 init failed: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def mem0_add_memory(memory: str) -> Mem0ToolResult:
    """Persist a new memory for the active user."""
    return _mem0_call("add", memory)


@function_tool
def mem0_search_memory(query: str, limit: Optional[int] = None) -> Mem0ToolResult:
    """Search stored memories relevant to the query."""
    return _mem0_call("search", query, limit=limit)


BUILTIN_TOOLS = [execute_command, mem0_add_memory, mem0_search_memory]
