"""Built-in tools for the agent."""

import subprocess
from typing import TypedDict

from agents import function_tool


class CommandResult(TypedDict):
    """Result of a command execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int


# Global timeout value for built-in tool execution
_BUILTIN_TOOL_TIMEOUT = 30


def set_builtin_tool_timeout(timeout: int) -> None:
    """
    Set the global timeout for built-in tool execution.
    
    Args:
        timeout: Timeout in seconds for tool execution.
    """
    global _BUILTIN_TOOL_TIMEOUT
    _BUILTIN_TOOL_TIMEOUT = timeout


def get_builtin_tool_timeout() -> int:
    """
    Get the current global built-in tool timeout value.
    
    Returns:
        Current timeout in seconds.
    """
    return _BUILTIN_TOOL_TIMEOUT


def _create_error_result(stderr_message: str, exit_code: int = -1) -> CommandResult:
    """
    Create an error result dictionary.
    
    Args:
        stderr_message: Error message.
        exit_code: Exit code to return.
        
    Returns:
        CommandResult with error information.
    """
    return {
        "success": False,
        "stdout": "",
        "stderr": stderr_message,
        "exit_code": exit_code
    }


@function_tool
def execute_command(command: str) -> CommandResult:
    """
    Execute a local operating system command.
    
    Args:
        command: The command to execute in the system shell.
    
    Returns:
        A dictionary with the result of the execution:
        - success: True if the command executed successfully
        - stdout: The standard output of the command
        - stderr: The error output of the command
        - exit_code: The exit code of the command
    
    Note:
        The timeout is controlled by the global built-in tool timeout setting.
        Use set_builtin_tool_timeout() to change it.
    """
    timeout = get_builtin_tool_timeout()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return _create_error_result(
            f"Error: The command exceeded the {timeout} second time limit"
        )
    except Exception as e:
        return _create_error_result(f"Error executing command: {str(e)}")
    