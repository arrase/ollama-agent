"""Built-in tools for the agent."""

import subprocess
from typing import Any


def execute_command(command: str) -> dict[str, Any]:
    """
    Executes a local operating system command.
    
    Args:
        command: The command to execute in the system shell.
    
    Returns:
        A dictionary with the result of the execution:
        - success: True if the command executed successfully
        - stdout: The standard output of the command
        - stderr: The error output of the command
        - exit_code: The exit code of the command
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Error: The command exceeded the 30 second time limit",
            "exit_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error executing command: {str(e)}",
            "exit_code": -1
        }
