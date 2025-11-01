"""Agente de IA usando openai-agents y Ollama."""

from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_disabled, set_default_openai_api


@function_tool
def execute_command(command: str) -> dict:
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
    import subprocess

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


class OllamaAgent:
    """AI agent that connects to Ollama."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1/", api_key: str = "ollama"):
        """
        Initializes the agent.

        Args:
            model: Name of the model to use.
            base_url: Base URL of the Ollama server.
            api_key: API key (required but ignored by Ollama).
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

        # Disable tracing to avoid attempts to connect to OpenAI
        set_tracing_disabled(True)

        # Configure to use chat_completions instead of responses
        set_default_openai_api("chat_completions")

        # Create OpenAI client compatible with Ollama
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )

        # Set the client globally for the agent
        set_default_openai_client(self.client, use_for_tracing=False)

        # Create the agent with the command execution tool
        self.agent = Agent(
            name="Ollama Assistant",
            instructions="""You are a helpful AI assistant that can help with various tasks.
You have access to a tool that allows you to execute operating system commands.""",
            model=model,
            tools=[execute_command]
        )

    def run(self, prompt: str) -> str:
        """
        Runs the agent with a prompt (synchronous fallback version).

        Args:
            prompt: The user's prompt.

        Returns:
            The agent's response.
        """
        import asyncio
        return asyncio.run(self.run_async(prompt))

    async def run_async(self, prompt: str) -> str:
        """
        Runs the agent asynchronously.

        Args:
            prompt: The user's prompt.

        Returns:
            La respuesta del agente.
        """
        try:
            result = await Runner.run(self.agent, input=prompt)
            return str(result.final_output)
        except Exception as e:
            return f"Error: {str(e)}"
