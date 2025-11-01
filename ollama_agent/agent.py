"""AI agent using openai-agents and Ollama."""

from typing import Literal, cast

from openai import AsyncOpenAI
from agents import Agent, Runner, ModelSettings, set_default_openai_client, set_tracing_disabled, set_default_openai_api
from openai.types.shared import Reasoning
from .tools import execute_command

ReasoningEffortValue = Literal["low", "medium", "high"]
ALLOWED_REASONING_EFFORTS: tuple[ReasoningEffortValue, ...] = ("low", "medium", "high")


def _validate_reasoning_effort(effort: str) -> ReasoningEffortValue:
    """Validates and normalizes reasoning effort value."""
    return effort if effort in ALLOWED_REASONING_EFFORTS else "medium"  # type: ignore


class OllamaAgent:
    """AI agent that connects to Ollama."""

    def __init__(
        self, 
        model: str, 
        base_url: str = "http://localhost:11434/v1/", 
        api_key: str = "ollama", 
        reasoning_effort: str = "medium"
    ):
        """
        Initializes the agent.

        Args:
            model: Name of the model to use.
            base_url: Base URL of the Ollama server.
            api_key: API key (required but ignored by Ollama).
            reasoning_effort: Reasoning effort level (low, medium, high).
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning_effort = _validate_reasoning_effort(reasoning_effort)

        # Configure openai-agents to work with Ollama
        set_tracing_disabled(True)
        set_default_openai_api("chat_completions")

        # Create and set OpenAI client compatible with Ollama
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        set_default_openai_client(self.client, use_for_tracing=False)

        # Create the agent with the command execution tool
        self.agent = Agent(
            name="Ollama Assistant",
            instructions="You are a helpful AI assistant that can help with various tasks. "
                        "You have access to a tool that allows you to execute operating system commands.",
            model=model,
            tools=[execute_command],
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=cast(ReasoningEffortValue, self.reasoning_effort))
            )
        )

    async def run_async(self, prompt: str) -> str:
        """
        Runs the agent asynchronously.

        Args:
            prompt: The user's prompt.

        Returns:
            The agent's response.
        """
        try:
            result = await Runner.run(self.agent, input=prompt)
            return str(result.final_output)
        except Exception as e:
            return f"Error: {str(e)}"
