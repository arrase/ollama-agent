"""Runner module for non-interactive execution."""

from rich.console import Console

from .agent import OllamaAgent
from .streaming import ConsoleStreamingRenderer, stream_agent_events_with_renderer


async def run_non_interactive(
    agent: OllamaAgent,
    prompt: str,
) -> None:
    """Stream agent output to the console."""
    renderer = ConsoleStreamingRenderer(Console())
    async with agent.lifespan():
        await stream_agent_events_with_renderer(
            agent,
            prompt,
            renderer,
            ignore={"agent_update"},
        )
