"""Runner module for non-interactive execution."""

from typing import Optional

from rich.console import Console

from .agent import OllamaAgent
from .streaming import ConsoleStreamingRenderer, stream_agent_events


async def run_non_interactive(
    agent: OllamaAgent,
    prompt: str,
    model: Optional[str] = None,
    effort: Optional[str] = None,
) -> None:
    """Stream agent output to the console."""
    await agent.initialize()
    renderer = ConsoleStreamingRenderer(Console())
    try:
        await stream_agent_events(
            agent,
            prompt,
            renderer,
            model=model,
            reasoning_effort=effort,
            ignore={"agent_update"},
        )
    finally:
        renderer.close()
        await agent.cleanup()
