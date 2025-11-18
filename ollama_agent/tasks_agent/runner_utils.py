"""Utility functions for task execution."""

from typing import Optional

from rich.console import Console

from ollama_agent.main_agent.agent import OllamaAgent
from ollama_agent.tasks import Task, TaskManager
from ollama_agent.streaming import stream_agent_events, ConsoleStreamingRenderer


async def run_non_interactive(
    agent: OllamaAgent,
    prompt: str,
    model: Optional[str] = None,
    effort: Optional[str] = None,
) -> None:
    """Stream agent output to the console."""
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


def find_task_or_exit(task_manager: TaskManager, task_id: str, console: Console) -> tuple[str, Task]:
    """Find a task by ID or prefix, exit if not found."""
    result = task_manager.find_task_by_prefix(task_id)

    if not result:
        console.print(f"[red]Task not found: {task_id}[/red]")
        raise SystemExit(1)

    return result
