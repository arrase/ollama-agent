"""Task execution logic."""

import asyncio
from typing import Callable

from rich.console import Console

from ollama_agent.main_agent.agent import OllamaAgent
from ollama_agent.tasks import TaskManager
from .runner_utils import run_non_interactive, find_task_or_exit

async def run_task(task_id: str, agent_factory: Callable[..., OllamaAgent]) -> None:
    """Execute a saved task."""
    console = Console()
    task_manager = TaskManager()

    found_id, task = find_task_or_exit(task_manager, task_id, console)

    console.print(
        f"[bold cyan]Executing task:[/bold cyan] {task.title} ({found_id})")
    console.print(f"[bold blue]Prompt:[/bold blue] {task.prompt}")
    console.print(
        f"[bold]Model:[/bold] {task.model} | [bold]Effort:[/bold] {task.reasoning_effort}")
    console.print("")

    agent = agent_factory(
        model=task.model, reasoning_effort=task.reasoning_effort)
    await run_non_interactive(agent, task.prompt)
