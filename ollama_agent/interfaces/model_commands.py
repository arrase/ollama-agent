"""Model-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import ollama
from rich.console import Console

from ..core import ModelCapabilityError, model_supports_tools

if TYPE_CHECKING:
    from ..agent import OllamaAgent


def list_models(console: Console, current_model: str) -> None:
    """Print available Ollama models with tool-support indicators."""
    try:
        models = getattr(ollama.list(), "models", [])
        if not models:
            console.print("[yellow]No models found in Ollama.[/yellow]")
            return
        console.print("[bold]Available Models:[/bold]\n[dim]─" + "─" * 59 + "[/dim]")
        for item in models:
            if not (name := getattr(item, "model", None)):
                continue
            marker = " [green]◀ current[/green]" if name == current_model else ""
            size_gb = getattr(item, "size", 0) / (1024 ** 3)
            size_str = f"{size_gb:.1f}GB" if size_gb else ""
            try:
                tool_icon = (
                    "[green]✓[/green]" if model_supports_tools(name) else "[red]✗[/red]"
                )
            except ModelCapabilityError:
                tool_icon = "[yellow]?[/yellow]"
            console.print(f"  {tool_icon} [cyan]{name}[/cyan] {size_str}{marker}")
        console.print(
            "[dim]─" * 60 + "[/dim]\n[dim]✓ = supports tools | Use /model-set <model> to switch[/dim]"
        )
    except Exception as exc:
        console.print(f"[red]Error listing models: {exc}[/red]")


async def set_model(
    console: Console,
    model_name: str,
    *,
    current_model: str,
    current_effort: str,
    active_agent: "OllamaAgent | None",
    agent_factory: Callable[..., "OllamaAgent"],
) -> tuple[str, "OllamaAgent | None"]:
    """Switch to model_name, returning (new_model, new_agent)."""
    try:
        available = {
            getattr(model, "model", "") for model in getattr(ollama.list(), "models", [])
        }
        if model_name not in available:
            console.print(
                f"[red]Model '{model_name}' not found.[/red]\n"
                "[dim]Use /models to see available models.[/dim]"
            )
            return current_model, active_agent
    except Exception as exc:
        console.print(f"[red]Error checking model: {exc}[/red]")
        return current_model, active_agent

    if model_name == current_model:
        console.print(f"[yellow]Already using model '{model_name}'.[/yellow]")
        return current_model, active_agent

    try:
        if not model_supports_tools(model_name):
            console.print(
                f"[red]Model '{model_name}' does not support tools.[/red]\n"
                "[dim]The agent requires tool support.[/dim]"
            )
            return current_model, active_agent
    except ModelCapabilityError as exc:
        console.print(f"[red]Cannot verify model capabilities: {exc}[/red]")
        return current_model, active_agent

    old_model = current_model
    old_session_id = active_agent.session_manager.get_session_id() if active_agent else None

    if active_agent:
        await active_agent.cleanup()

    try:
        new_agent: "OllamaAgent" = agent_factory(
            model=model_name, reasoning_effort=current_effort
        )
        await new_agent.initialize()
    except (ModelCapabilityError, SystemExit) as exc:
        console.print(
            f"[red]Failed to create agent with model '{model_name}': {exc}[/red]"
        )
        fallback = agent_factory(model=old_model, reasoning_effort=current_effort)
        if old_session_id:
            fallback.session_manager.load_session(old_session_id)
        return old_model, fallback

    if old_session_id:
        new_agent.session_manager.load_session(old_session_id)
    console.print(
        f"[green]✓ Switched from [cyan]{old_model}[/cyan] to [cyan]{model_name}[/cyan][/green]\n"
        "[dim]Conversation preserved. Continue chatting.[/dim]"
    )
    return model_name, new_agent