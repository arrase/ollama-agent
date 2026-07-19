"""Model-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import ollama
from rich.console import Console

from ..core import ModelCapabilityError, model_supports_tools

if TYPE_CHECKING:
    from ..agent import AgentRuntime


async def _list_models(base_url: str) -> list[Any]:
    """Fetch the list of available Ollama models asynchronously."""
    client = ollama.AsyncClient(host=base_url)
    response = await client.list()
    return getattr(response, "models", [])


async def list_models(
    console: Console,
    current_model: str,
    base_url: str,
) -> None:
    """Print available Ollama models with tool-support indicators."""
    try:
        models = await _list_models(base_url)
        if not models:
            console.print("[yellow]No models found in Ollama.[/yellow]")
            return
        
        valid_models = [m for m in models if m.model]
        
        async def get_tool_icon(model_name: str) -> str:
            try:
                supported = await model_supports_tools(model_name, base_url)
                return "[green]✓[/green]" if supported else "[red]✗[/red]"
            except ModelCapabilityError:
                return "[yellow]?[/yellow]"

        tool_icons = await asyncio.gather(
            *(get_tool_icon(m.model) for m in valid_models)
        )

        console.print("[bold]Available Models:[/bold]\n[dim]─" + "─" * 59 + "[/dim]")
        for item, tool_icon in zip(valid_models, tool_icons):
            name = item.model
            marker = " [green]◀ current[/green]" if name == current_model else ""
            size_gb = getattr(item, "size", 0) / (1024**3)
            size_str = f"{size_gb:.1f}GB" if size_gb else ""
            console.print(f"  {tool_icon} [cyan]{name}[/cyan] {size_str}{marker}")
        console.print(
            "[dim]─" * 60
            + "[/dim]\n[dim]✓ = supports tools | Use /model-set <model> to switch[/dim]"
        )
    except Exception as exc:
        console.print(f"[red]Error listing models: {exc}[/red]")


async def set_model(
    console: Console,
    model_name: str,
    *,
    runtime: "AgentRuntime",
) -> str:
    """Switch to model_name, returning the new model name."""
    try:
        base_url = runtime.settings.model.base_url
        available = {
            model.model for model in await _list_models(base_url)
        }
        if model_name not in available:
            console.print(
                f"[red]Model '{model_name}' not found.[/red]\n"
                "[dim]Use /models to see available models.[/dim]"
            )
            return runtime.settings.model.name
    except Exception as exc:
        console.print(f"[red]Error checking model: {exc}[/red]")
        return runtime.settings.model.name

    current = runtime.settings.model.name
    if model_name == current:
        console.print(f"[yellow]Already using model '{model_name}'.[/yellow]")
        return current

    try:
        if not await model_supports_tools(model_name, base_url):
            console.print(
                f"[red]Model '{model_name}' does not support tools.[/red]\n"
                "[dim]The agent requires tool support.[/dim]"
            )
            return current
    except ModelCapabilityError as exc:
        console.print(f"[red]Cannot verify model capabilities: {exc}[/red]")
        return current

    try:
        await runtime.set_model(model_name)
        console.print(
            f"[green]✓ Switched from [cyan]{current}[/cyan] to [cyan]{model_name}[/cyan][/green]\n"
            "[dim]Conversation preserved. Continue chatting.[/dim]"
        )
        return model_name
    except Exception as exc:
        console.print(f"[red]Failed to switch to model '{model_name}': {exc}[/red]")
        return current
