"""Model-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

import asyncio
from typing import Any

import ollama
from rich import box
from rich.console import Console
from rich.table import Table

from ..agent import AgentRuntime
from ..core import ModelCapabilityError, model_supports_tools
from ..settings import save_settings

VALID_SAMPLING_PARAMS: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "presence_penalty": float,
    "repeat_penalty": float,
    "repetition_penalty": float,
}


async def _list_models(base_url: str) -> list[Any]:
    """Fetch the list of available Ollama models asynchronously."""
    client = ollama.AsyncClient(host=base_url)
    response = await client.list()
    return list(response.models)


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
            size_str = f"{(item.size / (1024**3)):.1f}GB" if item.size else ""
            console.print(f"  {tool_icon} [cyan]{name}[/cyan] {size_str}{marker}")
        console.print(
            "[dim]─" * 60
            + "[/dim]\n[dim]✓ = supports tools | Use /model set <model> to switch[/dim]"
        )
    except (ollama.ResponseError, OSError) as exc:
        console.print(f"[red]Error listing models: {exc}[/red]")


async def set_model(
    console: Console,
    model_name: str,
    *,
    runtime: AgentRuntime,
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
                "[dim]Use /model to see available models.[/dim]"
            )
            return runtime.settings.model.name
    except (ollama.ResponseError, OSError) as exc:
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
    except (ollama.ResponseError, OSError) as exc:
        console.print(f"[red]Failed to switch to model '{model_name}': {exc}[/red]")
        return current


def show_model_params(console: Console, runtime: AgentRuntime) -> None:
    """Print the active model parameters and their resolution sources."""
    model_name = runtime.settings.model.name
    params = getattr(runtime, "effective_model_params", {})
    if not params:
        console.print(
            f"[yellow]No active parameter data for model '{model_name}'.[/yellow]"
        )
        return

    table = Table(
        title=f"Active Model Parameters: [bold cyan]{model_name}[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Effective Value", justify="right", style="green")
    table.add_column("Resolved From", justify="center")

    source_labels = {
        "user": "[bold yellow]User Config (settings.yaml)[/bold yellow]",
        "modelfile": "[bold cyan]Modelfile / Metadata[/bold cyan]",
        "default": "[dim]Ollama Default[/dim]",
    }

    for name, (val, source) in params.items():
        src_label = source_labels.get(source, source)
        table.add_row(name, str(val), src_label)

    console.print(table)


async def set_model_param(
    console: Console,
    param_name: str,
    value_str: str,
    *,
    runtime: AgentRuntime,
) -> None:
    """Set a model sampling parameter for the active session and save to settings."""
    norm_name = param_name.lower().strip()
    if norm_name == "repetition_penalty":
        norm_name = "repeat_penalty"

    if norm_name not in VALID_SAMPLING_PARAMS:
        valid_list = ", ".join(
            sorted(p for p in VALID_SAMPLING_PARAMS if p != "repetition_penalty")
        )
        console.print(
            f"[red]Unknown parameter '{param_name}'. Valid parameters: {valid_list}[/red]"
        )
        return

    expected_type = VALID_SAMPLING_PARAMS[norm_name]
    try:
        val = expected_type(value_str)
    except ValueError:
        type_name = "integer" if expected_type is int else "float"
        console.print(
            f"[red]Invalid value '{value_str}' for '{norm_name}'. Expected {type_name}.[/red]"
        )
        return

    setattr(runtime.settings.model, norm_name, val)
    await asyncio.to_thread(save_settings, runtime.settings)
    await runtime.reload()
    console.print(
        f"[green]✓ Set [cyan]{norm_name}[/cyan] to [cyan]{val}[/cyan][/green]\n"
        "[dim]Model reloaded with updated parameters.[/dim]"
    )

