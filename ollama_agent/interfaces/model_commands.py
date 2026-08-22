"""Model-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import ollama
from rich import box
from rich.console import Console
from rich.table import Table

from ..agent import AgentRuntime
from ..core import ModelCapabilityError, model_supports_tools
from ..settings import Settings, save_settings

VALID_SAMPLING_PARAMS: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "presence_penalty": float,
    "repeat_penalty": float,
    "repetition_penalty": float,
}


def _list_models_sync(base_url: str) -> list[Any]:
    """Fetch the list of available Ollama models synchronously."""
    client = ollama.Client(host=base_url)
    response = client.list()
    return list(response.models)


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
    params = runtime.effective_model_params
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


def ensure_model_configured(
    settings: Settings,
    console: Console | None = None,
    input_func: Callable[[str], str] = input,
) -> str:
    """Ensure the configured model is available in Ollama, or prompt the user to choose one."""
    base_url = settings.model.base_url
    try:
        models = _list_models_sync(base_url)
        available_models = [m for m in models if getattr(m, "model", None)]
    except Exception as exc:
        raise ModelCapabilityError(
            f"Could not connect to Ollama at '{base_url}': {exc}"
        ) from exc

    if not available_models:
        raise ModelCapabilityError(
            f"No models found in Ollama at '{base_url}'. Please pull a model first with 'ollama pull <model>'."
        )

    model_names = [m.model for m in available_models]
    configured = settings.model.name.strip()
    if configured:
        if configured in model_names:
            return configured
        if f"{configured}:latest" in model_names:
            settings.model.name = f"{configured}:latest"
            save_settings(settings)
            return settings.model.name

    out = console if console is not None else Console()
    if configured:
        out.print(
            f"[yellow]Configured model '[bold cyan]{configured}[/bold cyan]' is not available in Ollama.[/yellow]"
        )
    else:
        out.print("[yellow]No model is currently configured in settings.[/yellow]")

    out.print("[bold]Available Ollama models:[/bold]")
    for i, item in enumerate(available_models, start=1):
        size_str = (
            f" ({item.size / (1024**3):.1f} GB)"
            if getattr(item, "size", None)
            else ""
        )
        out.print(f"  [cyan]{i})[/cyan] [bold]{item.model}[/bold]{size_str}")

    while True:
        try:
            choice = input_func(
                f"Select a model [1-{len(available_models)}]: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            raise SystemExit(1)
        if not choice:
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                selected = available_models[idx].model
                break
        else:
            matched = next(
                (
                    m.model
                    for m in available_models
                    if m.model == choice or m.model == f"{choice}:latest"
                ),
                None,
            )
            if matched is not None:
                selected = matched
                break
        out.print(
            f"[red]Invalid selection '{choice}'. Please enter a number between 1 and {len(available_models)} or a model name.[/red]"
        )

    settings.model.name = selected
    save_settings(settings)
    out.print(
        f"[green]✓ Selected model '[bold cyan]{selected}[/bold cyan]' saved to configuration.[/green]\n"
    )
    return selected


