"""Model-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import ollama
from rich import box
from rich.console import Console
from rich.table import Table

from ..agent import AgentRuntime
from ..core import ALLOWED_REASONING_EFFORTS, ModelCapabilityError, model_supports_tools
from ..i18n import _
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
            console.print(f"[yellow]{_('No models found in Ollama.')}[/yellow]")
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

        console.print(f"[bold]{_('Available Models:')}[/bold]\n[dim]─" + "─" * 59 + "[/dim]")
        for item, tool_icon in zip(valid_models, tool_icons):
            name = item.model
            marker = f" [green]◀ {_('current')}[/green]" if name == current_model else ""
            size_str = f"{(item.size / (1024**3)):.1f}GB" if item.size else ""
            console.print(f"  {tool_icon} [cyan]{name}[/cyan] {size_str}{marker}")
        legend_str = _("supports tools | Use /model set <model> to switch")
        console.print(f"[dim]─" * 60 + f"[/dim]\n[dim]✓ = {legend_str}[/dim]")
    except (ollama.ResponseError, OSError) as exc:
        err_msg = _("Error listing models: {exc}", exc=exc)
        console.print(f"[red]{err_msg}[/red]")


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
            not_found_msg = _("Model '{model_name}' not found.\nUse /model to see available models.", model_name=model_name)
            console.print(f"[red]{not_found_msg}[/red]")
            return runtime.settings.model.name
    except (ollama.ResponseError, OSError) as exc:
        err_msg = _("Error checking model: {exc}", exc=exc)
        console.print(f"[red]{err_msg}[/red]")
        return runtime.settings.model.name

    current = runtime.settings.model.name
    if model_name == current:
        already_msg = _("Already using model '{model_name}'.", model_name=model_name)
        console.print(f"[yellow]{already_msg}[/yellow]")
        return current

    try:
        if not await model_supports_tools(model_name, base_url):
            no_tools_msg = _("Model '{model_name}' does not support tools.\nThe agent requires tool support.", model_name=model_name)
            console.print(f"[red]{no_tools_msg}[/red]")
            return current
    except ModelCapabilityError as exc:
        err_msg = _("Cannot verify model capabilities: {exc}", exc=exc)
        console.print(f"[red]{err_msg}[/red]")
        return current

    try:
        await runtime.set_model(model_name)
        switched_msg = _("Switched from {current} to {model_name}\nConversation preserved. Continue chatting.", current=current, model_name=model_name)
        console.print(f"[green]✓ {switched_msg}[/green]")
        return model_name
    except (ollama.ResponseError, OSError) as exc:
        failed_msg = _("Failed to switch to model '{model_name}': {exc}", model_name=model_name, exc=exc)
        console.print(f"[red]{failed_msg}[/red]")
        return current


def show_effort(console: Console, runtime: AgentRuntime) -> None:
    """Print the current reasoning effort and model."""
    effort = runtime.settings.model.reasoning_effort
    model = runtime.settings.model.name
    console.print(
        _("Current reasoning effort: {effort} (model: {model})\nUsage: /effort <level> (e.g. low, medium, high, disabled, hide, enabled)", effort=effort, model=model)
    )


async def set_effort(
    console: Console,
    effort: str,
    *,
    runtime: AgentRuntime,
) -> str:
    """Switch reasoning effort level, returning the new effort level."""
    norm_effort = effort.lower().strip()
    if norm_effort not in ALLOWED_REASONING_EFFORTS:
        valid_list = ", ".join(ALLOWED_REASONING_EFFORTS)
        err_msg = _("Invalid reasoning effort '{effort}'. Allowed values: {valid_list}", effort=effort, valid_list=valid_list)
        console.print(f"[red]{err_msg}[/red]")
        return runtime.settings.model.reasoning_effort

    current = runtime.settings.model.reasoning_effort
    if norm_effort == current:
        already_msg = _("Already using reasoning effort '{norm_effort}'.", norm_effort=norm_effort)
        console.print(f"[yellow]{already_msg}[/yellow]")
        return current

    try:
        await runtime.set_reasoning_effort(norm_effort)
        switched_msg = _("Switched reasoning effort from {current} to {norm_effort}\nConversation preserved. Continue chatting.", current=current, norm_effort=norm_effort)
        console.print(f"[green]✓ {switched_msg}[/green]")
        return norm_effort
    except (ollama.ResponseError, OSError, ValueError) as exc:
        failed_msg = _("Failed to switch reasoning effort to '{norm_effort}': {exc}", norm_effort=norm_effort, exc=exc)
        console.print(f"[red]{failed_msg}[/red]")
        return current


def show_model_params(console: Console, runtime: AgentRuntime) -> None:
    """Print the active model parameters and their resolution sources."""
    model_name = runtime.settings.model.name
    params = runtime.effective_model_params
    if not params:
        no_params_msg = _("No active parameter data for model '{model_name}'.", model_name=model_name)
        console.print(f"[yellow]{no_params_msg}[/yellow]")
        return

    table_title = _("Active Model Parameters: {model_name}", model_name=model_name)
    table = Table(
        title=table_title,
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column(_("Parameter"), style="cyan")
    table.add_column(_("Effective Value"), justify="right", style="green")
    table.add_column(_("Resolved From"), justify="center")

    user_label = _("User Config (settings.yaml)")
    meta_label = _("Modelfile / Metadata")
    default_label = _("Ollama Default")

    source_labels = {
        "user": f"[bold yellow]{user_label}[/bold yellow]",
        "modelfile": f"[bold cyan]{meta_label}[/bold cyan]",
        "default": f"[dim]{default_label}[/dim]",
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
        unknown_msg = _("Unknown parameter '{param_name}'. Valid parameters: {valid_list}", param_name=param_name, valid_list=valid_list)
        console.print(f"[red]{unknown_msg}[/red]")
        return

    expected_type = VALID_SAMPLING_PARAMS[norm_name]
    try:
        val = expected_type(value_str)
    except ValueError:
        type_name = "integer" if expected_type is int else "float"
        invalid_msg = _("Invalid value '{value_str}' for '{norm_name}'. Expected {type_name}.", value_str=value_str, norm_name=norm_name, type_name=type_name)
        console.print(f"[red]{invalid_msg}[/red]")
        return

    setattr(runtime.settings.model, norm_name, val)
    await asyncio.to_thread(save_settings, runtime.settings)
    await runtime.reload()
    success_msg = _("Set {norm_name} to {val}\nModel reloaded with updated parameters.", norm_name=norm_name, val=val)
    console.print(f"[green]✓ {success_msg}[/green]")


def ensure_model_configured(
    settings: Settings,
    console: Console | None = None,
    input_func: Callable[[str], str] = input,
) -> str:
    """Ensure the configured model is available in Ollama, or prompt the user to choose one."""
    base_url = settings.model.base_url
    try:
        models = _list_models_sync(base_url)
        available_models = [m for m in models if m.model]
    except Exception as exc:
        raise ModelCapabilityError(
            _("Could not connect to Ollama at '{base_url}': {exc}", base_url=base_url, exc=exc)
        ) from exc

    if not available_models:
        raise ModelCapabilityError(
            _("No models found in Ollama at '{base_url}'. Please pull a model first with 'ollama pull <model>'.", base_url=base_url)
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
        not_avail_msg = _("Configured model '{configured}' is not available in Ollama.", configured=configured)
        out.print(f"[yellow]{not_avail_msg}[/yellow]")
    else:
        out.print(f"[yellow]{_('No model is currently configured in settings.')}[/yellow]")

    out.print(f"[bold]{_('Available Ollama models:')}[/bold]")
    for i, item in enumerate(available_models, start=1):
        size_str = f" ({item.size / (1024**3):.1f} GB)" if item.size else ""
        out.print(f"  [cyan]{i})[/cyan] [bold]{item.model}[/bold]{size_str}")

    while True:
        try:
            choice = input_func(
                _("Select a model [1-{count}]: ", count=len(available_models))
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
        invalid_sel = _("Invalid selection '{choice}'. Please enter a number between 1 and {count} or a model name.", choice=choice, count=len(available_models))
        out.print(f"[red]{invalid_sel}[/red]")

    settings.model.name = selected
    save_settings(settings)
    saved_msg = _("Selected model '{selected}' saved to configuration.", selected=selected)
    out.print(f"[green]✓ {saved_msg}[/green]\n")
    return selected
