"""Model-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
import ollama
from rich.console import Console

from ...agent import AgentRuntime
from ...core import (
    ModelCapabilityError,
    model_supports_tools,
)
from ...i18n import _
from ...settings import Settings, save_settings


async def _list_models(base_url: str) -> list[Any]:
    """Fetch the list of available Ollama models asynchronously."""
    client = ollama.AsyncClient(host=base_url)
    response = await client.list()
    return list(response.models)


async def _tool_icon(model_name: str, base_url: str) -> str:
    try:
        supported = await model_supports_tools(model_name, base_url)
        return "[green]✓[/green]" if supported else "[red]✗[/red]"
    except ModelCapabilityError:
        return "[yellow]?[/yellow]"


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

        tool_icons = await asyncio.gather(*(_tool_icon(m.model, base_url) for m in models))

        console.print(f"[bold]{_('Available Models:')}[/bold]\n[dim]{'─' * 60}[/dim]")
        for item, tool_icon in zip(models, tool_icons, strict=True):
            marker = f" [green]◀ {_('current')}[/green]" if item.model == current_model else ""
            size_str = f"{(item.size / (1024**3)):.1f}GB" if item.size else ""
            console.print(f"  {tool_icon} [cyan]{item.model}[/cyan] {size_str}{marker}")
        legend_str = _("supports tools | Use /model set <model> to switch")
        console.print(f"[dim]{'─' * 60}[/dim]\n[dim]✓ = {legend_str}[/dim]")
    except (ollama.ResponseError, OSError) as exc:
        err_msg = _("Error listing models: {exc}", exc=exc)
        console.print(f"[red]{err_msg}[/red]")


async def set_model(
    console: Console,
    model_name: str,
    *,
    runtime: AgentRuntime,
) -> str | None:
    """Switch to model_name, returning the new model name."""
    base_url = runtime.settings.model.base_url
    try:
        available = {model.model for model in await _list_models(base_url)}
        if model_name not in available:
            not_found_msg = _(
                "Model '{model_name}' not found.\nUse /model to see available models.",
                model_name=model_name,
            )
            console.print(f"[red]{not_found_msg}[/red]")
            return None
    except (ollama.ResponseError, OSError) as exc:
        err_msg = _("Error checking model: {exc}", exc=exc)
        console.print(f"[red]{err_msg}[/red]")
        return None

    current = runtime.settings.model.name
    if model_name == current:
        already_msg = _("Already using model '{model_name}'.", model_name=model_name)
        console.print(f"[yellow]{already_msg}[/yellow]")
        return current

    try:
        if not await model_supports_tools(model_name, base_url):
            no_tools_msg = _(
                "Model '{model_name}' does not support tools.\nThe agent requires tool support.",
                model_name=model_name,
            )
            console.print(f"[red]{no_tools_msg}[/red]")
            return None
    except ModelCapabilityError as exc:
        err_msg = _("Cannot verify model capabilities: {exc}", exc=exc)
        console.print(f"[red]{err_msg}[/red]")
        return None

    try:
        await runtime.set_model(model_name)
        switched_msg = _(
            "Switched from {current} to {model_name}\nConversation preserved. Continue chatting.",
            current=current,
            model_name=model_name,
        )
        console.print(f"[green]✓ {switched_msg}[/green]")
        return model_name
    except (ollama.ResponseError, OSError) as exc:
        failed_msg = _("Failed to switch to model '{model_name}': {exc}", model_name=model_name, exc=exc)
        console.print(f"[red]{failed_msg}[/red]")
        return None


def _prompt_user_select_model(
    available_models: list[Any],
    base_url: str,
    console: Console,
    input_func: Callable[[str], str],
) -> str:
    async def _fetch_tool_icons() -> list[str]:
        return await asyncio.gather(*(_tool_icon(m.model, base_url) for m in available_models))

    tool_icons = asyncio.run(_fetch_tool_icons())
    icons_by_model = {m.model: icon for m, icon in zip(available_models, tool_icons, strict=True)}

    console.print(f"[bold]{_('Available Ollama models:')}[/bold]")
    for i, item in enumerate(available_models, start=1):
        icon = icons_by_model[item.model]
        size_str = f" ({item.size / (1024**3):.1f} GB)" if item.size else ""
        console.print(f"  [cyan]{i})[/cyan] {icon} [bold]{item.model}[/bold]{size_str}")

    while True:
        try:
            choice = input_func(_("Select a model [1-{count}]: ", count=len(available_models))).strip()
        except (KeyboardInterrupt, EOFError):
            raise SystemExit(1) from None
        if not choice:
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(available_models):
            selected = available_models[int(choice) - 1].model
            break
        selected = next((m.model for m in available_models if m.model in (choice, f"{choice}:latest")), None)
        if selected:
            break
        invalid_sel = _(
            "Invalid selection '{choice}'. Please enter a number between 1 and {count} or a model name.",
            choice=choice,
            count=len(available_models),
        )
        console.print(f"[red]{invalid_sel}[/red]")

    if icons_by_model.get(selected) == "[red]✗[/red]":
        warn_msg = _(
            "Model '{model_name}' does not support tools.\nThe agent requires tool support.",
            model_name=selected,
        )
        console.print(f"[yellow]{warn_msg}[/yellow]")

    return selected


def ensure_model_configured(
    settings: Settings,
    console: Console | None = None,
    input_func: Callable[[str], str] = input,
) -> str:
    """Ensure the configured model is available in Ollama, or prompt the user to choose one."""
    base_url = settings.model.base_url
    try:
        available_models = asyncio.run(_list_models(base_url))
    except (httpx.HTTPError, ollama.ResponseError, OSError) as exc:
        raise ModelCapabilityError(
            _("Could not connect to Ollama at '{base_url}': {exc}", base_url=base_url, exc=exc)
        ) from exc

    if not available_models:
        raise ModelCapabilityError(
            _(
                "No models found in Ollama at '{base_url}'. Please pull a model first with 'ollama pull <model>'.",
                base_url=base_url,
            )
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

    if console is None:
        console = Console()

    if configured:
        not_avail_msg = _("Configured model '{configured}' is not available in Ollama.", configured=configured)
        console.print(f"[yellow]{not_avail_msg}[/yellow]")
    else:
        console.print(f"[yellow]{_('No model is currently configured in settings.')}[/yellow]")

    selected = _prompt_user_select_model(available_models, base_url, console, input_func)

    settings.model.name = selected
    save_settings(settings)
    saved_msg = _("Selected model '{selected}' saved to configuration.", selected=selected)
    console.print(f"[green]✓ {saved_msg}[/green]\n")
    return selected
