"""Model-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import ollama
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ...agent import AgentRuntime
from ...core import (
    ModelCapabilityError,
    create_ollama_async_client,
    model_supports_tools,
)
from ...i18n import _
from ...settings import SETTINGS_PATH, Settings, save_settings


def format_model_size(size: int | None) -> str:
    """Format a model byte size in gigabytes (e.g. '4.7GB'), or '' when unknown."""
    return f"{(size / (1024**3)):.1f}GB" if size else ""


def list_subagents(
    console: Console,
    settings: Settings,
) -> None:
    """List all configured subagents and their properties in a table."""
    if not settings.subagents:
        console.print(
            f"[yellow]{_('No subagents configured.')}[/yellow]\n"
            f"[dim]{_('Configure subagents in {path}', path=SETTINGS_PATH)}[/dim]"
        )
        return

    table = Table(
        title=_("Configured Subagents"),
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column(_("Name"), style="bold cyan", no_wrap=True)
    table.add_column(_("Description"), style="white")
    table.add_column(_("Model"), style="green", no_wrap=True)
    table.add_column(_("Context"), style="yellow", no_wrap=True)
    table.add_column(_("MCP Servers"), style="blue")

    for sa in settings.subagents:
        model_str = escape(sa.model) if sa.model else f"[dim]{escape(settings.model.name)} ({_('inherited')})[/dim]"
        ctx_str = (
            escape(str(sa.context_window))
            if sa.context_window
            else f"[dim]{escape(str(settings.model.context_window))} ({_('inherited')})[/dim]"
        )
        mcp_str = ", ".join(escape(srv.name) for srv in sa.mcp_servers) if sa.mcp_servers else "[dim]-[/dim]"

        table.add_row(
            escape(sa.name),
            escape(sa.description),
            model_str,
            ctx_str,
            mcp_str,
        )

    console.print(table)


async def _list_models(base_url: str) -> list[Any]:
    """Fetch the list of available Ollama models asynchronously."""
    async with create_ollama_async_client(base_url) as client:
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
            size_str = format_model_size(item.size)
            console.print(f"  {tool_icon} [cyan]{escape(item.model)}[/cyan] {size_str}{marker}")
        legend_str = _("supports tools | Use /model set <model> to switch")
        console.print(f"[dim]{'─' * 60}[/dim]\n[dim]✓ = {legend_str}[/dim]")
    except (ollama.ResponseError, OSError) as exc:
        err_msg = _("Error listing models: {exc}", exc=exc)
        console.print(f"[red]{escape(err_msg)}[/red]")


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
            console.print(f"[red]{escape(not_found_msg)}[/red]")
            return None
    except (ollama.ResponseError, OSError) as exc:
        err_msg = _("Error checking model: {exc}", exc=exc)
        console.print(f"[red]{escape(err_msg)}[/red]")
        return None

    current = runtime.settings.model.name
    if model_name == current:
        already_msg = _("Already using model '{model_name}'.", model_name=model_name)
        console.print(f"[yellow]{escape(already_msg)}[/yellow]")
        return current

    try:
        if not await model_supports_tools(model_name, base_url):
            no_tools_msg = _(
                "Model '{model_name}' does not support tools.\nThe agent requires tool support.",
                model_name=model_name,
            )
            console.print(f"[red]{escape(no_tools_msg)}[/red]")
            return None
    except ModelCapabilityError as exc:
        err_msg = _("Cannot verify model capabilities: {exc}", exc=exc)
        console.print(f"[red]{escape(err_msg)}[/red]")
        return None

    try:
        await runtime.set_model(model_name)
        switched_msg = _(
            "Switched from {current} to {model_name}\nConversation preserved. Continue chatting.",
            current=current,
            model_name=model_name,
        )
        console.print(f"[green]✓ {escape(switched_msg)}[/green]")
        return model_name
    except (ollama.ResponseError, OSError) as exc:
        failed_msg = _("Failed to switch to model '{model_name}': {exc}", model_name=model_name, exc=exc)
        console.print(f"[red]{escape(failed_msg)}[/red]")
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
        size_gb = format_model_size(item.size)
        size_str = f" ({size_gb})" if size_gb else ""
        console.print(f"  [cyan]{i})[/cyan] {icon} [bold]{escape(item.model)}[/bold]{size_str}")

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
        console.print(f"[red]{escape(invalid_sel)}[/red]")

    if icons_by_model[selected] == "[red]✗[/red]":
        warn_msg = _(
            "Model '{model_name}' does not support tools.\nThe agent requires tool support.",
            model_name=selected,
        )
        console.print(f"[yellow]{escape(warn_msg)}[/yellow]")

    return selected


def ensure_model_configured(
    settings: Settings,
    console: Console,
    input_func: Callable[[str], str] = input,
) -> str:
    """Ensure the configured model is available in Ollama, or prompt the user to choose one."""
    base_url = settings.model.base_url
    try:
        available_models = asyncio.run(_list_models(base_url))
    except (ollama.ResponseError, OSError) as exc:
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

    if configured:
        not_avail_msg = _("Configured model '{configured}' is not available in Ollama.", configured=configured)
        console.print(f"[yellow]{escape(not_avail_msg)}[/yellow]")
    else:
        console.print(f"[yellow]{_('No model is currently configured in settings.')}[/yellow]")

    selected = _prompt_user_select_model(available_models, base_url, console, input_func)

    settings.model.name = selected
    save_settings(settings)
    saved_msg = _("Selected model '{selected}' saved to configuration.", selected=selected)
    console.print(f"[green]✓ {escape(saved_msg)}[/green]\n")
    return selected
