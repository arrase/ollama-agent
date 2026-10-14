"""Model parameter, effort, and context window commands."""

from __future__ import annotations

import asyncio

import ollama
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ...agent import AgentRuntime
from ...core import (
    ModelContextWindowError,
    get_model_thinking_config,
    parse_context_window_input,
    validate_and_normalize_effort,
)
from ...i18n import _
from ...settings import save_settings

VALID_SAMPLING_PARAMS: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "presence_penalty": float,
    "repeat_penalty": float,
    "repetition_penalty": float,
}

# Ollama accepts no meaningful value outside these ranges; accepting e.g.
# `temperature 99` would be persisted to settings.yaml and sent on every request.
SAMPLING_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0.0, 100_000.0),
    "min_p": (0.0, 1.0),
    "presence_penalty": (-2.0, 2.0),
    "repeat_penalty": (0.0, 2.0),
}


def show_effort(console: Console, runtime: AgentRuntime) -> None:
    """Print the current reasoning effort and model."""
    effort = runtime.settings.model.reasoning_effort
    model = runtime.settings.model.name
    thinking_cfg = get_model_thinking_config(runtime.model.show_info) if runtime.model else None

    effort_display = effort
    if thinking_cfg and effort == "default" and "default" in thinking_cfg:
        effort_display = f"{effort} (effective: {thinking_cfg['default']})"

    console.print(
        _(
            "Current reasoning effort: {effort} (model: {model})\nUsage: /effort <level> | default",
            effort=effort_display,
            model=model,
        )
    )
    if thinking_cfg and "values" in thinking_cfg and "default" in thinking_cfg:
        console.print(
            _(
                "Model thinking controls: {values} (default: {default})",
                values=thinking_cfg["values"],
                default=thinking_cfg["default"],
            )
        )


async def set_effort(
    console: Console,
    effort: str,
    *,
    runtime: AgentRuntime,
) -> str | None:
    """Switch reasoning effort level, returning the new effort level."""
    thinking_cfg = get_model_thinking_config(runtime.model.show_info) if runtime.model else None
    validated, err_msg = validate_and_normalize_effort(effort, thinking_cfg, runtime.settings.model.name)
    if validated is None:
        console.print(f"[red]{escape(str(err_msg))}[/red]")
        return None
    norm_effort = validated

    current = runtime.settings.model.reasoning_effort
    if norm_effort == current:
        already_msg = _("Already using reasoning effort '{norm_effort}'.", norm_effort=norm_effort)
        console.print(f"[yellow]{escape(already_msg)}[/yellow]")
        return current

    try:
        await runtime.set_reasoning_effort(norm_effort)
        switched_msg = _(
            "Switched reasoning effort from {current} to {norm_effort}\nConversation preserved. Continue chatting.",
            current=current,
            norm_effort=norm_effort,
        )
        console.print(f"[green]✓ {escape(switched_msg)}[/green]")
        return norm_effort
    except (ollama.ResponseError, OSError, ValueError) as exc:
        failed_msg = _(
            "Failed to switch reasoning effort to '{norm_effort}': {exc}",
            norm_effort=norm_effort,
            exc=exc,
        )
        console.print(f"[red]{escape(failed_msg)}[/red]")
        return None


def show_context_window(console: Console, runtime: AgentRuntime) -> None:
    """Print the current context window size and model."""
    ctx = runtime.settings.model.context_window
    effective = runtime.effective_context_window
    model = runtime.settings.model.name
    if effective and str(effective) != str(ctx):
        console.print(
            _(
                "Current context window: {ctx} (effective: {effective} tokens, model: {model})\n"
                "Usage: /context <size|max> (e.g. 8192, 16384, 32768, max)",
                ctx=ctx,
                effective=effective,
                model=model,
            )
        )
    else:
        console.print(
            _(
                "Current context window: {ctx} (model: {model})\n"
                "Usage: /context <size|max> (e.g. 8192, 16384, 32768, max)",
                ctx=ctx,
                model=model,
            )
        )


async def set_context_window(
    console: Console,
    context_window: str,
    *,
    runtime: AgentRuntime,
) -> str | None:
    """Switch context window size, returning the new context window."""
    try:
        val = parse_context_window_input(context_window)
    except ModelContextWindowError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return None

    current = runtime.settings.model.context_window
    if val == current:
        already_msg = _("Already using context window '{val}'.", val=val)
        console.print(f"[yellow]{escape(already_msg)}[/yellow]")
        return str(current)

    try:
        await runtime.set_context_window(val)
        eff = runtime.effective_context_window
        eff_info = f" ({eff} {_('tokens')})" if eff and str(eff) != str(val) else ""
        switched_msg = _(
            "Switched context window from {current} to {val}{eff_info}\nConversation preserved. Continue chatting.",
            current=current,
            val=val,
            eff_info=eff_info,
        )
        console.print(f"[green]✓ {escape(switched_msg)}[/green]")
        return str(val)
    except (ollama.ResponseError, OSError, ValueError, ModelContextWindowError) as exc:
        failed_msg = _("Failed to switch context window to '{val}': {exc}", val=val, exc=exc)
        console.print(f"[red]{escape(failed_msg)}[/red]")
        return None


def show_model_params(console: Console, runtime: AgentRuntime) -> None:
    """Print the active model parameters and their resolution sources."""
    model_name = runtime.settings.model.name
    params = runtime.effective_model_params
    if not params:
        no_params_msg = _("No active parameter data for model '{model_name}'.", model_name=model_name)
        console.print(f"[yellow]{escape(no_params_msg)}[/yellow]")
        return

    table_title = _("Active Model Parameters: {model_name}", model_name=model_name)
    table = Table(
        title=escape(table_title),
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column(_("Parameter"), style="cyan")
    table.add_column(_("Effective Value"), justify="right", style="green")
    table.add_column(_("Resolved From"), justify="center")

    user_label = _("User Config (settings.yaml)")
    meta_label = _("Modelfile / Metadata")

    source_labels = {
        "user": f"[bold yellow]{user_label}[/bold yellow]",
        "modelfile": f"[bold cyan]{meta_label}[/bold cyan]",
    }

    for name, (val, source) in params.items():
        table.add_row(name, str(val), source_labels[source])

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
        valid_list = ", ".join(sorted(p for p in VALID_SAMPLING_PARAMS if p != "repetition_penalty"))
        unknown_msg = _(
            "Unknown parameter '{param_name}'. Valid parameters: {valid_list}",
            param_name=param_name,
            valid_list=valid_list,
        )
        console.print(f"[red]{escape(unknown_msg)}[/red]")
        return

    expected_type = VALID_SAMPLING_PARAMS[norm_name]
    try:
        val = expected_type(value_str)
    except ValueError:
        type_name = "integer" if expected_type is int else "float"
        invalid_msg = _(
            "Invalid value '{value_str}' for '{norm_name}'. Expected {type_name}.",
            value_str=value_str,
            norm_name=norm_name,
            type_name=type_name,
        )
        console.print(f"[red]{escape(invalid_msg)}[/red]")
        return

    low, high = SAMPLING_PARAM_BOUNDS[norm_name]
    if not (low <= val <= high):
        out_of_range = _(
            "'{norm_name}' must be between {low} and {high}.",
            norm_name=norm_name,
            low=low,
            high=high,
        )
        console.print(f"[red]{escape(out_of_range)}[/red]")
        return

    setattr(runtime.settings.model, norm_name, val)
    await asyncio.to_thread(save_settings, runtime.settings)
    await runtime.reload()
    success_msg = _("Set {norm_name} to {val}\nModel reloaded with updated parameters.", norm_name=norm_name, val=val)
    console.print(f"[green]✓ {escape(success_msg)}[/green]")
