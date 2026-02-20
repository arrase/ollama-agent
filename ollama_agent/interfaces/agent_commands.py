"""High-level agent action functions shared by CLI and REPL.

Extracting these from :class:`~ollama_agent.interfaces.repl.OllamaREPL`
decouples the business logic (session management, model switching) from the
interactive loop, making the same operations trivially reusable from the CLI
or future interfaces, and reducing ``repl.py`` to a thin orchestration layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import ollama
from rich.console import Console

from ..core import ModelCapabilityError, model_supports_tools, resolve_unique_prefix

if TYPE_CHECKING:
    from ..agent import OllamaAgent


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def list_sessions(
    agent: "OllamaAgent",
    console: Console,
    *,
    page: int = 1,
    per_page: int = 10,
) -> None:
    """Print a paginated list of saved sessions to *console*."""
    sessions = agent.session_manager.list_sessions(
        limit=per_page, offset=(page - 1) * per_page
    )
    if not sessions:
        console.print(
            f"[yellow]{'No saved sessions found.' if page == 1 else f'No more sessions (page {page} is empty).'}[/yellow]"
        )
        return
    current_id = agent.session_manager.get_session_id()
    console.print(f"[bold]Sessions (page {page}):[/bold]\n[dim]─" + "─" * 59 + "[/dim]")
    for s in sessions:
        marker = " [green]◀ current[/green]" if s["session_id"] == current_id else ""
        preview = s["preview"][:40] + "..." if len(s["preview"]) > 40 else s["preview"]
        console.print(
            f"[cyan]{s['session_id'][:8]}[/cyan] │ {s['message_count']:>3} msgs │"
            f" {s['last_message'][:16]} │ [dim]{preview}[/dim]{marker}"
        )
    console.print("[dim]─" * 60 + "[/dim]")
    hint = f"Next page: /sessions {page + 1} | " if len(sessions) == per_page else ""
    console.print(f"[dim]{hint}Load: /session-load <id>[/dim]")


def find_session(
    prefix: str,
    sessions: list[dict[str, Any]],
    console: Console,
) -> dict[str, Any] | None:
    """Resolve a session by id prefix.

    Returns the matching session dict, or ``None`` after printing an error.
    """
    ids = [s.get("session_id", "") for s in sessions if isinstance(s, dict)]
    resolved = resolve_unique_prefix(prefix, ids)
    if resolved:
        return next((s for s in sessions if s.get("session_id") == resolved), None)

    matches = [s for s in sessions if str(s.get("session_id", "")).startswith(prefix)]
    if not matches:
        console.print(f"[red]No session found matching '{prefix}'[/red]")
        return None
    console.print(f"[yellow]Multiple sessions match '{prefix}':[/yellow]")
    for m in matches[:5]:
        sid = str(m.get("session_id", ""))
        preview = str(m.get("preview", ""))
        console.print(f"  [cyan]{sid[:8]}[/cyan] - {preview[:40]}")
    return None


def load_session(agent: "OllamaAgent", console: Console, session_id_prefix: str) -> None:
    """Load a session matching *session_id_prefix* and print a summary."""
    target = find_session(
        session_id_prefix, agent.session_manager.list_sessions(limit=100), console
    )
    if not target:
        return
    history = agent.session_manager.get_readable_history(target["session_id"])
    agent.session_manager.load_session(target["session_id"])
    console.print(
        f"\n[bold green]━━━ Session Loaded: {target['session_id'][:8]}... ━━━[/bold green]"
    )
    console.print(
        f"[dim]Messages: {target['message_count']} | Last active: {target['last_message']}[/dim]\n"
    )
    if history:
        console.print("[bold]Conversation History:[/bold]\n[dim]─" + "─" * 49 + "[/dim]")
        for msg in history[-10:]:
            limit = 200 if msg["role"] == "user" else 300
            content = (
                msg["content"][:limit] + "..."
                if len(msg["content"]) > limit
                else msg["content"]
            )
            prefix_str = "[bold blue]>>>[/bold blue] " if msg["role"] == "user" else ""
            console.print(f"{prefix_str}{content}\n")
        if len(history) > 10:
            console.print(f"[dim]... and {len(history) - 10} earlier messages[/dim]\n")
        console.print("[dim]─" * 50 + "[/dim]")
    console.print(
        "[green]✓ Session loaded. Continue typing to resume the conversation.[/green]\n"
    )


def delete_session(agent: "OllamaAgent", console: Console, session_id_prefix: str) -> None:
    """Delete a session matching *session_id_prefix*."""
    target = find_session(
        session_id_prefix, agent.session_manager.list_sessions(limit=100), console
    )
    if not target:
        return
    msg = (
        f"[green]✓ Deleted session:[/green] [cyan]{target['session_id'][:8]}[/cyan]"
        if agent.session_manager.delete_session(target["session_id"])
        else "[red]Failed to delete session[/red]"
    )
    console.print(msg)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


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
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")


async def set_model(
    console: Console,
    model_name: str,
    *,
    current_model: str,
    current_effort: str,
    active_agent: "OllamaAgent | None",
    agent_factory: Callable[..., "OllamaAgent"],
) -> tuple[str, "OllamaAgent | None"]:
    """Switch to *model_name*, returning ``(new_model, new_agent)``.

    On failure the original model and a freshly-created agent are returned so
    the caller can continue without interruption.  The caller is responsible
    for updating its own ``model`` / ``active_agent`` attributes from the
    return value.
    """
    try:
        available = {
            getattr(m, "model", "") for m in getattr(ollama.list(), "models", [])
        }
        if model_name not in available:
            console.print(
                f"[red]Model '{model_name}' not found.[/red]\n"
                "[dim]Use /models to see available models.[/dim]"
            )
            return current_model, active_agent
    except Exception as e:
        console.print(f"[red]Error checking model: {e}[/red]")
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
    except ModelCapabilityError as e:
        console.print(f"[red]Cannot verify model capabilities: {e}[/red]")
        return current_model, active_agent

    old_model = current_model
    old_session_id = (
        active_agent.session_manager.get_session_id() if active_agent else None
    )

    if active_agent:
        await active_agent.cleanup()

    try:
        new_agent: "OllamaAgent" = agent_factory(
            model=model_name, reasoning_effort=current_effort
        )
        await new_agent.initialize()
    except (ModelCapabilityError, SystemExit) as e:
        console.print(
            f"[red]Failed to create agent with model '{model_name}': {e}[/red]"
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
