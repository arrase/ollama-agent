"""Session-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from rich.console import Console
from ..core import resolve_unique_prefix

if TYPE_CHECKING:
    from ..agent import OllamaAgent


def list_sessions(
    agent: "OllamaAgent",
    console: Console,
    *,
    page: int = 1,
    per_page: int = 10,
) -> None:
    """Print a paginated list of saved sessions to console."""
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
    for session in sessions:
        marker = " [green]◀ current[/green]" if session["session_id"] == current_id else ""
        preview = session["preview"][:40] + "..." if len(session["preview"]) > 40 else session["preview"]
        console.print(
            f"[cyan]{session['session_id'][:8]}[/cyan] │ {session['message_count']:>3} msgs │"
            f" {session['last_message'][:16]} │ [dim]{preview}[/dim]{marker}"
        )
    console.print("[dim]─" * 60 + "[/dim]")
    hint = f"Next page: /sessions {page + 1} | " if len(sessions) == per_page else ""
    console.print(f"[dim]{hint}Load: /session-load <id>[/dim]")


def find_session(
    prefix: str,
    sessions: list[dict[str, Any]],
    console: Console,
) -> dict[str, Any] | None:
    """Resolve a session by id prefix."""
    ids = [session.get("session_id", "") for session in sessions if isinstance(session, dict)]
    resolved = resolve_unique_prefix(prefix, ids)
    if resolved:
        return next((session for session in sessions if session.get("session_id") == resolved), None)

    matches = [session for session in sessions if str(session.get("session_id", "")).startswith(prefix)]
    if not matches:
        console.print(f"[red]No session found matching '{prefix}'[/red]")
        return None
    console.print(f"[yellow]Multiple sessions match '{prefix}':[/yellow]")
    for match in matches[:5]:
        session_id = str(match.get("session_id", ""))
        preview = str(match.get("preview", ""))
        console.print(f"  [cyan]{session_id[:8]}[/cyan] - {preview[:40]}")
    return None


def load_session(agent: "OllamaAgent", console: Console, session_id_prefix: str) -> None:
    """Load a session matching session_id_prefix and print a summary."""
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
        for message in history[-10:]:
            limit = 200 if message["role"] == "user" else 300
            content = (
                message["content"][:limit] + "..."
                if len(message["content"]) > limit
                else message["content"]
            )
            prefix_str = "[bold blue]>>>[/bold blue] " if message["role"] == "user" else ""
            console.print(f"{prefix_str}{content}\n")
        if len(history) > 10:
            console.print(f"[dim]... and {len(history) - 10} earlier messages[/dim]\n")
        console.print("[dim]─" * 50 + "[/dim]")
    console.print(
        "[green]✓ Session loaded. Continue typing to resume the conversation.[/green]\n"
    )


def delete_session(agent: "OllamaAgent", console: Console, session_id_prefix: str) -> None:
    """Delete a session matching session_id_prefix."""
    target = find_session(
        session_id_prefix, agent.session_manager.list_sessions(limit=100), console
    )
    if not target:
        return
    message = (
        f"[green]✓ Deleted session:[/green] [cyan]{target['session_id'][:8]}[/cyan]"
        if agent.session_manager.delete_session(target["session_id"])
        else "[red]Failed to delete session[/red]"
    )
    console.print(message)