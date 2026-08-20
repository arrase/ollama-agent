"""Session-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from ..core.common import extract_text
from ..settings.paths import HISTORY_DB_PATH

if TYPE_CHECKING:
    from ..agent import AgentRuntime


def new_session(console: Console) -> str:
    """Start a new session by generating a new thread ID.

    Returns the new thread ID. The caller is responsible for assigning it
    to the runtime.
    """
    new_id = str(uuid.uuid4())[:8]
    console.print(f"[green]✓ New session started:[/green] [cyan]{new_id}[/cyan]")
    return new_id


def get_available_sessions(db_path: Path = HISTORY_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve unique sessions stored in the SQLite history database."""
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id, COUNT(*) as steps FROM checkpoints GROUP BY thread_id ORDER BY rowid DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"thread_id": row[0], "steps": row[1]} for row in rows]
    except (sqlite3.Error, OSError):
        return []


def list_sessions(
    console: Console,
    db_path: Path = HISTORY_DB_PATH,
    current_thread_id: str = "",
) -> list[dict[str, Any]]:
    """List available chat sessions with step counts and current session indicator."""
    sessions = get_available_sessions(db_path)
    if not sessions:
        console.print("[yellow]No saved sessions found in history.[/yellow]")
        return []

    table = Table(title="Saved Sessions", header_style="bold cyan")
    table.add_column("Session ID", style="bold")
    table.add_column("Steps", justify="right", style="dim")
    table.add_column("Status", justify="left")

    for s in sessions:
        tid = s["thread_id"]
        steps = str(s["steps"])
        is_current = tid == current_thread_id or (current_thread_id and tid.startswith(current_thread_id))
        marker = "[green]◀ current[/green]" if is_current else ""
        table.add_row(tid, steps, marker)

    console.print(table)
    console.print("[dim]Use /session resume <id> to switch to a previous session.[/dim]")
    return sessions


def resolve_session_id(target_id: str, available_sessions: list[dict[str, Any]]) -> str | None:
    """Resolve target_id by exact match or unambiguous prefix."""
    target_id = target_id.strip()
    if not target_id:
        return None

    # Exact match
    for s in available_sessions:
        if s["thread_id"] == target_id:
            return target_id

    # Prefix match
    matches = [s["thread_id"] for s in available_sessions if s["thread_id"].startswith(target_id)]
    if len(matches) == 1:
        return matches[0]
    return None


def resume_session(
    console: Console,
    target_id: str,
    available_sessions: list[dict[str, Any]] | None = None,
    db_path: Path = HISTORY_DB_PATH,
) -> str | None:
    """Resume a previous session by thread ID or prefix."""
    sessions = available_sessions if available_sessions is not None else get_available_sessions(db_path)
    if not sessions:
        console.print("[red]No sessions available to resume.[/red]")
        return None

    resolved = resolve_session_id(target_id, sessions)
    if resolved is None:
        prefix_matches = [s["thread_id"] for s in sessions if s["thread_id"].startswith(target_id)]
        if len(prefix_matches) > 1:
            console.print(f"[red]Ambiguous session ID '{target_id}'. Matches: {', '.join(prefix_matches[:5])}[/red]")
        else:
            console.print(f"[red]Session '{target_id}' not found.[/red]")
        return None

    console.print(f"[green]✓ Switched to session:[/green] [cyan]{resolved[:8]}[/cyan] [dim]({resolved})[/dim]")
    return resolved


def delete_session(
    console: Console,
    target_id: str,
    db_path: Path = HISTORY_DB_PATH,
) -> bool:
    """Delete a session from the SQLite checkpoint database."""
    sessions = get_available_sessions(db_path)
    resolved = resolve_session_id(target_id, sessions)
    if resolved is None:
        console.print(f"[red]Session '{target_id}' not found.[/red]")
        return False

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (resolved,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (resolved,))
        conn.commit()
        conn.close()
        console.print(f"[green]✓ Deleted session:[/green] [cyan]{resolved}[/cyan]")
        return True
    except (sqlite3.Error, OSError) as exc:
        console.print(f"[red]Failed to delete session '{resolved}': {exc}[/red]")
        return False


async def export_session(
    console: Console,
    runtime: AgentRuntime,
    target_id: str,
    output_path: str | None = None,
    db_path: Path = HISTORY_DB_PATH,
) -> Path | None:
    """Export conversation messages from a session to a Markdown file."""
    sessions = get_available_sessions(db_path)
    resolved = resolve_session_id(target_id, sessions) if sessions else target_id
    if not resolved:
        resolved = target_id

    if runtime.graph is None:
        await runtime.reload()

    config = {"configurable": {"thread_id": resolved}}
    state = await runtime.graph.aget_state(config)
    messages = state.values.get("messages", []) if state and state.values else []

    if not messages:
        console.print(f"[yellow]No messages found for session '{resolved}'.[/yellow]")
        return None

    lines: list[str] = [
        f"# Session Export: {resolved}",
        "",
    ]

    for msg in messages:
        role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
        raw_content = getattr(msg, "content", "")
        content = extract_text(raw_content)

        if role in ("human", "user"):
            lines.append("## 👤 User")
            lines.append("")
            lines.append(content)
            lines.append("")
        elif role in ("ai", "assistant"):
            lines.append("## 🤖 Assistant")
            lines.append("")
            lines.append(content)
            lines.append("")
        elif role in ("tool",):
            name = getattr(msg, "name", "tool")
            lines.append(f"### ⚙ Tool: `{name}`")
            lines.append("```")
            lines.append(content[:1000])
            lines.append("```")
            lines.append("")

    target_file = Path(output_path).expanduser().resolve() if output_path else Path.cwd() / f"session_{resolved[:8]}.md"
    try:
        target_file.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]✓ Session exported to:[/green] [cyan]{target_file}[/cyan]")
        return target_file
    except OSError as exc:
        console.print(f"[red]Failed to export session: {exc}[/red]")
        return None


async def compact_session(
    console: Console,
    runtime: AgentRuntime,
    target_id: str = "",
) -> dict[str, Any]:
    """Compact conversation context for a session into a structured summary."""
    res = await runtime.compact_context(target_id)
    if res["success"]:
        console.print("[green]✓ Context compacted successfully:[/green]")
        console.print(f"  [dim]• Messages summarized:[/dim] {res['messages_summarized']}")
        console.print(f"  [dim]• Recent messages preserved:[/dim] {res['messages_preserved']}")
        if res.get("file_path"):
            console.print(f"  [dim]• History offloaded to:[/dim] [cyan]{res['file_path']}[/cyan]")
    else:
        console.print(f"[yellow]{res.get('message', 'Compaction failed.')}[/yellow]")
    return res
