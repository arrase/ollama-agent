"""Session-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console
from rich.table import Table

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ..agent.episodic_memory import (
    HistoryError,
    connect_history,
    format_iso_timestamp,
    search_past_conversations_in_db,
)
from ..core.common import extract_text
from ..i18n import _
from ..settings.paths import HISTORY_DB_PATH

if TYPE_CHECKING:
    from ..agent import AgentRuntime

_serializer = JsonPlusSerializer()


def is_current(thread_id: str, current_thread_id: str) -> bool:
    """Return whether *thread_id* is the current session (exact match or bidirectional prefix match)."""
    if not thread_id.strip() or not current_thread_id.strip():
        return False
    return (
        thread_id == current_thread_id
        or thread_id.startswith(current_thread_id)
        or current_thread_id.startswith(thread_id)
    )


def new_session(console: Console) -> str:
    """Start a new session by generating a new thread ID.

    Returns the new thread ID. The caller is responsible for assigning it
    to the runtime.
    """
    new_id = str(uuid.uuid4())[:8]
    console.print(f"[green]✓ {_('New session started: {new_id}', new_id=new_id)}[/green]")
    return new_id


def get_available_sessions(db_path: Path = HISTORY_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve unique sessions stored in the SQLite history database with timestamps and step counts."""
    if not db_path.exists():
        return []

    try:
        with connect_history(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT thread_id, type, checkpoint FROM checkpoints ORDER BY rowid ASC"
            )
            timestamps: dict[str, str] = {}
            for tid, typ, chk in cursor.fetchall():
                c = _serializer.loads_typed((typ, chk))
                if isinstance(c, dict) and "ts" in c:
                    timestamps[tid] = str(c["ts"])

            cursor.execute(
                "SELECT thread_id, COUNT(*) as steps FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid) DESC"
            )
            rows = cursor.fetchall()
            return [
                {
                    "thread_id": row[0],
                    "steps": row[1],
                    "timestamp": format_iso_timestamp(timestamps.get(row[0], "")),
                }
                for row in rows
            ]
    except (sqlite3.Error, OSError, HistoryError):
        return []


def list_sessions(
    console: Console,
    db_path: Path = HISTORY_DB_PATH,
    current_thread_id: str = "",
) -> list[dict[str, Any]]:
    """List available chat sessions with step counts, timestamps, and current session indicator."""
    sessions = get_available_sessions(db_path)
    if not sessions:
        console.print(f"[yellow]{_('No saved sessions found in history.')}[/yellow]")
        return []

    table = Table(title=_("Saved Sessions"), box=box.ROUNDED, header_style="bold cyan")
    table.add_column(_("Session ID"), style="bold", no_wrap=True)
    table.add_column(_("Date / Time"), style="cyan", no_wrap=True)
    table.add_column(_("Steps"), justify="right", style="dim", no_wrap=True)
    table.add_column(_("Status"), justify="left", no_wrap=True)

    for s in sessions:
        tid = s["thread_id"]
        ts = s["timestamp"]
        steps = str(s["steps"])
        is_current_session = is_current(tid, current_thread_id)
        marker = f"[green]◀ {_('current')}[/green]" if is_current_session else ""
        table.add_row(tid, ts, steps, marker)

    console.print(table)
    console.print(f"[dim]{_('Use /session resume <id> to switch to a previous session.')}[/dim]")
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
        console.print(f"[red]{_('No sessions available to resume.')}[/red]")
        return None

    resolved = resolve_session_id(target_id, sessions)
    if resolved is None:
        prefix_matches = [s["thread_id"] for s in sessions if s["thread_id"].startswith(target_id)]
        if len(prefix_matches) > 1:
            matches_str = ", ".join(prefix_matches[:5])
            ambiguous_msg = _("Ambiguous session ID '{target_id}'. Matches: {matches}", target_id=target_id, matches=matches_str)
            console.print(f"[red]{ambiguous_msg}[/red]")
        else:
            not_found_msg = _("Session '{target_id}' not found.", target_id=target_id)
            console.print(f"[red]{not_found_msg}[/red]")
        return None

    switched_msg = _("Switched to session: {resolved}", resolved=f"{resolved[:8]} ({resolved})")
    console.print(f"[green]✓ {switched_msg}[/green]")
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
        not_found_msg = _("Session '{target_id}' not found.", target_id=target_id)
        console.print(f"[red]{not_found_msg}[/red]")
        return False

    try:
        with connect_history(db_path, read_only=False) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (resolved,))
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (resolved,))
            conn.commit()
        deleted_msg = _("Deleted session: {resolved}", resolved=resolved)
        console.print(f"[green]✓ {deleted_msg}[/green]")
        return True
    except (sqlite3.Error, OSError, HistoryError) as exc:
        failed_msg = _("Failed to delete session '{resolved}': {exc}", resolved=resolved, exc=exc)
        console.print(f"[red]{failed_msg}[/red]")
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
    resolved = resolve_session_id(target_id, sessions)
    if resolved is None:
        not_found_msg = _("Session '{target_id}' not found.", target_id=target_id)
        console.print(f"[red]{not_found_msg}[/red]")
        return None

    messages = await runtime.get_thread_messages(resolved)

    if not messages:
        no_msgs = _("No messages found for session '{resolved}'.", resolved=resolved)
        console.print(f"[yellow]{no_msgs}[/yellow]")
        return None

    export_title = _("Session Export: {resolved}", resolved=resolved)
    user_label = _("User")
    asst_label = _("Assistant")

    lines: list[str] = [
        f"# {export_title}",
        "",
    ]

    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = extract_text(getattr(msg, "content", ""))

        if role in ("human", "user"):
            lines.extend([f"## 👤 {user_label}", "", content, ""])
        elif role in ("ai", "assistant"):
            lines.extend([f"## 🤖 {asst_label}", ""])
            if content:
                lines.extend([content, ""])
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_name = tc.get("name", "tool") if isinstance(tc, dict) else getattr(tc, "name", "tool")
                tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tool_call_hdr = _("Tool: {name}", name=tc_name)
                args_str = json.dumps(tc_args, indent=2, ensure_ascii=False) if isinstance(tc_args, dict) else str(tc_args)
                lines.extend([f"### ⚙ {tool_call_hdr}", "```json", args_str, "```", ""])
        elif role == "tool":
            name = getattr(msg, "name", "tool")
            tool_hdr = _("Tool: {name}", name=name)
            lines.extend([f"### ⚙ {tool_hdr}", "```", content, "```", ""])

    target_file = Path(output_path).expanduser().resolve() if output_path else Path.cwd() / f"session_{resolved[:8]}.md"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_file.write_text("\n".join(lines), encoding="utf-8")
        exported_msg = _("Session exported to: {target_file}", target_file=target_file)
        console.print(f"[green]✓ {exported_msg}[/green]")
        return target_file
    except OSError as exc:
        failed_export = _("Failed to export session: {exc}", exc=exc)
        console.print(f"[red]{failed_export}[/red]")
        return None




def search_sessions(
    console: Console,
    query: str,
    db_path: Path = HISTORY_DB_PATH,
    current_thread_id: str = "",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search chat sessions matching query keywords and display formatted results."""
    clean_query = query.strip()
    if not clean_query:
        console.print(f"[yellow]{_('Please provide a search query.')}[/yellow]")
        return []

    results = search_past_conversations_in_db(
        query=clean_query,
        db_path=db_path,
        exclude_thread_id="",
        limit=limit,
    )
    if not results:
        no_res = _("No sessions found matching '{query}'.", query=clean_query)
        console.print(f"[yellow]{no_res}[/yellow]")
        return []

    table_title = _("Search Results for '{query}'", query=clean_query)
    table = Table(title=table_title, box=box.ROUNDED, header_style="bold cyan")
    table.add_column(_("Session ID"), style="bold", no_wrap=True)
    table.add_column(_("Date / Time"), style="cyan", no_wrap=True)
    table.add_column(_("Score"), justify="right", style="dim", no_wrap=True)
    table.add_column(_("Snippet Preview"), justify="left")

    for item in results:
        tid = item["thread_id"]
        is_current_session = is_current(tid, current_thread_id)
        tid_display = f"{tid[:8]}" + (f" [green]◀ {_('current')}[/green]" if is_current_session else "")
        ts_display = item["formatted_date"]
        score_display = str(item["score"])
        cleaned_snippets = []
        for s in item["snippets"][:2]:
            single = " ".join(s.split())
            if len(single) > 120:
                single = f"{single[:117]}..."
            cleaned_snippets.append(single)
        snippets_display = "\n".join(cleaned_snippets)
        table.add_row(tid_display, ts_display, score_display, snippets_display)

    console.print(table)
    console.print(f"[dim]{_('Use /session resume <id> to switch to a matched session.')}[/dim]")
    return results
