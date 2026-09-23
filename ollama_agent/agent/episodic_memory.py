"""Episodic memory search engine for past agent conversations and experiences."""

from __future__ import annotations

import contextlib
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ..core.common import extract_text
from ..i18n import _
from ..settings.paths import HISTORY_DB_PATH

_serializer = JsonPlusSerializer()


class HistoryError(RuntimeError):
    """Raised when the conversation history database cannot be read."""


@contextlib.contextmanager
def connect_history(db_path: Path, read_only: bool = True) -> Iterator[sqlite3.Connection]:
    """Open the history database as a context manager that closes on exit."""
    try:
        if read_only:
            conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as e:
        raise HistoryError(_("Failed to open history database {db_path}: {e}", db_path=db_path, e=e)) from e
    try:
        yield conn
    finally:
        conn.close()


def format_iso_timestamp(ts: str) -> str:
    """Format ISO timestamp into a human-readable UTC string (YYYY-MM-DD HH:MM UTC)."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def load_past_user_prompts(db_path: Path = HISTORY_DB_PATH) -> list[str]:
    """Load past user prompt strings from the SQLite history database in chronological order."""
    if not db_path.exists():
        return []

    prompts: list[str] = []
    seen: set[str] = set()
    try:
        with connect_history(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT type, value FROM writes WHERE channel = 'messages' ORDER BY rowid ASC")
            for typ, val in cursor.fetchall():
                msgs = _serializer.loads_typed((typ, val))
                msg_list = msgs if isinstance(msgs, list) else [msgs]
                for msg in msg_list:
                    if msg.type in ("human", "user"):
                        text = extract_text(msg.content).strip()
                        if text and text not in seen:
                            seen.add(text)
                            prompts.append(text)
    except (sqlite3.Error, OSError) as e:
        raise HistoryError(_("Failed to read history database {db_path}: {e}", db_path=db_path, e=e)) from e
    return prompts


def _read_history_checkpoints(cursor: sqlite3.Cursor, exclude_thread_id: str) -> dict[str, str]:
    thread_timestamps: dict[str, str] = {}
    cursor.execute("SELECT thread_id, type, checkpoint FROM checkpoints ORDER BY rowid ASC")
    for tid, typ, chk in cursor.fetchall():
        if exclude_thread_id and tid.startswith(exclude_thread_id):
            continue
        c = _serializer.loads_typed((typ, chk))
        thread_timestamps[tid] = str(c["ts"])
    return thread_timestamps


def _read_history_messages(cursor: sqlite3.Cursor, exclude_thread_id: str) -> defaultdict[str, list[Any]]:
    thread_messages: defaultdict[str, list[Any]] = defaultdict(list)
    cursor.execute("SELECT thread_id, type, value FROM writes WHERE channel = 'messages' ORDER BY rowid ASC")
    for tid, typ, val in cursor.fetchall():
        if exclude_thread_id and tid.startswith(exclude_thread_id):
            continue
        msgs = _serializer.loads_typed((typ, val))
        if isinstance(msgs, list):
            thread_messages[tid].extend(msgs)
        else:
            thread_messages[tid].append(msgs)
    return thread_messages


def load_past_conversations(
    db_path: Path = HISTORY_DB_PATH,
    exclude_thread_id: str = "",
) -> dict[str, dict[str, Any]]:
    """Load conversation messages and timestamps grouped by thread_id from SQLite history.

    Threads matching ``exclude_thread_id`` (e.g. active conversation) are skipped.
    """
    if not db_path.exists():
        return {}

    try:
        with connect_history(db_path) as conn:
            cursor = conn.cursor()
            thread_timestamps = _read_history_checkpoints(cursor, exclude_thread_id)
            thread_messages = _read_history_messages(cursor, exclude_thread_id)
    except (sqlite3.Error, OSError) as e:
        raise HistoryError(_("Failed to read history database {db_path}: {e}", db_path=db_path, e=e)) from e

    conversations: dict[str, dict[str, Any]] = {}
    for tid, msgs in thread_messages.items():
        if tid in thread_timestamps:
            raw_ts = thread_timestamps[tid]
            conversations[tid] = {
                "timestamp": raw_ts,
                "formatted_date": format_iso_timestamp(raw_ts),
                "messages": msgs,
            }

    return conversations


def _format_snippet(role: str, text: str) -> str:
    truncated = text if len(text) <= 300 else f"{text[:297]}..."
    role_label = _("User") if role in ("human", "user") else _("Assistant")
    return f"[{role_label}]: {truncated}"


def _score_conversation(data: dict[str, Any], terms: list[str]) -> tuple[int, list[str]]:
    msgs = data["messages"]
    formatted_date = data["formatted_date"]

    dialogue: list[tuple[str, str]] = []
    for msg in msgs:
        if msg.type in ("human", "ai", "user", "assistant"):
            text = extract_text(msg.content).strip()
            if text:
                dialogue.append((msg.type, text))

    snippets: list[str] = []
    match_count = sum(formatted_date.lower().count(t) for t in terms)

    for role, text in dialogue:
        term_hits = sum(text.lower().count(t) for t in terms)
        if term_hits > 0:
            match_count += term_hits
            if len(snippets) < 4:
                snippets.append(_format_snippet(role, text))

    if match_count > 0 and not snippets:
        snippets = [_format_snippet(r, t) for r, t in dialogue[:2]]

    return match_count, snippets


def search_past_conversations_in_db(
    query: str,
    db_path: Path = HISTORY_DB_PATH,
    exclude_thread_id: str = "",
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Search messages across past conversation sessions matching query keywords."""
    terms = [t.lower() for t in query.split()]
    if not terms:
        return []

    conversations = load_past_conversations(db_path, exclude_thread_id=exclude_thread_id)
    scored_results: list[dict[str, Any]] = []

    for tid, data in conversations.items():
        match_count, snippets = _score_conversation(data, terms)
        if match_count > 0:
            scored_results.append(
                {
                    "thread_id": tid,
                    "score": match_count,
                    "timestamp": data["timestamp"],
                    "formatted_date": data["formatted_date"],
                    "snippets": snippets,
                    "total_messages": len(data["messages"]),
                }
            )

    scored_results.sort(key=lambda item: (item["score"], item["timestamp"]), reverse=True)
    return scored_results[:limit]


def format_past_conversations_context(results: list[dict[str, Any]]) -> str:
    """Format matching episodic conversation sessions into a markdown context string with dates."""
    if not results:
        return _("No relevant past conversations found in episodic memory.")

    lines: list[str] = [_("Found {count} relevant past conversation(s) in episodic memory:", count=len(results)) + "\n"]
    for idx, item in enumerate(results, start=1):
        tid = item["thread_id"]
        short_id = tid[:8]
        header_date = f" - [{_('Date:')} {item['formatted_date']}]"
        lines.append(
            f"### {_('Session')} #{idx} ({short_id}){header_date} - [{_('Total messages:')} {item['total_messages']}]"
        )
        for snippet in item["snippets"]:
            indented = "\n  ".join(snippet.splitlines())
            lines.append(f"  {indented}")
        lines.append("")

    return "\n".join(lines).strip()
