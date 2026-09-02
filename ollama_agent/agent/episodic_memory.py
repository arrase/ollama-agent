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
                if not isinstance(msgs, list):
                    msgs = [msgs]
                for msg in msgs:
                    if msg.type in ("human", "user"):
                        text = extract_text(msg.content).strip()
                        if text and text not in seen:
                            seen.add(text)
                            prompts.append(text)
    except (sqlite3.Error, OSError) as e:
        raise HistoryError(_("Failed to read history database {db_path}: {e}", db_path=db_path, e=e)) from e
    return prompts


def load_past_conversations(
    db_path: Path = HISTORY_DB_PATH,
    exclude_thread_id: str = "",
) -> dict[str, dict[str, Any]]:
    """Load conversation messages and timestamps grouped by thread_id from SQLite history.

    Threads matching ``exclude_thread_id`` (e.g. active conversation) are skipped.
    """
    if not db_path.exists():
        return {}

    thread_timestamps: dict[str, str] = {}
    thread_messages: defaultdict[str, list[Any]] = defaultdict(list)

    try:
        with connect_history(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT thread_id, type, checkpoint FROM checkpoints ORDER BY rowid ASC")
            for tid, typ, chk in cursor.fetchall():
                if exclude_thread_id and tid.startswith(exclude_thread_id):
                    continue
                c = _serializer.loads_typed((typ, chk))
                if isinstance(c, dict) and "ts" in c:
                    thread_timestamps[tid] = str(c["ts"])

            cursor.execute("SELECT thread_id, type, value FROM writes WHERE channel = 'messages' ORDER BY rowid ASC")
            for tid, typ, val in cursor.fetchall():
                if exclude_thread_id and tid.startswith(exclude_thread_id):
                    continue
                msgs = _serializer.loads_typed((typ, val))
                if isinstance(msgs, list):
                    thread_messages[tid].extend(msgs)
                else:
                    thread_messages[tid].append(msgs)
    except (sqlite3.Error, OSError) as e:
        raise HistoryError(_("Failed to read history database {db_path}: {e}", db_path=db_path, e=e)) from e

    conversations: dict[str, dict[str, Any]] = {}
    for tid, msgs in thread_messages.items():
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


def search_past_conversations_in_db(
    query: str,
    db_path: Path = HISTORY_DB_PATH,
    exclude_thread_id: str = "",
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Search messages across past conversation sessions matching query keywords."""
    clean_query = query.strip()
    if not clean_query:
        return []

    conversations = load_past_conversations(db_path, exclude_thread_id=exclude_thread_id)
    if not conversations:
        return []

    terms = [t.lower() for t in clean_query.split() if t]
    scored_results: list[dict[str, Any]] = []

    for tid, data in conversations.items():
        msgs = data["messages"]
        raw_ts = data["timestamp"]
        formatted_date = data["formatted_date"]
        if not msgs:
            continue

        snippets: list[str] = []
        match_count = 0
        total_chars = 0
        snippets_full = False

        match_count += sum(formatted_date.lower().count(t) for t in terms)

        for msg in msgs:
            role = msg.type
            if role not in ("human", "ai", "user", "assistant"):
                continue

            text = extract_text(msg.content).strip()
            if not text:
                continue

            term_hits = sum(text.lower().count(t) for t in terms)
            if term_hits > 0:
                match_count += term_hits
                if not snippets_full:
                    snippet = _format_snippet(role, text)
                    snippets.append(snippet)
                    total_chars += len(snippet)
                    if total_chars > 1200:
                        snippets_full = True

        if match_count > 0:
            if not snippets:
                for msg in msgs:
                    role = msg.type
                    if role in ("human", "ai", "user", "assistant"):
                        text = extract_text(msg.content).strip()
                        if text:
                            snippets.append(_format_snippet(role, text))
                            if len(snippets) >= 2:
                                break

            scored_results.append(
                {
                    "thread_id": tid,
                    "score": match_count,
                    "timestamp": raw_ts,
                    "formatted_date": formatted_date,
                    "snippets": snippets,
                    "total_messages": len(msgs),
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
