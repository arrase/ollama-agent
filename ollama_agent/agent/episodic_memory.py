"""Episodic memory search engine for past agent conversations and experiences."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ..core.common import extract_text
from ..i18n import _
from ..settings.paths import HISTORY_DB_PATH

_serializer = JsonPlusSerializer()


class HistoryError(RuntimeError):
    """Raised when the conversation history database cannot be read."""


def connect_history(db_path: Path) -> sqlite3.Connection:
    """Open the history database, raising HistoryError when unreadable."""
    try:
        return sqlite3.connect(str(db_path))
    except sqlite3.Error as e:
        raise HistoryError(_("Failed to open history database {db_path}: {e}", db_path=db_path, e=e)) from e


def format_iso_timestamp(ts: str) -> str:
    """Format ISO timestamp into a human-readable UTC string (YYYY-MM-DD HH:MM UTC)."""
    if not ts:
        return ""
    return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M UTC")


def load_past_user_prompts(db_path: Path | None = None) -> list[str]:
    """Load past user prompt strings from the SQLite history database in chronological order."""
    db_path = db_path if db_path is not None else HISTORY_DB_PATH
    if not db_path.exists():
        return []

    prompts: list[str] = []
    seen: set[str] = set()
    try:
        with connect_history(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT type, value FROM writes WHERE channel = 'messages' ORDER BY rowid ASC"
            )
            for typ, val in cursor.fetchall():
                msgs = _serializer.loads_typed((typ, val))
                if not isinstance(msgs, list):
                    msgs = [msgs]
                for msg in msgs:
                    if getattr(msg, "type", "") in ("human", "user"):
                        text = extract_text(getattr(msg, "content", "")).strip()
                        if text and text not in seen:
                            seen.add(text)
                            prompts.append(text)
    except sqlite3.Error as e:
        raise HistoryError(_("Failed to read history database {db_path}: {e}", db_path=db_path, e=e)) from e
    except OSError as e:
        raise HistoryError(_("Failed to read history database {db_path}: {e}", db_path=db_path, e=e)) from e
    return prompts


def load_past_conversations(
    db_path: Path | None = None,
    exclude_thread_id: str = "",
) -> dict[str, dict[str, Any]]:
    """Load conversation messages and timestamps grouped by thread_id from SQLite history.

    Threads matching ``exclude_thread_id`` (e.g. active conversation) are skipped.
    """
    db_path = db_path if db_path is not None else HISTORY_DB_PATH
    if not db_path.exists():
        return {}

    thread_timestamps: dict[str, str] = {}
    thread_messages: defaultdict[str, list[Any]] = defaultdict(list)

    try:
        with connect_history(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT thread_id, type, checkpoint FROM checkpoints ORDER BY rowid ASC"
            )
            for tid, typ, chk in cursor.fetchall():
                if exclude_thread_id and tid.startswith(exclude_thread_id):
                    continue
                c = _serializer.loads_typed((typ, chk))
                if isinstance(c, dict) and "ts" in c:
                    thread_timestamps[tid] = str(c["ts"])

            cursor.execute(
                "SELECT thread_id, type, value FROM writes WHERE channel = 'messages' ORDER BY rowid ASC"
            )
            for tid, typ, val in cursor.fetchall():
                if exclude_thread_id and tid.startswith(exclude_thread_id):
                    continue
                msgs = _serializer.loads_typed((typ, val))
                if isinstance(msgs, list):
                    thread_messages[tid].extend(msgs)
                else:
                    thread_messages[tid].append(msgs)
    except sqlite3.Error as e:
        raise HistoryError(_("Failed to read history database {db_path}: {e}", db_path=db_path, e=e)) from e
    except OSError as e:
        raise HistoryError(_("Failed to read history database {db_path}: {e}", db_path=db_path, e=e)) from e

    conversations: dict[str, dict[str, Any]] = {}
    for tid, msgs in thread_messages.items():
        raw_ts = thread_timestamps.get(tid, "")
        conversations[tid] = {
            "timestamp": raw_ts,
            "formatted_date": format_iso_timestamp(raw_ts),
            "messages": msgs,
        }

    return conversations


def search_past_conversations_in_db(
    query: str,
    db_path: Path | None = None,
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

        match_count += sum(formatted_date.lower().count(t) for t in terms)

        for msg in msgs:
            role = getattr(msg, "type", "unknown")
            if role not in ("human", "ai", "user", "assistant"):
                continue

            text = extract_text(getattr(msg, "content", "")).strip()
            if not text:
                continue

            term_hits = sum(text.lower().count(t) for t in terms)
            if term_hits > 0:
                match_count += term_hits
                truncated = text if len(text) <= 300 else f"{text[:297]}..."
                role_label = _("User") if role in ("human", "user") else _("Assistant")
                snippets.append(f"[{role_label}]: {truncated}")
                total_chars += len(truncated)
                if total_chars > 1200:
                    break

        if match_count > 0:
            scored_results.append({
                "thread_id": tid,
                "score": match_count,
                "timestamp": raw_ts,
                "formatted_date": formatted_date,
                "snippets": snippets,
                "total_messages": len(msgs),
            })

    scored_results.sort(key=lambda item: (item["score"], item["timestamp"]), reverse=True)
    return scored_results[:limit]


def format_past_conversations_context(results: list[dict[str, Any]]) -> str:
    """Format matching episodic conversation sessions into a markdown context string with dates."""
    if not results:
        return _("No relevant past conversations found in episodic memory.")

    lines: list[str] = [
        _("Found {count} relevant past conversation(s) in episodic memory:", count=len(results)) + "\n"
    ]
    for idx, item in enumerate(results, start=1):
        tid = item["thread_id"]
        short_id = tid[:8]
        date_str = item["formatted_date"]
        header_date = f" - [{_('Date:')} {date_str}]" if date_str else ""
        lines.append(
            f"### {_('Session')} #{idx} ({short_id}){header_date} - [{_('Total messages:')} {item['total_messages']}]"
        )
        for snippet in item["snippets"]:
            lines.append(f"  {snippet}")
        lines.append("")

    return "\n".join(lines).strip()
