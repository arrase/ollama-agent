"""Episodic memory search engine for past agent conversations and experiences."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ..core.common import extract_text
from ..settings.paths import HISTORY_DB_PATH

_serializer = JsonPlusSerializer()


def load_past_conversations(
    db_path: Path = HISTORY_DB_PATH,
    exclude_thread_id: str = "",
) -> dict[str, list[Any]]:
    """Load conversation messages grouped by thread_id from SQLite history.

    Threads matching ``exclude_thread_id`` (e.g. active conversation) are skipped.
    """
    if not db_path.exists():
        return {}

    conversations: dict[str, list[Any]] = {}
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id, type, value FROM writes WHERE channel = 'messages' ORDER BY rowid ASC"
        )
        rows = cursor.fetchall()

    for tid, typ, val in rows:
        if exclude_thread_id and (tid == exclude_thread_id or tid.startswith(exclude_thread_id)):
            continue
        msgs = _serializer.loads_typed((typ, val))
        if tid not in conversations:
            conversations[tid] = []
        if isinstance(msgs, list):
            conversations[tid].extend(msgs)
        else:
            conversations[tid].append(msgs)

    return conversations


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

    for tid, msgs in conversations.items():
        if not msgs:
            continue

        snippets: list[str] = []
        match_count = 0
        total_chars = 0

        for msg in msgs:
            role = msg.type
            text = extract_text(msg.content).strip()
            if not text:
                continue

            text_lower = text.lower()
            term_hits = sum(text_lower.count(t) for t in terms)
            if term_hits > 0:
                match_count += term_hits
                truncated = text if len(text) <= 300 else f"{text[:297]}..."
                role_label = "User" if role == "human" else ("Assistant" if role == "ai" else role)
                snippets.append(f"[{role_label}]: {truncated}")
                total_chars += len(truncated)
                if total_chars > 1200:
                    break

        if match_count > 0 and snippets:
            scored_results.append({
                "thread_id": tid,
                "score": match_count,
                "snippets": snippets,
                "total_messages": len(msgs),
            })

    scored_results.sort(key=lambda item: item["score"], reverse=True)
    return scored_results[:max(1, limit)]


def format_past_conversations_context(results: list[dict[str, Any]]) -> str:
    """Format matching episodic conversation sessions into a markdown context string."""
    if not results:
        return "No relevant past conversations found in episodic memory."

    lines: list[str] = [
        f"Found {len(results)} relevant past conversation(s) in episodic memory:\n"
    ]
    for idx, item in enumerate(results, start=1):
        tid = item["thread_id"]
        short_id = tid[:8]
        lines.append(f"### Session #{idx} ({short_id}) - [Total messages: {item['total_messages']}]")
        for snippet in item["snippets"]:
            lines.append(f"  {snippet}")
        lines.append("")

    return "\n".join(lines).strip()
