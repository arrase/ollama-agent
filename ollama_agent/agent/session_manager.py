"""Session management backed by SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core import extract_text

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SessionManager:
    """Handles database operations for agent sessions."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.storage_path = database_path or Path.home() / ".ollama-agent" / "sessions.db"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self.storage_path)
        self.session_id: str | None = None
        self._ensure_schema()
        self.reset_session()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT,
                        updated_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        created_at TEXT,
                        message_data TEXT,
                        FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_messages_session_id ON agent_messages(session_id)"
                )
        except Exception as exc:
            logger.error("Error ensuring session schema: %s", exc)

    @staticmethod
    def _to_local_time(utc_str: str | None) -> str:
        if not utc_str:
            return "Unknown"
        try:
            utc_dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
            return (
                (utc_dt if utc_dt.tzinfo else utc_dt.replace(tzinfo=timezone.utc))
                .astimezone()
                .strftime("%Y-%m-%d %H:%M")
            )
        except (ValueError, TypeError):
            return utc_str[:16] if len(utc_str) > 16 else utc_str

    @staticmethod
    def _preview(blob: str | None) -> str:
        if not blob:
            return "No messages"
        try:
            data = json.loads(blob)
            return (extract_text(data.get("content") if isinstance(data, dict) else data) or str(data))[:50]
        except (json.JSONDecodeError, TypeError):
            return "No content"

    def _touch_session(self, session_id: str, *, create: bool = False) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            if create:
                conn.execute(
                    "INSERT OR IGNORE INTO agent_sessions(session_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (session_id, now, now),
                )
            conn.execute(
                "UPDATE agent_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def reset_session(self) -> str:
        self.session_id = str(uuid.uuid4())
        self._touch_session(self.session_id, create=True)
        return self.session_id

    def load_session(self, session_id: str) -> None:
        self.session_id = session_id
        self._touch_session(session_id, create=True)

    def get_session_id(self) -> str | None:
        return self.session_id

    def append_message(self, role: str, content: Any, *, session_id: str | None = None) -> None:
        sid = session_id or self.session_id
        if not sid:
            return
        now = _utc_now_iso()
        payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO agent_messages(session_id, created_at, message_data) VALUES (?, ?, ?)",
                    (sid, now, payload),
                )
            self._touch_session(sid, create=True)
        except Exception as exc:
            logger.error("Error appending message: %s", exc)

    def get_message_dicts(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Return message dicts in {role, content} format."""
        sid = session_id or self.session_id
        if not sid or not self.storage_path.exists():
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT message_data FROM agent_messages WHERE session_id = ? ORDER BY created_at ASC",
                    (sid,),
                ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                blob = row["message_data"]
                if not blob:
                    continue
                try:
                    data = json.loads(blob)
                    if isinstance(data, dict) and data.get("role") in ("user", "assistant"):
                        out.append({"role": data["role"], "content": data.get("content", "")})
                except (json.JSONDecodeError, TypeError):
                    continue
            return out
        except Exception as exc:
            logger.error("Error getting message dicts: %s", exc)
            return []

    def get_readable_history(self, session_id: str | None = None) -> list[dict[str, str]]:
        sid = session_id or self.session_id
        messages = self.get_message_dicts(sid)
        readable: list[dict[str, str]] = []
        for msg in messages:
            content = extract_text(msg.get("content", ""))
            if content:
                readable.append({"role": str(msg.get("role")), "content": content})
        return readable

    def list_sessions(self, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT s.session_id,
                           COUNT(m.id) AS message_count,
                           s.created_at,
                           s.updated_at,
                           (
                               SELECT message_data
                               FROM agent_messages
                               WHERE session_id = s.session_id
                               ORDER BY created_at ASC
                               LIMIT 1
                           ) AS first_message_data
                    FROM agent_sessions s
                    LEFT JOIN agent_messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()

            return [
                {
                    "session_id": r["session_id"],
                    "message_count": int(r["message_count"] or 0),
                    "first_message": self._to_local_time(r["created_at"]),
                    "last_message": self._to_local_time(r["updated_at"]),
                    "preview": self._preview(r["first_message_data"]),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.error("Error listing sessions: %s", exc)
            return []

    def delete_session(self, session_id: str) -> bool:
        if not self.storage_path.exists():
            return False
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM agent_messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM agent_sessions WHERE session_id = ?", (session_id,))
            if session_id == self.session_id:
                self.reset_session()
            return True
        except Exception as exc:
            logger.error("Error deleting session: %s", exc)
            return False
