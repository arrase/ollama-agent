"""Session management for the agent."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from agents import SQLiteSession

from ..core import extract_text

logger = logging.getLogger(__name__)


class SessionManager:
    """Handles database operations for agent sessions."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.storage_path = database_path or Path.home() / ".ollama-agent" / "sessions.db"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self.storage_path)
        self.session_id: str | None = None
        self.session: SQLiteSession | None = None
        self.reset_session()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _make_session(self, session_id: str) -> SQLiteSession:
        return SQLiteSession(session_id, self._db_path)

    @staticmethod
    def _preview(blob: str | None) -> str:
        if not blob:
            return "No messages"
        try:
            data = json.loads(blob)
            return (extract_text(data.get("content") if isinstance(data, dict) else data) or str(data))[:50]
        except (json.JSONDecodeError, TypeError):
            return "No content"

    def reset_session(self) -> str:
        self.session_id = str(uuid.uuid4())
        self.session = self._make_session(self.session_id)
        return self.session_id

    def load_session(self, session_id: str) -> None:
        self.session_id = session_id
        self.session = self._make_session(session_id)

    def get_session_id(self) -> str | None:
        return self.session_id

    def get_session(self) -> SQLiteSession | None:
        return self.session

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT s.session_id, COUNT(m.id) AS message_count, s.created_at, s.updated_at,
                           (SELECT message_data FROM agent_messages WHERE session_id = s.session_id ORDER BY created_at ASC LIMIT 1) AS first_message_data
                    FROM agent_sessions s LEFT JOIN agent_messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id ORDER BY s.updated_at DESC""").fetchall()
            return [{"session_id": r["session_id"], "message_count": int(r["message_count"] or 0),
                     "first_message": r["created_at"] or "Unknown", "last_message": r["updated_at"] or "Unknown",
                     "preview": self._preview(r["first_message_data"])} for r in rows]
        except Exception as e:
            logger.error("Error listing sessions: %s", e)
            return []

    async def get_session_history(self, session_id: str | None = None) -> list[Any]:
        sid = session_id or self.session_id
        if not sid:
            return []
        try:
            return list(await self._make_session(sid).get_items())
        except Exception as e:
            logger.error("Error getting session history: %s", e)
            return []

    def delete_session(self, session_id: str) -> bool:
        if not self.storage_path.exists():
            return False
        try:
            with self._connect() as conn:
                for table in ("agent_messages", "agent_sessions"):
                    conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            if session_id == self.session_id:
                self.reset_session()
            return True
        except Exception as e:
            logger.error("Error deleting session: %s", e)
            return False
