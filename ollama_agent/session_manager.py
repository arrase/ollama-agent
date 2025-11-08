"""Session management for the agent."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

from agents import SQLiteSession
from .utils import extract_text

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

    @staticmethod
    def _extract_preview_text(message_blob: Optional[str]) -> str:
        if not message_blob:
            return "No messages"
        try:
            message_data = json.loads(message_blob)
        except (json.JSONDecodeError, TypeError):
            return "No content"

        content: Any = message_data.get("content") if isinstance(message_data, dict) else message_data
        text_preview = extract_text(content)
        return text_preview[:50] if text_preview else str(message_data)[:50]

    def reset_session(self) -> str:
        """Resets the current session and returns a new session ID."""
        self.session_id = str(uuid.uuid4())
        self.session = SQLiteSession(self.session_id, self._db_path)
        return self.session_id

    def load_session(self, session_id: str) -> None:
        """Loads an existing session."""
        self.session_id = session_id
        self.session = SQLiteSession(session_id, self._db_path)

    def get_session_id(self) -> Optional[str]:
        """Returns the current session ID."""
        return self.session_id

    def get_session(self) -> Optional[SQLiteSession]:
        """Returns the current session."""
        return self.session

    def list_sessions(self) -> list[dict[str, Any]]:
        """Lists all available sessions."""
        if not self.storage_path.exists():
            return []
        try:
            with sqlite3.connect(self._db_path) as conn:
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
                           ) AS first_message
                    FROM agent_sessions s
                    LEFT JOIN agent_messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    """
                ).fetchall()

            return [
                {
                    "session_id": session_id,
                    "message_count": message_count,
                    "first_message": first_time or "Unknown",
                    "last_message": last_time or "Unknown",
                    "preview": self._extract_preview_text(first_message),
                }
                for session_id, message_count, first_time, last_time, first_message in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("Error listing sessions: %s", exc)
            return []

    async def get_session_history(self, session_id: Optional[str] = None) -> list[Any]:
        """Retrieves the history for a given session."""
        session_id = session_id or self.session_id
        if not session_id:
            return []
        temp_session = SQLiteSession(session_id, self._db_path)
        try:
            return list(await temp_session.get_items())
        except Exception as exc:  # noqa: BLE001
            logger.error("Error getting session history: %s", exc)
            return []

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session from the database."""
        if not self.storage_path.exists():
            return False
        try:
            with sqlite3.connect(self._db_path) as conn:
                for table in ("agent_messages", "agent_sessions"):
                    conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            if session_id == self.session_id:
                self.reset_session()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Error deleting session: %s", exc)
            return False
