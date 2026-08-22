from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from rich.console import Console

from ollama_agent.interfaces.session_commands import (
    compact_session,
    delete_session,
    export_session,
    get_available_sessions,
    list_sessions,
    new_session,
    resolve_session_id,
    resume_session,
    search_sessions,
)


class TestSessionCommands(unittest.IsolatedAsyncioTestCase):
    """Unit tests for session management and persistence."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "history.db"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _init_sample_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT, checkpoint BLOB);")
        cursor.execute("CREATE TABLE writes (thread_id TEXT, task_id TEXT, channel TEXT);")
        cursor.execute("INSERT INTO checkpoints VALUES ('session-12345678', 'cp-1', 'blob');")
        cursor.execute("INSERT INTO checkpoints VALUES ('session-12345678', 'cp-2', 'blob');")
        cursor.execute("INSERT INTO checkpoints VALUES ('session-87654321', 'cp-3', 'blob');")
        conn.commit()
        conn.close()

    def test_new_session(self) -> None:
        console = Console(file=io.StringIO())
        sid = new_session(console)
        self.assertEqual(len(sid), 8)

    def test_get_available_sessions_empty_when_no_db(self) -> None:
        sessions = get_available_sessions(self.db_path)
        self.assertEqual(sessions, [])

    def test_get_available_sessions_and_list_sessions(self) -> None:
        self._init_sample_db()
        sessions = get_available_sessions(self.db_path)
        self.assertEqual(len(sessions), 2)
        step_map = {s["thread_id"]: s["steps"] for s in sessions}
        self.assertEqual(step_map["session-12345678"], 2)
        self.assertEqual(step_map["session-87654321"], 1)

        console = Console(file=io.StringIO(), record=True)
        rendered_sessions = list_sessions(console, self.db_path, current_thread_id="session-12345678")
        out = console.export_text()
        self.assertEqual(len(rendered_sessions), 2)
        self.assertIn("session-12345678", out)
        self.assertIn("current", out)

    def test_resolve_session_id(self) -> None:
        available = [
            {"thread_id": "abcdef123456"},
            {"thread_id": "abcxyz987654"},
            {"thread_id": "1234567890ab"},
        ]
        # Exact match
        self.assertEqual(resolve_session_id("abcdef123456", available), "abcdef123456")
        # Unambiguous prefix
        self.assertEqual(resolve_session_id("1234", available), "1234567890ab")
        # Ambiguous prefix
        self.assertIsNone(resolve_session_id("abc", available))
        # Nonexistent
        self.assertIsNone(resolve_session_id("nonexistent", available))
        # Empty
        self.assertIsNone(resolve_session_id("", available))

    def test_resume_session(self) -> None:
        self._init_sample_db()
        console = Console(file=io.StringIO(), record=True)

        # Successful resume by prefix
        res = resume_session(console, "session-1", db_path=self.db_path)
        self.assertEqual(res, "session-12345678")
        self.assertIn("Switched to session", console.export_text())

        # Nonexistent resume
        res2 = resume_session(console, "missing-id", db_path=self.db_path)
        self.assertIsNone(res2)

    def test_delete_session(self) -> None:
        self._init_sample_db()
        console = Console(file=io.StringIO(), record=True)

        success = delete_session(console, "session-12345678", db_path=self.db_path)
        self.assertTrue(success)

        sessions_after = get_available_sessions(self.db_path)
        self.assertEqual(len(sessions_after), 1)
        self.assertEqual(sessions_after[0]["thread_id"], "session-87654321")

    async def test_export_session(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime_mock = MagicMock()
        runtime_mock.graph = MagicMock()

        human_msg = MagicMock(type="human", content="How do I create a class?")
        ai_msg = MagicMock(type="ai", content="Use `class MyClass:` syntax.")
        mock_state = MagicMock()
        mock_state.values = {"messages": [human_msg, ai_msg]}
        runtime_mock.graph.aget_state = AsyncMock(return_value=mock_state)

        out_file = Path(self.tmpdir.name) / "export_test.md"
        exported_path = await export_session(
            console, runtime_mock, "session-1234", output_path=str(out_file), db_path=self.db_path
        )

        self.assertIsNotNone(exported_path)
        self.assertTrue(out_file.exists())
        content = out_file.read_text(encoding="utf-8")
        self.assertIn("User", content)
        self.assertIn("How do I create a class?", content)
        self.assertIn("Assistant", content)
        self.assertIn("Use `class MyClass:` syntax.", content)

    async def test_compact_session_success(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime_mock = MagicMock()
        runtime_mock.compact_context = AsyncMock(
            return_value={
                "success": True,
                "messages_summarized": 8,
                "messages_preserved": 2,
                "file_path": "/conversation_history/session-1234.md",
                "summary": "Summary of conversation",
            }
        )

        res = await compact_session(console, runtime_mock, "session-1234")
        self.assertTrue(res["success"])
        out = console.export_text()
        self.assertIn("Context compacted successfully", out)
        self.assertIn("Messages summarized: 8", out)
        self.assertIn("Recent messages preserved: 2", out)
        self.assertIn("/conversation_history/session-1234.md", out)

    async def test_compact_session_failed(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime_mock = MagicMock()
        runtime_mock.compact_context = AsyncMock(
            return_value={
                "success": False,
                "message": "Not enough messages in session to compact (at least 2 messages required).",
            }
        )

        res = await compact_session(console, runtime_mock, "session-1234")
        self.assertFalse(res["success"])
        out = console.export_text()
        self.assertIn("Not enough messages in session to compact", out)

    def test_search_sessions(self) -> None:
        console = Console(file=io.StringIO(), record=True)

        # Empty query
        res_empty = search_sessions(console, "", db_path=self.db_path)
        self.assertEqual(res_empty, [])
        self.assertIn("Please provide a search query", console.export_text())

        # No database
        res_nodb = search_sessions(console, "python", db_path=self.db_path)
        self.assertEqual(res_nodb, [])

        # Populated search
        self._init_sample_db()
        serializer = JsonPlusSerializer()
        t_msgs = [HumanMessage(content="Explain python async"), AIMessage(content="Async uses asyncio event loop")]
        typ, val = serializer.dumps_typed(t_msgs)
        chk_typ, chk_val = serializer.dumps_typed({"ts": "2026-08-20T10:00:00+00:00"})
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS checkpoints_new (thread_id TEXT, checkpoint_id TEXT, type TEXT, checkpoint BLOB);")
        cur.execute("DROP TABLE checkpoints;")
        cur.execute("ALTER TABLE checkpoints_new RENAME TO checkpoints;")
        cur.execute("INSERT INTO checkpoints VALUES ('session-12345678', 'cp-1', ?, ?);", (chk_typ, chk_val))
        cur.execute("CREATE TABLE IF NOT EXISTS writes_new (thread_id TEXT, checkpoint_id TEXT, task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB);")
        cur.execute("DROP TABLE writes;")
        cur.execute("ALTER TABLE writes_new RENAME TO writes;")
        cur.execute("INSERT INTO writes VALUES ('session-12345678', 'cp-1', 'task-1', 0, 'messages', ?, ?);", (typ, val))
        conn.commit()
        conn.close()

        console_search = Console(file=io.StringIO(), record=True)
        results = search_sessions(console_search, "async", db_path=self.db_path)
        self.assertEqual(len(results), 1)
        out = console_search.export_text()
        self.assertIn("session-", out)
        self.assertIn("async", out.lower())


if __name__ == "__main__":
    unittest.main()
