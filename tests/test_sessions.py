from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from rich.console import Console

from ollama_agent.agent.episodic_memory import HistoryError
from ollama_agent.interfaces.commands.sessions import (
    delete_session,
    export_session,
    get_available_sessions,
    is_current,
    list_sessions,
    new_session,
    resume_session,
    search_sessions,
)

_serializer = JsonPlusSerializer()


class TestSessionCommands(unittest.IsolatedAsyncioTestCase):
    """Unit tests for session management and persistence."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "history.db"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _init_sample_db(self) -> None:
        """Create a history DB matching the langgraph AsyncSqliteSaver schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT,
                task_id TEXT,
                idx INTEGER,
                channel TEXT,
                type TEXT,
                value BLOB
            );
            """
        )
        for tid, ts in [
            ("session-12345678", "2026-01-01T10:00:00+00:00"),
            ("session-12345678", "2026-01-01T10:05:00+00:00"),
            ("session-87654321", "2026-01-02T09:00:00+00:00"),
        ]:
            typ, blob = _serializer.dumps_typed({"v": 1, "ts": ts})
            cursor.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint) "
                "VALUES (?, '', ?, ?, ?)",
                (tid, f"cp-{tid}-{ts}", typ, blob),
            )
        conn.commit()
        conn.close()

    def test_new_session(self) -> None:
        runtime = MagicMock(thread_id="old", last_context_tokens=100)
        sid = new_session(runtime)
        assert len(sid) == 8
        assert runtime.thread_id == sid
        assert runtime.last_context_tokens == 0

    def test_get_available_sessions_empty_when_no_db(self) -> None:
        sessions = get_available_sessions(self.db_path)
        assert sessions == []

    def test_get_available_sessions_corrupted_db_raises_history_error(self) -> None:
        self.db_path.write_text("this is not a valid sqlite database", encoding="utf-8")
        with pytest.raises(HistoryError):
            get_available_sessions(self.db_path)

    def test_get_available_sessions_and_list_sessions(self) -> None:
        self._init_sample_db()
        sessions = get_available_sessions(self.db_path)
        assert len(sessions) == 2
        step_map = {s["thread_id"]: s["steps"] for s in sessions}
        assert step_map["session-12345678"] == 2
        assert step_map["session-87654321"] == 1

        console = Console(file=io.StringIO(), record=True)
        list_sessions(console, self.db_path, current_thread_id="session-12345678")
        out = console.export_text()
        assert "session-12345678" in out
        assert "session-87654321" in out
        assert "current" in out

    def test_get_available_sessions_missing_timestamp(self) -> None:
        self._init_sample_db()
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        typ, blob = _serializer.dumps_typed({"v": 1})
        cursor.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint) "
            "VALUES (?, '', ?, ?, ?)",
            ("session-nots", "cp-session-nots", typ, blob),
        )
        conn.commit()
        conn.close()

        with pytest.raises(HistoryError):
            get_available_sessions(self.db_path)

    def test_resume_session(self) -> None:
        self._init_sample_db()
        console = Console(file=io.StringIO(), record=True)

        # Successful resume by prefix and exact match
        res = resume_session(console, "session-1", db_path=self.db_path)
        assert res == "session-12345678"
        res_exact = resume_session(console, "session-87654321", db_path=self.db_path)
        assert res_exact == "session-87654321"

        # Ambiguous prefix
        res_amb = resume_session(console, "session-", db_path=self.db_path)
        assert res_amb is None
        assert "Ambiguous session ID" in console.export_text()

        # Nonexistent / empty resume
        res2 = resume_session(console, "missing-id", db_path=self.db_path)
        assert res2 is None
        res_empty = resume_session(console, "", db_path=self.db_path)
        assert res_empty is None

    def test_delete_session(self) -> None:
        self._init_sample_db()
        console = Console(file=io.StringIO(), record=True)

        success = delete_session(console, "session-12345678", db_path=self.db_path)
        assert success

        sessions_after = get_available_sessions(self.db_path)
        assert len(sessions_after) == 1
        assert sessions_after[0]["thread_id"] == "session-87654321"

    async def test_export_session(self) -> None:
        self._init_sample_db()
        console = Console(file=io.StringIO(), record=True)
        runtime_mock = MagicMock()

        human_msg = MagicMock(type="human", content="How do I create a class?")
        ai_msg = MagicMock(type="ai", content="Use `class MyClass:` syntax.")
        runtime_mock.get_thread_messages = AsyncMock(return_value=[human_msg, ai_msg])

        out_file = Path(self.tmpdir.name) / "export_test.md"
        exported_path = await export_session(
            console, runtime_mock, "session-1234", output_path=str(out_file), db_path=self.db_path
        )

        assert exported_path is not None
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "User" in content
        assert "How do I create a class?" in content
        assert "Assistant" in content
        assert "Use `class MyClass:` syntax." in content

    async def test_export_session_not_found(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime_mock = MagicMock(stealth_mode=False)
        runtime_mock.get_thread_messages = AsyncMock(return_value=[])

        res = await export_session(console, runtime_mock, "nonexistent-id", db_path=self.db_path)

        assert res is None
        runtime_mock.get_thread_messages.assert_not_called()
        assert "not found" in console.export_text()

    def test_search_sessions(self) -> None:
        console = Console(file=io.StringIO(), record=True)

        # Empty query
        search_sessions(console, "", db_path=self.db_path)
        assert "Please provide a search query" in console.export_text()

        # No database
        console_nodb = Console(file=io.StringIO(), record=True)
        search_sessions(console_nodb, "python", db_path=self.db_path)
        assert "No sessions found" in console_nodb.export_text()

        # Populated search
        self._init_sample_db()
        serializer = JsonPlusSerializer()
        t_msgs = [HumanMessage(content="Explain python async"), AIMessage(content="Async uses asyncio event loop")]
        typ, val = serializer.dumps_typed(t_msgs)
        chk_typ, chk_val = serializer.dumps_typed({"ts": "2026-08-20T10:00:00+00:00"})
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints_new "
            "(thread_id TEXT, checkpoint_id TEXT, type TEXT, checkpoint BLOB);"
        )
        cur.execute("DROP TABLE checkpoints;")
        cur.execute("ALTER TABLE checkpoints_new RENAME TO checkpoints;")
        cur.execute("INSERT INTO checkpoints VALUES ('session-12345678', 'cp-1', ?, ?);", (chk_typ, chk_val))
        cur.execute(
            "CREATE TABLE IF NOT EXISTS writes_new (thread_id TEXT, checkpoint_id TEXT, task_id TEXT, "
            "idx INTEGER, channel TEXT, type TEXT, value BLOB);"
        )
        cur.execute("DROP TABLE writes;")
        cur.execute("ALTER TABLE writes_new RENAME TO writes;")
        cur.execute(
            "INSERT INTO writes VALUES ('session-12345678', 'cp-1', 'task-1', 0, 'messages', ?, ?);", (typ, val)
        )
        conn.commit()
        conn.close()

        console_search = Console(file=io.StringIO(), record=True)
        search_sessions(console_search, "async", db_path=self.db_path)
        out = console_search.export_text()
        assert "session-" in out
        assert "async" in out.lower()

    def test_is_current_bidirectional_and_empty(self) -> None:
        assert is_current("12345678", "12345678")
        assert is_current("12345678", "1234")
        assert is_current("1234", "12345678")
        assert not is_current("", "1234")
        assert not is_current("1234", "")
        assert not is_current("abcd", "1234")

    async def test_export_session_with_tool_calls_and_nested_dir(self) -> None:
        self._init_sample_db()
        console = Console(file=io.StringIO(), record=True)
        runtime_mock = MagicMock()

        ai_msg = MagicMock(
            type="ai",
            content="I am calling a tool.",
            tool_calls=[{"name": "test_tool", "args": {"arg1": "val1"}}],
        )
        tool_msg = MagicMock(type="tool", name="test_tool", content="tool output result")
        runtime_mock.get_thread_messages = AsyncMock(return_value=[ai_msg, tool_msg])

        out_file = Path(self.tmpdir.name) / "nested" / "dir" / "export_test.md"
        exported_path = await export_session(
            console, runtime_mock, "session-1234", output_path=str(out_file), db_path=self.db_path
        )

        assert exported_path is not None
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "Tool: test_tool" in content
        assert "val1" in content
        assert "tool output result" in content

    async def test_delete_and_export_session_ambiguous_prefix(self) -> None:
        self._init_sample_db()
        console_del = Console(file=io.StringIO(), record=True)
        assert not delete_session(console_del, "session-", db_path=self.db_path)
        assert "Ambiguous session ID" in console_del.export_text()

        console_exp = Console(file=io.StringIO(), record=True)
        runtime_mock = MagicMock(stealth_mode=False)
        res = await export_session(console_exp, runtime_mock, "session-", db_path=self.db_path)
        assert res is None
        assert "Ambiguous session ID" in console_exp.export_text()

    async def test_uncreatable_parent_is_reported_not_raised(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        session_id = "abc12345"
        runtime = MagicMock(stealth_mode=False)
        runtime.get_thread_messages = AsyncMock(return_value=[MagicMock(type="human", content="hi")])
        # A path under an existing *file* cannot have its parent created.
        blocker = Path(self.tmpdir.name) / "blocker"
        blocker.write_text("x", encoding="utf-8")
        target = str(blocker / "sub" / "out.md")
        with patch("ollama_agent.interfaces.commands.sessions.get_available_sessions") as sessions:
            sessions.return_value = [{"thread_id": session_id, "steps": 1, "timestamp": ""}]
            result = await export_session(console, runtime, session_id, output_path=target)
        assert result is None
        assert "Failed to export" in console.export_text()


if __name__ == "__main__":
    unittest.main()
