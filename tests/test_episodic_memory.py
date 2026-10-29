from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Overwrite

from ollama_agent.agent.builtin_tools import (
    get_tool_timeout,
    search_past_conversations,
    set_active_thread_id,
    set_tool_timeout,
)
from ollama_agent.agent.episodic_memory import (
    HistoryError,
    format_iso_timestamp,
    format_past_conversations_context,
    load_past_conversations,
    load_past_user_prompts,
    load_session_timestamps,
    search_past_conversations_in_db,
)


class TestEpisodicMemory(unittest.IsolatedAsyncioTestCase):
    """Unit tests for episodic memory loading, keyword search, and formatting."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "history.db"
        self.serializer = JsonPlusSerializer()
        set_active_thread_id("")

    def tearDown(self) -> None:
        set_active_thread_id("")
        self.tmpdir.cleanup()

    def _seed_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE checkpoints (
                thread_id TEXT,
                checkpoint_ns TEXT DEFAULT '',
                checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB
            );"""
        )
        cur.execute(
            """CREATE TABLE writes (
                thread_id TEXT,
                checkpoint_ns TEXT DEFAULT '',
                checkpoint_id TEXT,
                task_id TEXT,
                idx INTEGER,
                channel TEXT,
                type TEXT,
                value BLOB
            );"""
        )

        # Thread 1: FastApi and Docker discussion (2026-08-20)
        t1_chk = {"ts": "2026-08-20T10:00:00+00:00"}
        chk_typ1, chk_val1 = self.serializer.dumps_typed(t1_chk)
        cur.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint) VALUES (?, ?, ?, ?)",
            ("thread-100", "cp-1", chk_typ1, chk_val1),
        )
        t1_msgs = [
            HumanMessage(content="How do I dockerize a FastAPI application?"),
            AIMessage(content="You should create a Dockerfile using python:3.11-slim and install uvicorn."),
        ]
        typ1, val1 = self.serializer.dumps_typed(t1_msgs)
        cur.execute(
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("thread-100", "cp-1", "task-1", 0, "messages", typ1, val1),
        )

        # Thread 2: React frontend setup (2026-08-21)
        t2_chk = {"ts": "2026-08-21T14:30:00+00:00"}
        chk_typ2, chk_val2 = self.serializer.dumps_typed(t2_chk)
        cur.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint) VALUES (?, ?, ?, ?)",
            ("thread-200", "cp-2", chk_typ2, chk_val2),
        )
        t2_msgs = [
            HumanMessage(content="Set up a React Vite project with TypeScript."),
            AIMessage(content="Run npm create vite@latest my-app -- --template react-ts."),
        ]
        typ2, val2 = self.serializer.dumps_typed(t2_msgs)
        cur.execute(
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("thread-200", "cp-2", "task-2", 0, "messages", typ2, val2),
        )

        # Thread 3: PostgreSQL and Alembic migrations (2026-08-22)
        t3_chk = {"ts": "2026-08-22T08:15:00+00:00"}
        chk_typ3, chk_val3 = self.serializer.dumps_typed(t3_chk)
        cur.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint) VALUES (?, ?, ?, ?)",
            ("thread-300", "cp-3", chk_typ3, chk_val3),
        )
        t3_msgs = [
            HumanMessage(content="How to handle PostgreSQL migrations with Alembic?"),
            AIMessage(content="Run alembic revision --autogenerate and alembic upgrade head."),
            ToolMessage(content="Migration output: success", tool_call_id="call-1"),
        ]
        typ3, val3 = self.serializer.dumps_typed(t3_msgs)
        cur.execute(
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("thread-300", "cp-3", "task-3", 0, "messages", typ3, val3),
        )

        conn.commit()
        conn.close()

    def test_load_past_conversations_empty_db(self) -> None:
        convs = load_past_conversations(self.db_path)
        assert convs == {}

    def test_load_past_user_prompts_empty_db(self) -> None:
        prompts = load_past_user_prompts(self.db_path)
        assert prompts == []

    def test_load_past_user_prompts_with_data(self) -> None:
        self._seed_db()
        prompts = load_past_user_prompts(self.db_path)
        assert prompts == [
            "How do I dockerize a FastAPI application?",
            "Set up a React Vite project with TypeScript.",
            "How to handle PostgreSQL migrations with Alembic?",
        ]

    def test_load_past_conversations_with_data(self) -> None:
        self._seed_db()
        convs = load_past_conversations(self.db_path)
        assert len(convs) == 3
        assert "thread-100" in convs
        assert "thread-200" in convs
        assert "thread-300" in convs
        assert len(convs["thread-100"]["messages"]) == 2
        assert len(convs["thread-300"]["messages"]) == 3
        assert "2026-08-20" in convs["thread-100"]["formatted_date"]

    def test_load_past_conversations_exclude_thread(self) -> None:
        self._seed_db()
        convs = load_past_conversations(self.db_path, exclude_thread_id="thread-200")
        assert len(convs) == 2
        assert "thread-200" not in convs

    def test_search_past_conversations_in_db(self) -> None:
        self._seed_db()

        # Search for FastAPI
        results = search_past_conversations_in_db("fastapi docker", db_path=self.db_path)
        assert len(results) >= 1
        assert results[0]["thread_id"] == "thread-100"
        assert "FastAPI" in results[0]["snippets"][0]
        assert "2026-08-20" in results[0]["formatted_date"]

        # Search for Alembic
        results_alembic = search_past_conversations_in_db("alembic", db_path=self.db_path)
        assert len(results_alembic) == 1
        assert results_alembic[0]["thread_id"] == "thread-300"

        # Search by date
        results_date = search_past_conversations_in_db("2026-08-21", db_path=self.db_path)
        assert len(results_date) == 1
        assert results_date[0]["thread_id"] == "thread-200"

        # Search with no matching terms returns empty list
        results_none = search_past_conversations_in_db("quantum computing", db_path=self.db_path)
        assert results_none == []

        # Search with empty query returns empty list
        assert search_past_conversations_in_db("", db_path=self.db_path) == []

    def test_load_past_conversations_thread_without_checkpoint(self) -> None:
        self._seed_db()
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        t_msgs = [HumanMessage(content="Orphan message")]
        typ, val = self.serializer.dumps_typed(t_msgs)
        cur.execute(
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("orphan-thread", "cp-orphan", "task-orphan", 0, "messages", typ, val),
        )
        conn.commit()
        conn.close()

        conversations = load_past_conversations(self.db_path)
        assert "orphan-thread" not in conversations
        assert "thread-100" in conversations

    def test_search_past_conversations_limit(self) -> None:
        self._seed_db()
        results = search_past_conversations_in_db("how to", db_path=self.db_path, limit=1)
        assert len(results) == 1

    def test_format_past_conversations_context(self) -> None:
        # Empty results formatting
        empty_formatted = format_past_conversations_context([])
        assert "No relevant past conversations found" in empty_formatted

        # Populated results formatting
        sample_results = [
            {
                "thread_id": "thread-12345678",
                "score": 3,
                "formatted_date": "2026-08-21 14:30 UTC",
                "snippets": ["[User]: How to configure CORS?", "[Assistant]: Use CORSMiddleware."],
                "total_messages": 2,
            }
        ]
        formatted = format_past_conversations_context(sample_results)
        assert "Session #1 (thread-1)" in formatted
        assert "2026-08-21 14:30 UTC" in formatted
        assert "[User]: How to configure CORS?" in formatted
        assert "[Assistant]: Use CORSMiddleware." in formatted

    async def test_search_past_conversations_tool(self) -> None:
        self._seed_db()
        set_active_thread_id("thread-100")

        with patch("ollama_agent.agent.episodic_memory.HISTORY_DB_PATH", self.db_path):
            output = await search_past_conversations.ainvoke({"query": "React Vite"})
            assert "react" in output.lower()
            assert "thread-2" in output
            assert "2026-08-21" in output

            # Excluded active thread should not appear
            output_active = await search_past_conversations.ainvoke({"query": "FastAPI"})
            assert "No relevant past conversations found" in output_active

            # Non-positive limit raises ValueError
            with pytest.raises(ValueError):
                await search_past_conversations.ainvoke({"query": "x", "limit": 0})
            with pytest.raises(ValueError):
                await search_past_conversations.ainvoke({"query": "React Vite", "limit": -5})

    def test_tool_timeout_setter_getter(self) -> None:
        set_tool_timeout(45)
        assert get_tool_timeout() == 45

        with pytest.raises(ValueError):
            set_tool_timeout(0)

        with pytest.raises(ValueError):
            set_tool_timeout(-5)

    def test_format_iso_timestamp(self) -> None:
        # Naive timestamp should not shift timezone
        assert format_iso_timestamp("2026-08-20T10:00:00") == "2026-08-20 10:00 UTC"
        # Timezone-aware timestamp should be converted to UTC
        assert format_iso_timestamp("2026-08-20T12:00:00+02:00") == "2026-08-20 10:00 UTC"

    def test_load_with_scalar_message(self) -> None:
        # Seed DB with a single BaseMessage scalar (not inside a list)
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE checkpoints (
                thread_id TEXT,
                checkpoint_ns TEXT DEFAULT '',
                checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB
            );"""
        )
        cur.execute(
            """CREATE TABLE writes (
                thread_id TEXT,
                checkpoint_ns TEXT DEFAULT '',
                checkpoint_id TEXT,
                task_id TEXT,
                idx INTEGER,
                channel TEXT,
                type TEXT,
                value BLOB
            );"""
        )
        t_chk = {"ts": "2026-08-20T10:00:00"}
        chk_typ, chk_val = self.serializer.dumps_typed(t_chk)
        cur.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint) VALUES (?, ?, ?, ?)",
            ("thread-scalar", "cp-scalar", chk_typ, chk_val),
        )
        # Dump scalar message instead of list
        scalar_msg = HumanMessage(content="Scalar prompt question")
        typ, val = self.serializer.dumps_typed(scalar_msg)
        cur.execute(
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("thread-scalar", "cp-scalar", "task-scalar", 0, "messages", typ, val),
        )
        conn.commit()
        conn.close()

        prompts = load_past_user_prompts(self.db_path)
        assert prompts == ["Scalar prompt question"]

        convs = load_past_conversations(self.db_path)
        assert "thread-scalar" in convs
        assert len(convs["thread-scalar"]["messages"]) == 1
        assert convs["thread-scalar"]["messages"][0].content == "Scalar prompt question"

    def test_load_session_timestamps(self) -> None:
        assert load_session_timestamps(self.db_path) == []
        self._seed_db()
        assert load_session_timestamps(self.db_path) == [
            ("thread-100", "2026-08-20 10:00 UTC"),
            ("thread-200", "2026-08-21 14:30 UTC"),
            ("thread-300", "2026-08-22 08:15 UTC"),
        ]

    def test_corrupted_checkpoint_raises_history_error(self) -> None:
        self._seed_db()
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        chk_typ, chk_val = self.serializer.dumps_typed({"no_ts_key": 123})
        cur.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint) VALUES (?, ?, ?, ?)",
            ("thread-bad", "cp-bad", chk_typ, chk_val),
        )
        conn.commit()
        conn.close()

        with pytest.raises(HistoryError):
            load_past_conversations(self.db_path)

    def test_overwrite_and_message_deduplication(self) -> None:
        self._seed_db()
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        chk_typ, chk_val = self.serializer.dumps_typed({"ts": "2026-08-23T12:00:00"})
        cur.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint) VALUES (?, ?, ?, ?)",
            ("thread-ow", "cp-ow", chk_typ, chk_val),
        )
        msg1 = HumanMessage(content="Before overwrite", id="msg-1")
        msg2 = HumanMessage(content="After overwrite", id="msg-2")
        msg2_dup = HumanMessage(content="After overwrite", id="msg-2")
        t1, v1 = self.serializer.dumps_typed([msg1])
        t2, v2 = self.serializer.dumps_typed(Overwrite(value=[msg2]))
        t3, v3 = self.serializer.dumps_typed([msg2_dup])
        cur.executemany(
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("thread-ow", "cp-ow", "t1", 0, "messages", t1, v1),
                ("thread-ow", "cp-ow", "t2", 1, "messages", t2, v2),
                ("thread-ow", "cp-ow", "t3", 2, "messages", t3, v3),
            ],
        )
        conn.commit()
        conn.close()

        prompts = load_past_user_prompts(self.db_path)
        assert "After overwrite" in prompts
        convs = load_past_conversations(self.db_path)
        assert [m.content for m in convs["thread-ow"]["messages"]] == ["After overwrite"]


if __name__ == "__main__":
    unittest.main()
