from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ollama_agent.agent.builtin_tools import (
    get_tool_timeout,
    search_past_conversations,
    set_active_thread_id,
    set_tool_timeout,
)
from ollama_agent.agent.episodic_memory import (
    format_past_conversations_context,
    load_past_conversations,
    load_past_user_prompts,
    search_past_conversations_in_db,
)
from ollama_agent.i18n import set_locale


class TestEpisodicMemory(unittest.IsolatedAsyncioTestCase):
    """Unit tests for episodic memory loading, keyword search, and formatting."""

    def setUp(self) -> None:
        set_locale("en")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "history.db"
        self.serializer = JsonPlusSerializer()
        set_active_thread_id("")

    def tearDown(self) -> None:
        set_active_thread_id("")
        self.tmpdir.cleanup()
        set_locale("en")

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
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("thread-300", "cp-3", "task-3", 0, "messages", typ3, val3),
        )

        conn.commit()
        conn.close()

    def test_load_past_conversations_empty_db(self) -> None:
        convs = load_past_conversations(self.db_path)
        self.assertEqual(convs, {})

    def test_load_past_user_prompts_empty_db(self) -> None:
        prompts = load_past_user_prompts(self.db_path)
        self.assertEqual(prompts, [])

    def test_load_past_user_prompts_with_data(self) -> None:
        self._seed_db()
        prompts = load_past_user_prompts(self.db_path)
        self.assertEqual(
            prompts,
            [
                "How do I dockerize a FastAPI application?",
                "Set up a React Vite project with TypeScript.",
                "How to handle PostgreSQL migrations with Alembic?",
            ],
        )

    def test_load_past_conversations_with_data(self) -> None:
        self._seed_db()
        convs = load_past_conversations(self.db_path)
        self.assertEqual(len(convs), 3)
        self.assertIn("thread-100", convs)
        self.assertIn("thread-200", convs)
        self.assertIn("thread-300", convs)
        self.assertEqual(len(convs["thread-100"]["messages"]), 2)
        self.assertEqual(len(convs["thread-300"]["messages"]), 3)
        self.assertIn("2026-08-20", convs["thread-100"]["formatted_date"])

    def test_load_past_conversations_exclude_thread(self) -> None:
        self._seed_db()
        convs = load_past_conversations(self.db_path, exclude_thread_id="thread-200")
        self.assertEqual(len(convs), 2)
        self.assertNotIn("thread-200", convs)

    def test_search_past_conversations_in_db(self) -> None:
        self._seed_db()

        # Search for FastAPI
        results = search_past_conversations_in_db("fastapi docker", db_path=self.db_path)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["thread_id"], "thread-100")
        self.assertIn("FastAPI", results[0]["snippets"][0])
        self.assertIn("2026-08-20", results[0]["formatted_date"])

        # Search for Alembic
        results_alembic = search_past_conversations_in_db("alembic", db_path=self.db_path)
        self.assertEqual(len(results_alembic), 1)
        self.assertEqual(results_alembic[0]["thread_id"], "thread-300")

        # Search by date
        results_date = search_past_conversations_in_db("2026-08-21", db_path=self.db_path)
        self.assertEqual(len(results_date), 1)
        self.assertEqual(results_date[0]["thread_id"], "thread-200")

        # Search with no matching terms returns empty list
        results_none = search_past_conversations_in_db("quantum computing", db_path=self.db_path)
        self.assertEqual(results_none, [])

        # Search with empty query returns empty list
        self.assertEqual(search_past_conversations_in_db("", db_path=self.db_path), [])

    def test_load_past_conversations_thread_without_checkpoint(self) -> None:
        self._seed_db()
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        t_msgs = [HumanMessage(content="Orphan message")]
        typ, val = self.serializer.dumps_typed(t_msgs)
        cur.execute(
            "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("orphan-thread", "cp-orphan", "task-orphan", 0, "messages", typ, val),
        )
        conn.commit()
        conn.close()

        conversations = load_past_conversations(self.db_path)
        self.assertNotIn("orphan-thread", conversations)
        self.assertIn("thread-100", conversations)

    def test_search_past_conversations_limit(self) -> None:
        self._seed_db()
        results = search_past_conversations_in_db("how to", db_path=self.db_path, limit=1)
        self.assertEqual(len(results), 1)

    def test_format_past_conversations_context(self) -> None:
        # Empty results formatting
        empty_formatted = format_past_conversations_context([])
        self.assertIn("No relevant past conversations found", empty_formatted)

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
        self.assertIn("Session #1 (thread-1)", formatted)
        self.assertIn("2026-08-21 14:30 UTC", formatted)
        self.assertIn("[User]: How to configure CORS?", formatted)
        self.assertIn("[Assistant]: Use CORSMiddleware.", formatted)

    async def test_search_past_conversations_tool(self) -> None:
        self._seed_db()
        set_active_thread_id("thread-100")

        with patch("ollama_agent.agent.episodic_memory.HISTORY_DB_PATH", self.db_path):
            output = await search_past_conversations.ainvoke({"query": "React Vite"})
            self.assertIn("react", output.lower())
            self.assertIn("thread-2", output)
            self.assertIn("2026-08-21", output)

            # Excluded active thread should not appear
            output_active = await search_past_conversations.ainvoke({"query": "FastAPI"})
            self.assertIn("No relevant past conversations found", output_active)

            # Non-positive limit raises ValueError
            with self.assertRaises(ValueError):
                await search_past_conversations.ainvoke({"query": "React Vite", "limit": -5})

    def test_tool_timeout_setter_getter(self) -> None:
        set_tool_timeout(45)
        self.assertEqual(get_tool_timeout(), 45)

        with self.assertRaises(ValueError):
            set_tool_timeout(0)

        with self.assertRaises(ValueError):
            set_tool_timeout(-10)


if __name__ == "__main__":
    unittest.main()
