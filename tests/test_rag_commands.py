from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.rag.commands import (
    AmbiguousRAGDatabaseError,
    RAGContext,
    RAGDatabaseNotFoundError,
    add_rag_directory,
    add_rag_file,
    create_rag_database,
    delete_rag_database,
    list_rag_databases,
    load_rag_database,
    show_rag_status,
    unload_rag_database,
)
from ollama_agent.rag.manager import RAGManager
from ollama_agent.settings import RAGSettings


class TestRAGCommands(unittest.IsolatedAsyncioTestCase):
    """Unit tests for RAG command operations and resolution."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rag_dir = Path(self.temp_dir.name)
        self.settings = RAGSettings(
            rag_dir=str(self.rag_dir),
            chunk_size=100,
            chunk_overlap=20,
            embedding_dims=4,
        )
        self.manager = RAGManager(self.settings)
        self.console = Console(file=io.StringIO(), record=True)
        self.ctx = RAGContext(rag_manager=self.manager, console=self.console)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_rag_context_resolve_database(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_database_names.return_value = ["docs", "docs_v2", "code"]
        ctx = RAGContext(rag_manager=mock_mgr, console=self.console)

        # Exact match should take priority even if a prefix match exists
        self.assertEqual(ctx.resolve_database("docs"), "docs")
        self.assertEqual(ctx.resolve_database("docs_v2"), "docs_v2")
        self.assertEqual(ctx.resolve_database("code"), "code")

        # Nonexistent DB
        with self.assertRaises(RAGDatabaseNotFoundError):
            ctx.resolve_database("nonexistent")

        # Ambiguous prefix
        mock_mgr.list_database_names.return_value = ["alpha_1", "alpha_2"]
        with self.assertRaises(AmbiguousRAGDatabaseError):
            ctx.resolve_database("alpha")

    def test_database_resolution_case_insensitive_exact_priority(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_database_names.return_value = ["Alpha", "alphabet"]
        ctx = RAGContext(rag_manager=mock_mgr, console=self.console)
        # "alpha" matches "Alpha" exactly (case-insensitive) instead of matching "alphabet" as prefix
        self.assertEqual(ctx.resolve_database("alpha"), "Alpha")

    def test_database_resolution_empty_name_raises(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_database_names.return_value = ["docs"]
        ctx = RAGContext(rag_manager=mock_mgr, console=self.console)
        with self.assertRaises(RAGDatabaseNotFoundError) as exc:
            ctx.resolve_database("")
        self.assertIn("cannot be empty", str(exc.exception))
        with self.assertRaises(RAGDatabaseNotFoundError) as exc:
            ctx.resolve_database("   ")
        self.assertIn("cannot be empty", str(exc.exception))

    def test_rag_command_handlers(self) -> None:
        # Empty databases
        list_rag_databases(self.ctx)
        self.assertIn("No RAG databases found", self.console.export_text())

        # Status not loaded
        show_rag_status(self.ctx)
        self.assertIn("No RAG database is currently loaded", self.console.export_text())

        # Create, Load, Show, Unload, Delete
        with patch("ollama_agent.rag.manager.QdrantClient"):
            create_rag_database(self.ctx, "manual_db")
            load_rag_database(self.ctx, "manual_db")
            self.assertEqual(self.manager.current_database, "manual_db")

            show_rag_status(self.ctx)
            self.assertIn("Active RAG database", self.console.export_text())

            unload_rag_database(self.ctx)
            self.assertIsNone(self.manager.current_database)

            delete_rag_database(self.ctx, "manual_db")
            self.assertFalse((self.rag_dir / "manual_db").exists())

    def test_unload_when_not_loaded(self) -> None:
        unload_rag_database(self.ctx)
        self.assertIn("No RAG database is currently loaded", self.console.export_text())

    async def test_add_rag_file(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.add_file = AsyncMock(return_value={"file": "doc.txt", "chunks": 3})
        ctx = RAGContext(rag_manager=mock_mgr, console=self.console)
        await add_rag_file(ctx, "doc.txt")
        self.assertIn("Added to RAG: doc.txt (3 chunks)", self.console.export_text())

    async def test_add_rag_directory_styling(self) -> None:
        mock_mgr = MagicMock()
        console = Console(file=io.StringIO(), record=True)
        ctx = RAGContext(rag_manager=mock_mgr, console=console)

        # 1. All succeeded -> green checkmark
        mock_mgr.add_directory = AsyncMock(return_value={"added": 3, "skipped": 0, "failed": 0})
        await add_rag_directory(ctx, "dir")
        self.assertIn("✓", console.export_text())

        # 2. Partial failures -> yellow warning
        console = Console(file=io.StringIO(), record=True)
        ctx = RAGContext(rag_manager=mock_mgr, console=console)
        mock_mgr.add_directory = AsyncMock(return_value={"added": 2, "skipped": 0, "failed": 1})
        await add_rag_directory(ctx, "dir")
        self.assertIn("⚠", console.export_text())

        # 3. Total failure -> red cross
        console = Console(file=io.StringIO(), record=True)
        ctx = RAGContext(rag_manager=mock_mgr, console=console)
        mock_mgr.add_directory = AsyncMock(return_value={"added": 0, "skipped": 0, "failed": 2})
        await add_rag_directory(ctx, "dir")
        self.assertIn("✕", console.export_text())


if __name__ == "__main__":
    unittest.main()
