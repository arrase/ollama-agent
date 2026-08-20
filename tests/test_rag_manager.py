from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.rag.commands import (
    AmbiguousRAGDatabaseError,
    RAGContext,
    RAGDatabaseNotFoundError,
    list_rag_databases,
)
from ollama_agent.rag.manager import (
    RAGDatabaseExistsError,
    RAGError,
    RAGManager,
    RAGNotLoadedError,
)
from ollama_agent.settings.config import RAGSettings


class TestRAGManagerAndCommands(unittest.IsolatedAsyncioTestCase):
    """Unit tests for RAG manager operations and command handlers."""

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

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_chunk_text_short(self) -> None:
        text = "Hello world short"
        chunks = self.manager._chunk_text(text)
        self.assertEqual(chunks, [text])

    def test_chunk_text_monotonic_forward_progress(self) -> None:
        # Edge case where overlap >= chunk_size
        self.settings.chunk_size = 50
        self.settings.chunk_overlap = 60
        text = "Paragraph one with some detailed words.\n\nParagraph two with more details here."
        chunks = self.manager._chunk_text(text)
        self.assertTrue(len(chunks) > 1)

    def test_chunk_text_long_paragraphs(self) -> None:
        text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6."
        chunks = self.manager._chunk_text(text)
        self.assertTrue(len(chunks) >= 1)

    def test_read_file_encodings(self) -> None:
        # UTF-8
        f_utf8 = self.rag_dir / "test_utf8.txt"
        f_utf8.write_text("Hello UTF-8 world!", encoding="utf-8")
        self.assertEqual(self.manager._read_file(f_utf8), "Hello UTF-8 world!")

        # Latin-1
        f_latin = self.rag_dir / "test_latin.txt"
        f_latin.write_bytes("Café".encode("latin-1"))
        self.assertIn("Caf", self.manager._read_file(f_latin))

    def test_read_file_unsupported_extension(self) -> None:
        f_bin = self.rag_dir / "test.bin"
        f_bin.write_bytes(b"\x00\x01\x02")
        with self.assertRaises(RAGError):
            self.manager._read_file(f_bin)

    def test_database_crud(self) -> None:
        # Create database
        with patch("ollama_agent.rag.manager.QdrantClient") as mock_qdrant_cls:
            mock_client = MagicMock()
            mock_qdrant_cls.return_value = mock_client

            name = self.manager.create_database("knowledge")
            self.assertEqual(name, "knowledge")
            self.assertTrue((self.rag_dir / "knowledge").exists())

            # Creating duplicate raises
            with self.assertRaises(RAGDatabaseExistsError):
                self.manager.create_database("knowledge")

            # List databases
            dbs = self.manager.list_databases()
            self.assertEqual(len(dbs), 1)
            self.assertEqual(dbs[0]["name"], "knowledge")

            # Load database
            self.manager.load_database("knowledge")
            self.assertEqual(self.manager.current_database, "knowledge")

            # Unload database
            self.manager.unload()
            self.assertIsNone(self.manager.current_database)

            # Delete database
            deleted = self.manager.delete_database("knowledge")
            self.assertTrue(deleted)
            self.assertFalse((self.rag_dir / "knowledge").exists())

    def test_rag_context_find_or_exit(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_databases.return_value = [
            {"name": "docs", "active": False, "chunks": 10},
            {"name": "docs_v2", "active": False, "chunks": 20},
            {"name": "code", "active": False, "chunks": 5},
        ]
        ctx = RAGContext(rag_manager=mock_mgr, console=Console(record=True))

        # Exact match should take priority even if a prefix match exists
        self.assertEqual(ctx._find_or_exit("docs"), "docs")
        self.assertEqual(ctx._find_or_exit("docs_v2"), "docs_v2")
        self.assertEqual(ctx._find_or_exit("code"), "code")

        # Nonexistent DB
        with self.assertRaises(RAGDatabaseNotFoundError):
            ctx._find_or_exit("nonexistent")

        # Ambiguous prefix
        mock_mgr.list_databases.return_value = [
            {"name": "alpha_1", "active": False, "chunks": 1},
            {"name": "alpha_2", "active": False, "chunks": 2},
        ]
        with self.assertRaises(AmbiguousRAGDatabaseError):
            ctx._find_or_exit("alpha")
