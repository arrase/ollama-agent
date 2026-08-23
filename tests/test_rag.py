from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_agent.rag import (
    AmbiguousRAGDatabaseError,
    RAGContext,
    RAGDatabaseNotFoundError,
    RAGError,
    RAGManager,
    RAGSettings,
)


class TestRAGComponents(unittest.TestCase):
    """Unit tests for RAG manager utilities and commands."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rag_dir = Path(self.temp_dir.name)
        self.settings = RAGSettings(rag_dir=str(self.rag_dir), chunk_size=100, chunk_overlap=20)
        self.mgr = RAGManager(self.settings)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_validate_database_name(self) -> None:
        self.assertEqual(RAGManager._validate_name("docs_v1"), "docs_v1")
        with self.assertRaises(RAGError):
            RAGManager._validate_name("invalid name with spaces")

    def test_generate_point_id_deterministic(self) -> None:
        id1 = RAGManager._generate_point_id("/path/to/file.py", 0)
        id2 = RAGManager._generate_point_id("/path/to/file.py", 0)
        id3 = RAGManager._generate_point_id("/path/to/file.py", 1)

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_chunk_text_splits_overlapping(self) -> None:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."
        chunks = self.mgr._chunk_text(text)
        self.assertTrue(len(chunks) >= 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)

    def test_read_file_supported_encodings(self) -> None:
        file = self.rag_dir / "sample.txt"
        file.write_text("Testing UTF-8 encoding support", encoding="utf-8")
        content = self.mgr._read_file(file)
        self.assertEqual(content, "Testing UTF-8 encoding support")

    def test_read_file_unsupported_extension_raises(self) -> None:
        file = self.rag_dir / "sample.png"
        file.write_bytes(b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(RAGError):
            self.mgr._read_file(file)

    def test_rag_context_find_or_exit_not_found(self) -> None:
        ctx = RAGContext(rag_manager=self.mgr)
        with self.assertRaises(RAGDatabaseNotFoundError):
            ctx._find_or_exit("nonexistent_db")

    def test_rag_context_find_or_exit_case_insensitive(self) -> None:
        ctx = RAGContext(rag_manager=self.mgr)
        # Create a database
        self.mgr.create_database("MyDatabase")
        # Find it using lowercase
        found = ctx._find_or_exit("mydatabase")
        self.assertEqual(found, "MyDatabase")


if __name__ == "__main__":
    unittest.main()
