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
    create_rag_database,
    delete_rag_database,
    list_rag_databases,
    load_rag_database,
    show_rag_status,
    unload_rag_database,
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

    def test_chunk_text_invalid_configuration_raises(self) -> None:
        self.settings.chunk_size = 0
        with self.assertRaises(RAGError):
            self.manager._chunk_text("some text")
        self.settings.chunk_size = 50
        self.settings.chunk_overlap = -1
        with self.assertRaises(RAGError):
            self.manager._chunk_text("some text")

    def test_chunk_text_monotonic_forward_progress(self) -> None:
        self.settings.chunk_size = 50
        self.settings.chunk_overlap = 60
        with self.assertRaises(RAGError):
            self.manager._chunk_text("Paragraph one with some detailed words.\n\nParagraph two with more details here.")

    def test_chunk_text_long_paragraphs(self) -> None:
        text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6."
        chunks = self.manager._chunk_text(text)
        self.assertTrue(len(chunks) >= 1)

    def test_read_file_encodings(self) -> None:
        # UTF-8
        f_utf8 = self.rag_dir / "test_utf8.txt"
        f_utf8.write_text("Hello UTF-8 world!", encoding="utf-8")
        self.assertEqual(self.manager._read_file(f_utf8), "Hello UTF-8 world!")

        # Non-UTF-8 content fails loudly
        f_latin = self.rag_dir / "test_latin.txt"
        f_latin.write_bytes("Café".encode("latin-1"))
        with self.assertRaises(RAGError):
            self.manager._read_file(f_latin)

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
            self.assertEqual(self.manager.list_database_names(), ["knowledge"])

            # Load database
            self.manager.load_database("knowledge")
            self.assertEqual(self.manager.current_database, "knowledge")

            # Unload database
            self.manager.unload()
            self.assertIsNone(self.manager.current_database)

            # Delete database
            self.manager.delete_database("knowledge")
            self.assertFalse((self.rag_dir / "knowledge").exists())

            # Deleting a missing database raises
            with self.assertRaises(RAGError):
                self.manager.delete_database("knowledge")

    async def test_add_file_and_search(self) -> None:
        doc = self.rag_dir / "doc.txt"
        doc.write_text("LangChain and LangGraph are framework libraries for building agents.", encoding="utf-8")

        mock_client = MagicMock()
        mock_hit = MagicMock()
        mock_hit.payload = {
            "content": "LangChain and LangGraph are framework libraries",
            "source": str(doc),
            "filename": "doc.txt",
            "chunk_index": 0,
        }
        mock_hit.score = 0.92
        mock_resp = MagicMock(points=[mock_hit])
        mock_client.query_points.return_value = mock_resp

        self.manager._client = mock_client
        self.manager._current_db = "test_db"

        with patch.object(RAGManager, "_get_embeddings", AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])):
            with patch.object(RAGManager, "_get_embedding", AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])):
                res = await self.manager.add_file(str(doc))
                self.assertEqual(res["chunks"], 1)
                mock_client.upsert.assert_called_once()

                hits = await self.manager.search("LangGraph")
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]["filename"], "doc.txt")
                self.assertEqual(hits[0]["score"], 0.92)

                # empty query returns [] immediately
                empty_hits = await self.manager.search("   ")
                self.assertEqual(empty_hits, [])

                # top_k <= 0 raises RAGError
                with self.assertRaises(RAGError):
                    await self.manager.search("LangGraph", top_k=0)

    async def test_add_file_rejects_unsupported_extension(self) -> None:
        doc = self.rag_dir / "data.xyz"
        doc.write_text("content", encoding="utf-8")
        self.manager._client = MagicMock()
        self.manager._current_db = "test_db"
        with self.assertRaises(RAGError):
            await self.manager.add_file(str(doc))

    async def test_add_file_keeps_points_when_embeddings_fail(self) -> None:
        doc = self.rag_dir / "doc.txt"
        doc.write_text("Some content for indexing.", encoding="utf-8")
        self.manager._client = MagicMock()
        self.manager._current_db = "test_db"

        with patch.object(RAGManager, "_delete_source_points") as mock_delete:
            with patch.object(RAGManager, "_get_embeddings", AsyncMock(side_effect=RAGError("Ollama down"))):
                with self.assertRaises(RAGError):
                    await self.manager.add_file(str(doc))
                mock_delete.assert_not_called()

    async def test_add_directory_and_errors(self) -> None:
        sub_dir = self.rag_dir / "docs_dir"
        sub_dir.mkdir()
        f1 = sub_dir / "file1.md"
        f1.write_text("# Doc 1\nSome useful content.", encoding="utf-8")
        f2 = sub_dir / "file2.py"
        f2.write_text("def hello(): return 'world'", encoding="utf-8")
        f_empty = sub_dir / "empty.txt"
        f_empty.write_text("   \n", encoding="utf-8")

        mock_client = MagicMock()
        self.manager._client = mock_client
        self.manager._current_db = "dir_db"

        with patch.object(
            RAGManager, "_get_embeddings", AsyncMock(side_effect=lambda chunks: [[0.1, 0.2, 0.3, 0.4]] * len(chunks))
        ):
            res = await self.manager.add_directory(str(sub_dir))
            self.assertEqual(res["added"], 2)
            self.assertEqual(res["failed"], 0)

        # Nonexistent directory raises RAGError
        with self.assertRaises(RAGError):
            await self.manager.add_directory(str(self.rag_dir / "nonexistent_dir"))

        # File passed as directory raises RAGError
        with self.assertRaises(RAGError):
            await self.manager.add_directory(str(f1))

        # Batch failure during add_directory: embeddings raise immediately (Fail Fast)
        with patch.object(RAGManager, "_get_embeddings", AsyncMock(side_effect=RAGError("Ollama connection failed"))):
            with self.assertRaises(RAGError):
                await self.manager.add_directory(str(sub_dir))

        # Unloaded database error
        self.manager.unload()
        with self.assertRaises(RAGNotLoadedError):
            await self.manager.add_file(str(f1))

    def test_rag_context_resolve_database(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_database_names.return_value = ["docs", "docs_v2", "code"]
        ctx = RAGContext(rag_manager=mock_mgr, console=Console(file=io.StringIO(), record=True))

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

    def test_rag_command_handlers(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        ctx = RAGContext(rag_manager=self.manager, console=console)

        # Empty databases
        list_rag_databases(ctx)
        self.assertIn("No RAG databases found", console.export_text())

        # Status not loaded
        show_rag_status(ctx)
        self.assertIn("No RAG database is currently loaded", console.export_text())

        # Create, Load, Show, Unload, Delete
        with patch("ollama_agent.rag.manager.QdrantClient"):
            create_rag_database(ctx, "manual_db")
            load_rag_database(ctx, "manual_db")
            self.assertEqual(self.manager.current_database, "manual_db")

            show_rag_status(ctx)
            self.assertIn("Active RAG database", console.export_text())

            unload_rag_database(ctx)
            self.assertIsNone(self.manager.current_database)

            delete_rag_database(ctx, "manual_db")
            self.assertFalse((self.rag_dir / "manual_db").exists())

    def test_read_file_custom_allowed_extensions(self) -> None:
        f_custom = self.rag_dir / "test.custom"
        f_custom.write_text("custom content", encoding="utf-8")
        # Fails with default extensions
        with self.assertRaises(RAGError):
            self.manager._read_file(f_custom)
        # Succeeds when allowed_extensions includes .custom
        content = self.manager._read_file(f_custom, allowed_extensions=frozenset({".custom"}))
        self.assertEqual(content, "custom content")

    def test_delete_source_points(self) -> None:
        mock_client = MagicMock()
        self.manager._delete_source_points(mock_client, "/path/to/file.py")
        mock_client.delete.assert_called_once()

    def test_create_database_failure_cleanup(self) -> None:
        with patch("ollama_agent.rag.manager.QdrantClient") as mock_qdrant_cls:
            mock_client = MagicMock()
            mock_client.create_collection.side_effect = RuntimeError("Disk full")
            mock_qdrant_cls.return_value = mock_client

            with self.assertRaises(RAGError):
                self.manager.create_database("failing_db")

            self.assertFalse((self.rag_dir / "failing_db").exists())
            mock_client.close.assert_called()

    async def test_get_embeddings_batching(self) -> None:
        texts = [f"Text chunk {i}" for i in range(250)]
        mock_response = MagicMock(embeddings=[[0.1, 0.2, 0.3, 0.4]] * 100)
        mock_response_rem = MagicMock(embeddings=[[0.1, 0.2, 0.3, 0.4]] * 50)
        self.manager._ollama_client = AsyncMock()
        self.manager._ollama_client.embed.side_effect = [
            mock_response,
            mock_response,
            mock_response_rem,
        ]

        embeddings = await self.manager._get_embeddings(texts, batch_size=100)
        self.assertEqual(len(embeddings), 250)
        self.assertEqual(self.manager._ollama_client.embed.call_count, 3)

    async def test_add_directory_partial_failure(self) -> None:
        sub_dir = self.rag_dir / "partial_dir"
        sub_dir.mkdir()
        f1 = sub_dir / "file1.md"
        f1.write_text("# Doc 1", encoding="utf-8")
        f2 = sub_dir / "file2.md"
        f2.write_text("# Doc 2", encoding="utf-8")

        mock_client = MagicMock()
        self.manager._client = mock_client
        self.manager._current_db = "partial_db"

        # file1 fails embedding: fail fast and propagate exception immediately
        async def mock_embed(texts: list[str], batch_size: int = 100) -> list[list[float]]:
            if "Doc 1" in texts[0]:
                raise RAGError("Embed failed for doc 1")
            return [[0.1, 0.2, 0.3, 0.4]] * len(texts)

        with patch.object(RAGManager, "_get_embeddings", side_effect=mock_embed):
            with self.assertRaises(RAGError):
                await self.manager.add_directory(str(sub_dir))

    def test_database_resolution_case_insensitive_exact_priority(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_database_names.return_value = ["Alpha", "alphabet"]
        ctx = RAGContext(rag_manager=mock_mgr, console=Console(file=io.StringIO(), record=True))
        # "alpha" matches "Alpha" exactly (case-insensitive) instead of matching "alphabet" as prefix
        self.assertEqual(ctx.resolve_database("alpha"), "Alpha")

    def test_database_resolution_empty_name_raises(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_databases.return_value = [{"name": "docs", "active": False, "chunks": 5}]
        ctx = RAGContext(rag_manager=mock_mgr, console=Console(file=io.StringIO(), record=True))
        with self.assertRaises(RAGDatabaseNotFoundError) as exc:
            ctx.resolve_database("")
        self.assertIn("cannot be empty", str(exc.exception))
        with self.assertRaises(RAGDatabaseNotFoundError) as exc:
            ctx.resolve_database("   ")
        self.assertIn("cannot be empty", str(exc.exception))

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
