"""RAG manager for document storage and retrieval using Qdrant."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ..core.common import validate_identifier
from ..i18n import _
from ..settings import RAGSettings

logger = logging.getLogger(__name__)


def _validate_name(name: str) -> str:
    """Validate database name."""
    try:
        return validate_identifier(name, "name")
    except ValueError as e:
        raise RAGError(str(e)) from e


def _generate_point_id(source: str, chunk_index: int) -> str:
    """Generate a unique point ID as a UUID string from source and chunk index."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_index}"))


SUPPORTED_RAG_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".sh",
        ".yaml",
        ".yml",
        ".json",
        ".xml",
        ".md",
        ".txt",
        ".toml",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".css",
        ".html",
        ".sql",
        ".ini",
        ".cfg",
        ".properties",
        ".java",
        ".kt",
        ".gradle",
        ".bat",
        ".ps1",
        ".csv",
        ".rst",
    }
)


class RAGError(RuntimeError):
    """Base exception for RAG operations."""


class RAGNotLoadedError(RAGError):
    """Raised when no RAG database is loaded."""


class RAGDatabaseExistsError(RAGError):
    """Raised when attempting to create a database that already exists."""


class RAGManager:
    """Manages RAG databases with Qdrant vector storage."""

    __slots__ = ("settings", "_client", "_current_db", "_rag_dir", "_ollama_client")

    COLLECTION_NAME = "documents"

    def __init__(self, settings: RAGSettings) -> None:
        self.settings = settings
        self._rag_dir = Path(self.settings.rag_dir).expanduser().resolve()
        self._rag_dir.mkdir(parents=True, exist_ok=True)
        self._client: QdrantClient | None = None
        self._current_db: str | None = None
        self._ollama_client = ollama.AsyncClient(host=self.settings.embedder_base_url.rstrip("/"))

    @property
    def current_database(self) -> str | None:
        """Return the name of the currently loaded database."""
        return self._current_db

    def _db_path(self, name: str) -> Path:
        """Get the path for a database directory."""
        return self._rag_dir / name

    def _ensure_loaded(self) -> QdrantClient:
        """Ensure a database is loaded and return the client."""
        if self._client is None:
            raise RAGNotLoadedError(_("No RAG database loaded. Use /rag load <name> first."))
        return self._client

    def list_database_names(self) -> list[str]:
        """List the names of all available RAG database directories."""
        return sorted(path.name for path in self._rag_dir.iterdir() if path.is_dir())

    def list_databases(self) -> list[dict[str, Any]]:
        """List all available RAG databases."""
        dbs = []
        for path in sorted(self._rag_dir.iterdir()):
            if not path.is_dir():
                continue
            is_active = path.name == self._current_db
            chunks = None
            if is_active:
                try:
                    info = self._client.get_collection(self.COLLECTION_NAME)
                    chunks = info.points_count
                except Exception as e:
                    raise RAGError(_("Failed to get collection info: {e}", e=e)) from e
            dbs.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "chunks": chunks,
                    "active": is_active,
                }
            )
        return dbs

    def create_database(self, name: str) -> str:
        """Create a new RAG database."""
        name = _validate_name(name)
        db_path = self._db_path(name)

        if db_path.exists():
            raise RAGDatabaseExistsError(_("Database '{name}' already exists", name=name))

        db_path.mkdir(parents=True, exist_ok=True)

        client = QdrantClient(path=str(db_path))
        try:
            client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dims,
                    distance=Distance.COSINE,
                ),
            )
        except Exception as e:
            client.close()
            shutil.rmtree(db_path)
            raise RAGError(_("Failed to create database '{name}': {e}", name=name, e=e)) from e
        else:
            client.close()

        logger.info("Created RAG database: %s", name)
        return name

    def delete_database(self, name: str) -> None:
        """Delete a RAG database."""
        name = _validate_name(name)
        db_path = self._db_path(name)

        if not db_path.exists():
            raise RAGError(_("Database '{name}' not found", name=name))

        # Unload if currently active
        if self._current_db == name:
            self.unload()

        # Remove directory
        shutil.rmtree(db_path)
        logger.info("Deleted RAG database: %s", name)

    def load_database(self, name: str) -> str:
        """Load a RAG database for use."""
        name = _validate_name(name)
        db_path = self._db_path(name)

        if not db_path.exists():
            raise RAGError(_("Database '{name}' not found", name=name))

        self.unload()

        try:
            self._client = QdrantClient(path=str(db_path))
        except Exception as e:
            raise RAGError(_("Failed to load database '{name}': {e}", name=name, e=e)) from e
        self._current_db = name
        logger.info("Loaded RAG database: %s", name)
        return name

    def unload(self) -> None:
        """Unload the current database."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._current_db = None

    async def _index_chunks(self, client: QdrantClient, path: Path, chunks: list[str]) -> None:
        """Generate embeddings and upsert points for chunks of a file."""
        embeddings = await self._get_embeddings(chunks)
        source = str(path)
        self._delete_source_points(client, source)
        points = [
            PointStruct(
                id=_generate_point_id(source, i),
                vector=embedding,
                payload={
                    "content": chunk,
                    "source": source,
                    "filename": path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        try:
            client.upsert(collection_name=self.COLLECTION_NAME, points=points)
        except Exception as e:
            raise RAGError(_("Failed to upsert points into vector database: {e}", e=e)) from e

    async def add_file(self, file_path: str) -> dict[str, Any]:
        """Add a file to the current RAG database."""
        client = self._ensure_loaded()
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise RAGError(_("File not found: {file_path}", file_path=file_path))

        if not path.is_file():
            raise RAGError(_("Not a file: {file_path}", file_path=file_path))

        # Read file content
        content = self._read_file(path)
        if not content.strip():
            raise RAGError(_("File is empty: {file_path}", file_path=file_path))

        # Chunk the content
        chunks = self._chunk_text(content)

        await self._index_chunks(client, path, chunks)

        logger.info("Added file to RAG: %s (%d chunks)", path.name, len(chunks))
        return {
            "file": str(path),
            "chunks": len(chunks),
        }

    async def add_directory(
        self,
        dir_path: str,
        extensions: frozenset[str] = SUPPORTED_RAG_EXTENSIONS,
    ) -> dict[str, Any]:
        """Add all files from a directory to the current RAG database."""
        client = self._ensure_loaded()
        path = Path(dir_path).expanduser().resolve()

        if not path.exists():
            raise RAGError(_("Directory not found: {dir_path}", dir_path=dir_path))

        if not path.is_dir():
            raise RAGError(_("Not a directory: {dir_path}", dir_path=dir_path))

        results: dict[str, Any] = {"added": 0, "failed": 0, "skipped": 0}

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in extensions:
                results["skipped"] += 1
                continue

            try:
                content = self._read_file(file_path, allowed_extensions=extensions)
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("Failed to read %s: %s", file_path, e)
                results["failed"] += 1
                continue

            if not content.strip():
                continue

            chunks = self._chunk_text(content)
            await self._index_chunks(client, file_path, chunks)
            results["added"] += 1
            logger.info("Added file to RAG from batch: %s (%d chunks)", file_path.name, len(chunks))

        return results

    async def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Search the RAG database for relevant documents."""
        if not query.strip():
            raise RAGError(_("Search query cannot be empty."))
        limit = self.settings.default_top_k if top_k is None else top_k
        if limit <= 0:
            raise RAGError(_("Limit must be greater than 0"))

        client = self._ensure_loaded()

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Prefer stable API across qdrant-client versions
        try:
            response = client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_embedding,
                limit=limit,
                with_payload=True,
            )
        except Exception as e:
            raise RAGError(_("Failed to query vector database: {e}", e=e)) from e

        return [
            {
                "content": hit.payload["content"],
                "source": hit.payload["source"],
                "filename": hit.payload["filename"],
                "score": hit.score,
                "chunk_index": hit.payload["chunk_index"],
            }
            for hit in response.points
        ]

    async def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using Ollama."""
        embeddings = await self._get_embeddings([text])
        return embeddings[0]

    async def _get_embeddings(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Generate embeddings for a batch of texts using Ollama."""
        all_embeddings: list[list[float]] = []
        expected_dim = self.settings.embedding_dims
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                response = await self._ollama_client.embed(
                    model=self.settings.embedder_model,
                    input=batch_texts,
                )
            except Exception as e:
                raise RAGError(_("Failed to generate embeddings: {e}", e=e)) from e
            embeddings: list[list[float]] = [list(vec) for vec in response.embeddings]

            if len(embeddings) != len(batch_texts):
                raise RAGError(
                    _(
                        "Embedding generation returned {actual} vectors for {expected} inputs",
                        actual=len(embeddings),
                        expected=len(batch_texts),
                    )
                )

            for idx, vec in enumerate(embeddings):
                if len(vec) != expected_dim:
                    raise RAGError(
                        _(
                            "Embedding dimension mismatch for text {idx}: got {actual}, expected {expected}",
                            idx=i + idx,
                            actual=len(vec),
                            expected=expected_dim,
                        )
                    )
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _delete_source_points(self, client: QdrantClient, source: str) -> None:
        """Delete all points previously indexed for a given source path."""
        filt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        try:
            client.delete(collection_name=self.COLLECTION_NAME, points_selector=filt, wait=True)
        except Exception as e:
            raise RAGError(_("Failed to delete existing points for source '{source}': {e}", source=source, e=e)) from e

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap."""
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap

        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise RAGError(
                _(
                    "Invalid chunk configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}",
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
            )

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If not at the end of text, try to find a good break point
            if end < len(text):
                # Look for paragraph break, then line break, then sentence, then word
                for sep in ["\n\n", "\n", ". ", " "]:
                    pos = text.rfind(sep, start, end)
                    if pos != -1 and pos > start + (chunk_size // 2):
                        end = pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            next_start = end - overlap
            start = next_start if next_start > start else end

        return chunks

    def _read_file(
        self,
        path: Path,
        allowed_extensions: frozenset[str] = SUPPORTED_RAG_EXTENSIONS,
    ) -> str:
        """Read file content as UTF-8."""
        if path.suffix.lower() not in allowed_extensions:
            raise RAGError(_("Unsupported file type: {file_path}", file_path=path))
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise RAGError(_("Could not decode file: {path}", path=path)) from e
