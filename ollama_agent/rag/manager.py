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
    MatchAny,
    PointStruct,
    VectorParams,
)

from ..core.common import validate_identifier
from ..i18n import _
from ..settings import RAGSettings

logger = logging.getLogger(__name__)

SUPPORTED_RAG_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".yaml", ".yml",
    ".json", ".xml", ".md", ".txt", ".toml", ".c", ".cpp", ".h",
    ".hpp", ".go", ".rs", ".css", ".html", ".sql", ".ini", ".cfg",
    ".properties", ".java", ".kt", ".gradle", ".bat", ".ps1",
    ".csv", ".rst",
})


class RAGError(RuntimeError):
    """Base exception for RAG operations."""


class RAGNotLoadedError(RAGError):
    """Raised when no RAG database is loaded."""


class RAGDatabaseExistsError(RAGError):
    """Raised when attempting to create a database that already exists."""


class RAGManager:
    """Manages RAG databases with Qdrant vector storage."""

    __slots__ = ("settings", "_client", "_current_db", "_rag_dir")

    COLLECTION_NAME = "documents"

    def __init__(self, settings: RAGSettings) -> None:
        self.settings = settings
        self._rag_dir = Path(self.settings.rag_dir).expanduser().resolve()
        self._rag_dir.mkdir(parents=True, exist_ok=True)
        self._client: QdrantClient | None = None
        self._current_db: str | None = None

    @property
    def current_database(self) -> str | None:
        """Return the name of the currently loaded database."""
        return self._current_db

    def _db_path(self, name: str) -> Path:
        """Get the path for a database directory."""
        return self._rag_dir / name

    def _ensure_loaded(self) -> QdrantClient:
        """Ensure a database is loaded and return the client."""
        if self._client is None or self._current_db is None:
            raise RAGNotLoadedError(
                _("No RAG database loaded. Use /rag load <name> first.")
            )
        return self._client

    def list_databases(self) -> list[dict[str, Any]]:
        """List all available RAG databases."""
        dbs = []
        for path in sorted(self._rag_dir.iterdir()):
            if not path.is_dir():
                continue
            is_active = path.name == self._current_db
            chunks = None
            if is_active and self._client is not None:
                info = self._client.get_collection(self.COLLECTION_NAME)
                chunks = info.points_count
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
        name = self._validate_name(name)
        db_path = self._db_path(name)

        if db_path.exists():
            raise RAGDatabaseExistsError(_("Database '{name}' already exists", name=name))

        db_path.mkdir(parents=True, exist_ok=True)

        # Initialize Qdrant with the collection
        client = QdrantClient(path=str(db_path))
        client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.settings.embedding_dims,
                distance=Distance.COSINE,
            ),
        )
        client.close()

        logger.info("Created RAG database: %s", name)
        return name

    def delete_database(self, name: str) -> None:
        """Delete a RAG database."""
        name = self._validate_name(name)
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
        name = self._validate_name(name)
        db_path = self._db_path(name)

        if not db_path.exists():
            raise RAGError(_("Database '{name}' not found", name=name))

        # Unload current database if any
        if self._client is not None:
            self._client.close()

        self._client = QdrantClient(path=str(db_path))
        self._current_db = name
        logger.info("Loaded RAG database: %s", name)
        return name

    def unload(self) -> None:
        """Unload the current database."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._current_db = None

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

        # Generate embeddings in batch and store. Old points are deleted only
        # after embeddings succeed, so a failure never loses indexed content.
        embeddings = await self._get_embeddings(chunks)
        self._delete_source_points(client, str(path))

        points: list[PointStruct] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            point_id = self._generate_point_id(str(path), i)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "content": chunk,
                        "source": str(path),
                        "filename": path.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
            )

        client.upsert(collection_name=self.COLLECTION_NAME, points=points)

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

        all_file_chunks = []
        valid_files = []

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue

            if extensions and file_path.suffix.lower() not in extensions:
                results["skipped"] += 1
                continue

            try:
                content = self._read_file(file_path)
                if not content.strip():
                    raise RAGError(_("File is empty: {file_path}", file_path=file_path))

                chunks = self._chunk_text(content)
                all_file_chunks.append((file_path, chunks))
                valid_files.append(str(file_path))
            except Exception as e:
                logger.warning("Failed to process %s: %s", file_path, e)
                results["failed"] += 1

        if not valid_files:
            return results

        # Prepare a flat list of chunk data
        flat_chunks_data = []
        for fpath, chunks in all_file_chunks:
            source = str(fpath)
            fname = fpath.name
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                flat_chunks_data.append((source, fname, chunk, i, total_chunks))

        # Process in batches. Old points are deleted only after the batch
        # embeddings succeed, so a failure never loses already-indexed content.
        BATCH_SIZE = 100
        failed_sources: set[str] = set()
        for i in range(0, len(flat_chunks_data), BATCH_SIZE):
            batch = flat_chunks_data[i : i + BATCH_SIZE]
            batch_texts = [b[2] for b in batch]
            batch_sources = {b[0] for b in batch}
            try:
                embeddings = await self._get_embeddings(batch_texts)
            except RAGError as e:
                logger.warning(
                    "Failed to generate embeddings for batch starting at index %d: %s",
                    i,
                    e,
                )
                failed_sources.update(batch_sources)
                continue

            self._delete_sources_points(client, batch_sources)

            points = []
            for (
                (source, fname, chunk, chunk_idx, total_chunks),
                embedding,
            ) in zip(batch, embeddings, strict=True):
                point_id = self._generate_point_id(source, chunk_idx)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "content": chunk,
                            "source": source,
                            "filename": fname,
                            "chunk_index": chunk_idx,
                            "total_chunks": total_chunks,
                        },
                    )
                )
            client.upsert(collection_name=self.COLLECTION_NAME, points=points)

        # Populate results for successful files
        for fpath, chunks in all_file_chunks:
            if str(fpath) in failed_sources:
                results["failed"] += 1
                continue

            results["added"] += 1
            logger.info(
                "Added file to RAG from batch: %s (%d chunks)", fpath.name, len(chunks)
            )

        return results

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Search the RAG database for relevant documents."""
        client = self._ensure_loaded()
        limit = self.settings.default_top_k if top_k is None else top_k

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Prefer stable API across qdrant-client versions
        response = client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        )

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

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts using Ollama."""
        try:
            client = ollama.AsyncClient(host=self.settings.embedder_base_url)
            response = await client.embed(
                model=self.settings.embedder_model,
                input=texts,
            )
            embeddings: list[list[float]] = [list(vec) for vec in response.embeddings]

            if len(embeddings) != len(texts):
                raise RAGError(
                    _("Embedding generation returned {actual} vectors for {expected} inputs", actual=len(embeddings), expected=len(texts))
                )

            expected = self.settings.embedding_dims
            for idx, vec in enumerate(embeddings):
                if len(vec) != expected:
                    raise RAGError(
                        _("Embedding dimension mismatch for text {idx}: got {actual}, expected {expected}", idx=idx, actual=len(vec), expected=expected)
                    )

            return embeddings
        except RAGError:
            raise
        except Exception as e:
            raise RAGError(_("Failed to generate embeddings: {e}", e=e)) from e

    def _delete_source_points(self, client: QdrantClient, source: str) -> None:
        """Delete all points previously indexed for a given source path."""
        self._delete_sources_points(client, {source})

    def _delete_sources_points(self, client: QdrantClient, sources: set[str]) -> None:
        """Delete all points previously indexed for the given source paths."""
        filt = Filter(
            must=[FieldCondition(key="source", match=MatchAny(any=sorted(sources)))]
        )
        try:
            client.delete(
                collection_name=self.COLLECTION_NAME, points_selector=filt, wait=True
            )
        except Exception as e:
            raise RAGError(
                _("Failed to delete existing points for source '{source}': {e}", source=sorted(sources)[0], e=e)
            ) from e

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap."""
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap

        if chunk_size <= 0 or overlap < 0:
            raise RAGError(
                _("Invalid chunk configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}", chunk_size=chunk_size, chunk_overlap=overlap)
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

    def _read_file(self, path: Path) -> str:
        """Read file content as UTF-8."""
        if path.suffix.lower() not in SUPPORTED_RAG_EXTENSIONS:
            raise RAGError(_("Unsupported file type: {file_path}", file_path=path))
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise RAGError(_("Could not decode file: {path}", path=path)) from e

    @staticmethod
    def _validate_name(name: str) -> str:
        """Validate database name."""
        try:
            return validate_identifier(name, "name")
        except ValueError as e:
            raise RAGError(str(e)) from e

    @staticmethod
    def _generate_point_id(source: str, chunk_index: int) -> str:
        """Generate a unique point ID as a UUID string from source and chunk index."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_index}"))
