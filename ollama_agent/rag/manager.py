"""RAG manager for document storage and retrieval using Qdrant."""

from __future__ import annotations

import re
import hashlib
import logging
import mimetypes
import shutil
from dataclasses import dataclass, field
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

from .settings import RAGSettings

logger = logging.getLogger(__name__)


class RAGError(RuntimeError):
    """Base exception for RAG operations."""


class RAGNotLoadedError(RAGError):
    """Raised when no RAG database is loaded."""


class RAGDatabaseExistsError(RAGError):
    """Raised when attempting to create a database that already exists."""


@dataclass
class RAGDocument:
    """Represents a document chunk in the RAG database."""

    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGManager:
    """Manages RAG databases with Qdrant vector storage."""

    __slots__ = ("settings", "_client", "_current_db", "_rag_dir")

    COLLECTION_NAME = "documents"

    def __init__(self, settings: RAGSettings | None = None) -> None:
        self.settings = settings or RAGSettings()
        self._rag_dir = Path(self.settings.rag_dir)
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
            raise RAGNotLoadedError("No RAG database loaded. Use /rag-load <name> first.")
        return self._client

    def list_databases(self) -> list[dict[str, Any]]:
        """List all available RAG databases."""
        dbs = []
        for path in self._rag_dir.iterdir():
            if not path.is_dir():
                continue
            try:
                client = QdrantClient(path=str(path))
                info = client.get_collection(self.COLLECTION_NAME)
                count = info.points_count
                client.close()
            except Exception:
                # Not a valid RAG database (or unreadable)
                continue
            dbs.append({
                "name": path.name,
                "path": str(path),
                "chunks": count,
                "active": path.name == self._current_db,
            })
        return sorted(dbs, key=lambda x: x["name"])

    def create_database(self, name: str) -> str:
        """Create a new RAG database."""
        name = self._validate_name(name)
        db_path = self._db_path(name)

        if db_path.exists():
            raise RAGDatabaseExistsError(f"Database '{name}' already exists")

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

    def delete_database(self, name: str) -> bool:
        """Delete a RAG database."""
        name = self._validate_name(name)
        db_path = self._db_path(name)

        if not db_path.exists():
            return False

        # Unload if currently active
        if self._current_db == name:
            self.unload()

        # Remove directory
        shutil.rmtree(db_path)
        logger.info("Deleted RAG database: %s", name)
        return True

    def load_database(self, name: str) -> str:
        """Load a RAG database for use."""
        name = self._validate_name(name)
        db_path = self._db_path(name)

        if not db_path.exists():
            raise RAGError(f"Database '{name}' not found")

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

    def add_file(self, file_path: str) -> dict[str, Any]:
        """Add a file to the current RAG database."""
        client = self._ensure_loaded()
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise RAGError(f"File not found: {file_path}")

        if not path.is_file():
            raise RAGError(f"Not a file: {file_path}")

        # Read file content
        content = self._read_file(path)
        if not content.strip():
            raise RAGError(f"File is empty: {file_path}")

        # Remove any previously indexed chunks for this file to avoid stale points
        # when chunking changes (e.g., file edits, config changes).
        self._delete_source_points(client, str(path))

        # Chunk the content
        chunks = self._chunk_text(content)

        # Generate embeddings in batch and store
        embeddings = self._get_embeddings(chunks)
        points: list[PointStruct] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
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
            "database": self._current_db,
        }

    def add_directory(self, dir_path: str, extensions: list[str] | None = None) -> dict[str, Any]:
        """Add all files from a directory to the current RAG database."""
        path = Path(dir_path).expanduser().resolve()

        if not path.exists():
            raise RAGError(f"Directory not found: {dir_path}")

        if not path.is_dir():
            raise RAGError(f"Not a directory: {dir_path}")

        # Default extensions for text files
        if extensions is None:
            extensions = [".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
                          ".html", ".css", ".xml", ".csv", ".rst", ".ini", ".cfg", ".sh"]

        results = {"added": 0, "failed": 0, "skipped": 0, "files": []}

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue

            if extensions and file_path.suffix.lower() not in extensions:
                results["skipped"] += 1
                continue

            try:
                result = self.add_file(str(file_path))
                results["added"] += 1
                results["files"].append(result)
            except RAGError as e:
                logger.warning("Failed to add %s: %s", file_path, e)
                results["failed"] += 1

        return results

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Search the RAG database for relevant documents."""
        client = self._ensure_loaded()
        top_k = top_k or self.settings.default_top_k

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Prefer stable API across qdrant-client versions
        response = client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        results = response.points

        return [
            {
                "content": hit.payload.get("content", "") if hit.payload else "",
                "source": hit.payload.get("source", "") if hit.payload else "",
                "filename": hit.payload.get("filename", "") if hit.payload else "",
                "score": hit.score,
                "chunk_index": hit.payload.get("chunk_index", 0) if hit.payload else 0,
            }
            for hit in results
        ]

    def get_context(self, query: str, top_k: int | None = None) -> str:
        """Get formatted context from RAG for use in prompts."""
        results = self.search(query, top_k)
        if not results:
            return ""

        context_parts = []
        for r in results:
            source = r.get("filename", r.get("source", "unknown"))
            context_parts.append(f"[Source: {source}]\n{r['content']}")

        return "\n\n---\n\n".join(context_parts)

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using Ollama."""
        embeddings = self._get_embeddings([text])
        return embeddings[0] if embeddings else []

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts using Ollama."""
        if not texts:
            return []
        try:
            client = ollama.Client(host=self.settings.embedder_base_url)
            response = client.embed(
                model=self.settings.embedder_model,
                input=texts,
            )
            embeddings: list[list[float]] = response.get("embeddings", []) or []

            if len(embeddings) != len(texts):
                raise RAGError(
                    f"Embedding generation returned {len(embeddings)} vectors for {len(texts)} inputs"
                )

            expected = self.settings.embedding_dims
            for idx, vec in enumerate(embeddings):
                if expected and len(vec) != expected:
                    raise RAGError(
                        f"Embedding dims mismatch at index {idx}: expected {expected}, got {len(vec)}"
                    )

            return embeddings
        except RAGError:
            raise
        except Exception as e:
            raise RAGError(f"Embedding generation failed: {e}") from e

    def _delete_source_points(self, client: QdrantClient, source: str) -> None:
        """Delete all points previously indexed for a given source path."""
        try:
            client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source),
                        )
                    ]
                ),
                wait=True,
            )
        except TypeError:
            # Older qdrant-client versions may not support wait= or this signature.
            client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source),
                        )
                    ]
                ),
            )
        except Exception as e:
            raise RAGError(f"Failed to delete existing points for source '{source}': {e}") from e

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for natural break points
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size // 2:
                        chunk = chunk[:last_sep + len(sep)]
                        end = start + len(chunk)
                        break

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]

    def _read_file(self, path: Path) -> str:
        """Read file content, handling different encodings."""
        # Check if it's a text file
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and not mime_type.startswith(("text/", "application/json", "application/xml")):
            raise RAGError(f"Unsupported file type: {mime_type}")

        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise RAGError(f"Could not decode file: {path}")

    @staticmethod
    def _validate_name(name: str) -> str:
        """Validate database name."""
        name = (name or "").strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise RAGError("Invalid name. Use only letters, numbers, '_' and '-'.")
        return name

    @staticmethod
    def _generate_point_id(source: str, chunk_index: int) -> int:
        """Generate a unique point ID from source and chunk index."""
        combined = f"{source}:{chunk_index}"
        return int(hashlib.md5(combined.encode()).hexdigest()[:15], 16)
