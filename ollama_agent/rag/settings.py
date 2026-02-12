"""RAG settings configuration."""

from dataclasses import dataclass, field

from ..settings.paths import RAG_DIR


DEFAULT_RAG_DIR = RAG_DIR


@dataclass(frozen=True, slots=True)
class RAGSettings:
    """Configuration for RAG (Retrieval Augmented Generation) integration."""

    # Storage
    rag_dir: str = field(default_factory=lambda: str(DEFAULT_RAG_DIR))

    # Embedder (uses Ollama)
    embedder_model: str = "nomic-embed-text:latest"
    embedder_base_url: str = "http://localhost:11434"
    embedding_dims: int = 768

    # Search defaults
    default_top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50
