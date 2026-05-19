"""RAG settings — canonical definition lives in settings.config."""

from ..settings.config import RAGSettings
from ..settings.paths import RAG_DIR

DEFAULT_RAG_DIR = RAG_DIR

__all__ = ["DEFAULT_RAG_DIR", "RAGSettings"]
