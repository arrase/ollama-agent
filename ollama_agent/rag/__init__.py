"""RAG module for document storage and retrieval."""

from .commands import (
    RAGContext,
    add_rag_directory,
    add_rag_file,
    create_rag_database,
    delete_rag_database,
    list_rag_databases,
    load_rag_database,
    show_rag_status,
    unload_rag_database,
)
from .manager import (
    RAGDatabaseExistsError,
    RAGError,
    RAGManager,
    RAGNotLoadedError,
)
from .settings import DEFAULT_RAG_DIR, RAGSettings

__all__ = [
    # Settings
    "DEFAULT_RAG_DIR",
    "RAGSettings",
    # Manager
    "RAGError",
    "RAGDatabaseExistsError",
    "RAGManager",
    "RAGNotLoadedError",
    # Commands
    "RAGContext",
    "add_rag_directory",
    "add_rag_file",
    "create_rag_database",
    "delete_rag_database",
    "list_rag_databases",
    "load_rag_database",
    "show_rag_status",
    "unload_rag_database",
]
