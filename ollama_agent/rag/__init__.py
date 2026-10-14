"""RAG (Retrieval-Augmented Generation) package for document indexing and retrieval."""

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
    RAGError,
    RAGManager,
    RAGNotLoadedError,
)

__all__ = [
    # Manager
    "RAGError",
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

