"""Abstract base for file-backed resource managers (skills, tasks, etc.).

Provides the shared boilerplate that :class:`SkillManager` and
:class:`TaskManager` (and any future manager) would otherwise duplicate:

* ``__init__`` that creates the base directory
* ``_path`` that builds an item path using a configurable file extension

Subclasses must supply:

* ``_ext`` – file extension including the dot (``".yaml"``) or ``""`` for
  directory-based items.
* Abstract ``find_matches``, ``list_all`` and ``delete`` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseFileStoreManager(ABC, Generic[T]):
    """Common base for managers that persist resources as files or directories."""

    #: File extension used when building item paths. Set to ``""`` for
    #: directory-based resources (e.g. skills) or ``".yaml"`` for files.
    _ext: str = ""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _path(self, item_id: str) -> Path:
        """Return the filesystem path for *item_id*."""
        resolved = (self.base_dir / f"{item_id}{self._ext}").resolve()
        if not resolved.is_relative_to(self.base_dir.resolve()):
            raise ValueError(f"Path traversal detected: {item_id}")
        return resolved

    # ------------------------------------------------------------------
    # Abstract interface – subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def get(self, item_id: str) -> T | None:
        """Retrieve a resource by *item_id*, or return None if it doesn't exist."""

    @abstractmethod
    def find_matches(self, prefix: str) -> list[tuple[str, T]]:
        """Return items whose id starts with *prefix*."""

    @abstractmethod
    def list_all(self) -> list[tuple[str, T]]:
        """Return all items sorted."""

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Delete item by id. Returns ``True`` on success."""
