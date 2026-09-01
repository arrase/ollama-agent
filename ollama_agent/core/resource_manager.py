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
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar

from ..i18n import _

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


class BaseFileStoreManager(ABC, Generic[T]):
    """Common base for managers that persist resources as files or directories."""

    #: File extension used when building item paths. Set to ``""`` for
    #: directory-based resources (e.g. skills) or ``".yaml"`` for files.
    _ext: str = ""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _path(self, item_id: str) -> Path:
        """Return the filesystem path for *item_id*."""
        resolved = (self.base_dir / f"{item_id}{self._ext}").resolve()
        if not resolved.is_relative_to(self.base_dir) or resolved == self.base_dir:
            raise ValueError(_("Path traversal detected: {item_id}", item_id=item_id))
        return resolved

    # ------------------------------------------------------------------
    # Abstract interface – subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def get(self, item_id: str) -> T:
        """Retrieve a resource by *item_id*. Raise FileNotFoundError if missing."""

    @abstractmethod
    def find_matches(self, prefix: str) -> list[tuple[str, T]]:
        """Return items whose id starts with *prefix*."""

    @abstractmethod
    def list_all(self) -> list[tuple[str, T]]:
        """Return all items sorted."""

    @abstractmethod
    def delete(self, item_id: str) -> None:
        """Delete item by id. Raise FileNotFoundError if it doesn't exist."""


def require_text(value: str, name: str, error: type[E]) -> str:
    """Return *value* stripped, raising *error* if it is empty."""
    if not (cleaned := value.strip()):
        raise error(_("{name} cannot be empty.", name=name))
    return cleaned


def resolve_unique_match(
    matches: Sequence[tuple[str, T]],
    prefix: str,
    *,
    label: str,
    not_found_error: type[E],
    ambiguous_error: type[E],
) -> tuple[str, T]:
    """Return the only match for *prefix* or raise the matching domain error."""
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise not_found_error(_("{label} not found: {prefix}", label=label, prefix=prefix))
    raise ambiguous_error(
        _("Ambiguous prefix: {name} -> {matches}", name=prefix, matches=", ".join(m[0] for m in matches))
    )


__all__ = [
    "BaseFileStoreManager",
    "require_text",
    "resolve_unique_match",
]
