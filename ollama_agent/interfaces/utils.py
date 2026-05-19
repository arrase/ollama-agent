"""Interface-layer utilities.

.. deprecated::
    As of the latest refactor, no modules outside ``interfaces/`` import from
    this file.  The helper functions ``find_or_exit`` and ``require_or_exit``
    have been inlined into their respective domain command modules
    (``rag/commands``, ``skills/commands``, ``tasks/commands``) to eliminate
    cross-layer (domain → presentation) dependencies.  This module is kept for
    backward compatibility but may be removed in a future release.

Functions here are allowed to depend on presentation libraries such as
``rich``, and may raise :exc:`SystemExit`. They are intended for command-layer
code, including the CLI/REPL adapters and the command helpers in
``tasks/``, ``rag/`` and ``skills/``. Pure core/runtime modules should keep
avoiding this layer to prevent presentation concerns from leaking inward.
"""

from __future__ import annotations

from rich.console import Console

from ..core import resolve_unique_prefix


def find_or_exit(
    prefix: str,
    candidates: list[str],
    console: Console,
    *,
    label: str = "item",
) -> str:
    """Resolve a unique prefix match from *candidates* or exit with an error.

    * If there is exactly one match, its full value is returned.
    * If there is no match, or more than one, an error is printed to *console*
      and :exc:`SystemExit(1)` is raised.
    """
    if resolved := resolve_unique_prefix(prefix, candidates):
        return resolved
    matches = [c for c in candidates if c.startswith((prefix or "").strip())]
    msg = (
        f"{label} not found: {prefix}"
        if not matches
        else f"Ambiguous prefix: {prefix} -> {', '.join(matches)}"
    )
    console.print(f"[red]{msg}[/red]")
    raise SystemExit(1)


def require_or_exit(value: str, name: str, console: Console) -> str:
    """Return a non-empty, stripped value or exit with an error."""
    if not (cleaned := (value or "").strip().strip("\n")):
        console.print(f"[red]{name} cannot be empty.[/red]")
        raise SystemExit(1)
    return cleaned
