"""Interface-layer utilities.

Functions here are allowed to depend on presentation libraries such as
``rich``, and may raise :exc:`SystemExit`.  They must **not** be imported by
pure-core modules (``core/``, ``rag/``, ``skills/``, ``tasks/``) to avoid
circular dependencies — those modules own their own error-display logic via
their context objects.
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
