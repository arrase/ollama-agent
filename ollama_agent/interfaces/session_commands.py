"""Session-related commands shared by CLI and REPL interfaces."""

from __future__ import annotations

from rich.console import Console


def new_session(console: Console, runtime_thread_id: str) -> str:
    """Start a new session by generating a new thread ID.

    Returns the new thread ID. The caller is responsible for assigning it
    to the runtime.
    """
    import uuid

    new_id = str(uuid.uuid4())[:8]
    console.print(
        f"[green]✓ New session started:[/green] [cyan]{new_id}[/cyan]"
    )
    return new_id
