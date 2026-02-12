"""Shared RAG management commands used by CLI and REPL."""

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from .manager import RAGManager, RAGError, RAGDatabaseExistsError, RAGNotLoadedError
from ..core import resolve_unique_prefix


@dataclass
class RAGContext:
    """Holds shared resources for RAG commands."""
    console: Console = field(default_factory=Console)
    rag_manager: RAGManager = field(default_factory=RAGManager)

    def _find_or_exit(self, name: str) -> str:
        """Find a database by name/prefix or exit."""
        dbs = self.rag_manager.list_databases()
        names = [d.get("name", "") for d in dbs if isinstance(d, dict)]
        resolved = resolve_unique_prefix(name, names)
        if resolved:
            return resolved

        matches = [n for n in names if n.startswith((name or "").strip())]
        msg = f"Database not found: {name}" if not matches else f"Ambiguous prefix: {name} -> {', '.join(matches)}"
        self.console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)


def list_rag_databases(ctx: RAGContext) -> None:
    """List all RAG databases."""
    dbs = ctx.rag_manager.list_databases()
    if not dbs:
        ctx.console.print("[yellow]No RAG databases found.[/yellow]")
        ctx.console.print("[dim]Create one with /rag-create <name>[/dim]")
        return

    table = Table(title="RAG Databases", show_header=True, header_style="bold magenta")
    for col, style in [("Name", "cyan"), ("Chunks", "green"), ("Status", "yellow")]:
        table.add_column(col, style=style)

    for db in dbs:
        status = "[green]◀ active[/green]" if db["active"] else ""
        table.add_row(db["name"], str(db["chunks"]), status)

    ctx.console.print(table)


def create_rag_database(ctx: RAGContext, name: str) -> None:
    """Create a new RAG database."""
    try:
        created = ctx.rag_manager.create_database(name)
        ctx.console.print(f"[green]RAG database created:[/green] [cyan]{created}[/cyan]")
        ctx.console.print(f"[dim]Load it with /rag-load {created}[/dim]")
    except RAGDatabaseExistsError:
        ctx.console.print(f"[red]Database already exists:[/red] {name}")
        raise SystemExit(1)
    except RAGError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise SystemExit(1)


def delete_rag_database(ctx: RAGContext, name: str) -> None:
    """Delete a RAG database."""
    full_name = ctx._find_or_exit(name)
    if ctx.rag_manager.delete_database(full_name):
        ctx.console.print(f"[green]Deleted RAG database:[/green] [cyan]{full_name}[/cyan]")
    else:
        ctx.console.print(f"[red]Failed to delete database: {full_name}[/red]")
        raise SystemExit(1)


def load_rag_database(ctx: RAGContext, name: str) -> None:
    """Load a RAG database."""
    full_name = ctx._find_or_exit(name)
    try:
        ctx.rag_manager.load_database(full_name)
        ctx.console.print(f"[green]Loaded RAG database:[/green] [cyan]{full_name}[/cyan]")
    except RAGError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise SystemExit(1)


def unload_rag_database(ctx: RAGContext) -> None:
    """Unload the current RAG database."""
    if ctx.rag_manager.current_database is None:
        ctx.console.print("[yellow]No RAG database is currently loaded.[/yellow]")
        return
    name = ctx.rag_manager.current_database
    ctx.rag_manager.unload()
    ctx.console.print(f"[green]Unloaded RAG database:[/green] [cyan]{name}[/cyan]")


def add_rag_file(ctx: RAGContext, file_path: str) -> None:
    """Add a file to the current RAG database."""
    try:
        result = ctx.rag_manager.add_file(file_path)
        ctx.console.print(
            f"[green]Added to RAG:[/green] [cyan]{result['file']}[/cyan] "
            f"([dim]{result['chunks']} chunks[/dim])"
        )
    except RAGNotLoadedError:
        ctx.console.print("[red]No RAG database loaded.[/red] Use /rag-load <name> first.")
        raise SystemExit(1)
    except RAGError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise SystemExit(1)


def add_rag_directory(ctx: RAGContext, dir_path: str) -> None:
    """Add all files from a directory to the current RAG database."""
    try:
        result = ctx.rag_manager.add_directory(dir_path)
        ctx.console.print(
            f"[green]Added {result['added']} files[/green] "
            f"([dim]skipped: {result['skipped']}, failed: {result['failed']}[/dim])"
        )
    except RAGNotLoadedError:
        ctx.console.print("[red]No RAG database loaded.[/red] Use /rag-load <name> first.")
        raise SystemExit(1)
    except RAGError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise SystemExit(1)


def show_rag_status(ctx: RAGContext) -> None:
    """Show current RAG status."""
    current = ctx.rag_manager.current_database
    if current:
        ctx.console.print(f"[bold]Active RAG database:[/bold] [cyan]{current}[/cyan]")
    else:
        ctx.console.print("[yellow]No RAG database is currently loaded.[/yellow]")
        ctx.console.print("[dim]Use /rag-list to see available databases[/dim]")
