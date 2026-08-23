"""Shared RAG management commands used by CLI and REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from ..i18n import _
from .manager import RAGManager, RAGError, RAGDatabaseExistsError, RAGNotLoadedError


class RAGDatabaseNotFoundError(RAGError):
    """Raised when a RAG database cannot be resolved by name/prefix."""


class AmbiguousRAGDatabaseError(RAGError):
    """Raised when a database prefix matches multiple databases."""


@dataclass
class RAGContext:
    """Holds shared resources for RAG commands."""

    rag_manager: RAGManager
    console: Console = field(default_factory=Console)

    def _find_or_exit(self, name: str) -> str:
        """Find a database by name/prefix or raise RAGError."""
        target = name.strip()
        names = [d["name"] for d in self.rag_manager.list_databases()]
        if target in names:
            return target
        matches = [c for c in names if c.startswith(target)]
        if not matches:
            matches = [c for c in names if c.lower().startswith(target.lower())]
        if len(matches) == 1:
            return matches[0]
        msg = (
            _("Database not found: {name}", name=name)
            if not matches
            else _("Ambiguous prefix: {name} -> {matches}", name=name, matches=", ".join(matches))
        )
        if not matches:
            raise RAGDatabaseNotFoundError(msg)
        raise AmbiguousRAGDatabaseError(msg)


def list_rag_databases(ctx: RAGContext) -> None:
    """List all RAG databases."""
    dbs = ctx.rag_manager.list_databases()
    if not dbs:
        ctx.console.print(f"[yellow]{_('No RAG databases found.')}[/yellow]")
        ctx.console.print(f"[dim]{_('Create one with /rag create <name>')}[/dim]")
        return

    table = Table(title=_("RAG Databases"), show_header=True, header_style="bold magenta")
    for col, style in [(_("Name"), "cyan"), (_("Chunks"), "green"), (_("Status"), "yellow")]:
        table.add_column(col, style=style)

    for db in dbs:
        status = f"[green]◀ {_('active')}[/green]" if db["active"] else ""
        table.add_row(db["name"], str(db["chunks"]) if db["chunks"] is not None else "-", status)

    ctx.console.print(table)


def create_rag_database(ctx: RAGContext, name: str) -> None:
    """Create a new RAG database."""
    try:
        created = ctx.rag_manager.create_database(name)
        ctx.console.print(
            f"[green]✓ {_('RAG database created: {created}', created=created)}[/green]"
        )
        ctx.console.print(f"[dim]{_('Load it with /rag load {created}', created=created)}[/dim]")
    except RAGDatabaseExistsError:
        ctx.console.print(f"[red]{_('Database already exists: {name}', name=name)}[/red]")
        raise
    except RAGError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise


def delete_rag_database(ctx: RAGContext, name: str) -> None:
    """Delete a RAG database."""
    full_name = ctx._find_or_exit(name)
    if ctx.rag_manager.delete_database(full_name):
        ctx.console.print(
            f"[green]✓ {_('Deleted RAG database: {full_name}', full_name=full_name)}[/green]"
        )
    else:
        ctx.console.print(f"[red]{_('Failed to delete database: {full_name}', full_name=full_name)}[/red]")
        raise RAGError(_("Failed to delete database: {full_name}", full_name=full_name))


def load_rag_database(ctx: RAGContext, name: str) -> None:
    """Load a RAG database."""
    full_name = ctx._find_or_exit(name)
    try:
        ctx.rag_manager.load_database(full_name)
        ctx.console.print(
            f"[green]✓ {_('Loaded RAG database: {full_name}', full_name=full_name)}[/green]"
        )
    except RAGError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise


def unload_rag_database(ctx: RAGContext) -> None:
    """Unload the current RAG database."""
    if ctx.rag_manager.current_database is None:
        ctx.console.print(f"[yellow]{_('No RAG database is currently loaded.')}[/yellow]")
        return
    name = ctx.rag_manager.current_database
    ctx.rag_manager.unload()
    ctx.console.print(f"[green]✓ {_('Unloaded RAG database: {name}', name=name)}[/green]")


async def add_rag_file(ctx: RAGContext, file_path: str) -> None:
    """Add a file to the current RAG database."""
    try:
        result = await ctx.rag_manager.add_file(file_path)
        ctx.console.print(
            f"[green]✓ {_('Added to RAG: {file} ({chunks} chunks)', file=result['file'], chunks=result['chunks'])}[/green]"
        )
    except RAGNotLoadedError:
        ctx.console.print(
            f"[red]{_('No RAG database loaded. Use /rag load <name> first.')}[/red]"
        )
        raise
    except RAGError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise


async def add_rag_directory(ctx: RAGContext, dir_path: str) -> None:
    """Add all files from a directory to the current RAG database."""
    try:
        result = await ctx.rag_manager.add_directory(dir_path)
        ctx.console.print(
            f"[green]✓ {_('Added {added} files (skipped: {skipped}, failed: {failed})', added=result['added'], skipped=result['skipped'], failed=result['failed'])}[/green]"
        )
    except RAGNotLoadedError:
        ctx.console.print(
            f"[red]{_('No RAG database loaded. Use /rag load <name> first.')}[/red]"
        )
        raise
    except RAGError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise


def show_rag_status(ctx: RAGContext) -> None:
    """Show current RAG status."""
    current = ctx.rag_manager.current_database
    if current:
        ctx.console.print(f"[bold]{_('Active RAG database: {current}', current=current)}[/bold]")
    else:
        ctx.console.print(f"[yellow]{_('No RAG database is currently loaded.')}[/yellow]")
        ctx.console.print(f"[dim]{_('Use /rag list to see available databases')}[/dim]")
