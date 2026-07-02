"""Shared skill management commands used by CLI and REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .manager import SkillInfo, SkillManager


@dataclass
class SkillsContext:
    """Holds shared resources for skill-related commands."""

    console: Console = field(default_factory=Console)
    skill_manager: SkillManager = field(default_factory=SkillManager)

    def _find_or_exit(self, skill_id: str) -> tuple[str, SkillInfo]:
        matches = self.skill_manager.find_matches(skill_id)
        if len(matches) == 1:
            return matches[0]
        msg = (
            f"Skill not found: {skill_id}"
            if not matches
            else f"Ambiguous prefix: {skill_id} -> {', '.join(t[0] for t in matches)}"
        )
        self.console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    def _require(self, value: str, name: str) -> str:
        if not (cleaned := value.strip().strip("\n")):
            self.console.print(f"[red]{name} cannot be empty.[/red]")
            raise SystemExit(1)
        return cleaned


def list_skills(ctx: SkillsContext) -> None:
    """List all skills in the managed directory."""
    if not (skills := ctx.skill_manager.list_all()):
        ctx.console.print("[yellow]No skills found.[/yellow]")
        return
    table = Table(title="Skills", show_header=True, header_style="bold magenta")
    for col, style in [("ID", "cyan"), ("Name", "green"), ("Description", "blue")]:
        table.add_column(col, style=style)
    for sid, s in skills:
        table.add_row(sid, s.name, s.description[:80] or "-")
    ctx.console.print(table)


def show_skill(ctx: SkillsContext, skill_id: str) -> None:
    """Display the full contents of a skill's SKILL.md."""
    sid, info = ctx._find_or_exit(skill_id)
    ctx.console.print(
        Panel(Markdown(info.content), title=f"Skill: {sid}", border_style="cyan")
    )


def create_skill(
    ctx: SkillsContext,
    skill_id: str,
    *,
    name: str,
    description: str,
    instructions: str,
    module: str | None = None,
    force: bool = False,
) -> None:
    """Create a new skill."""
    metadata = {}
    if module:
        metadata["module"] = module

    try:
        created = ctx.skill_manager.create(
            skill_id,
            name=ctx._require(name, "Name"),
            description=ctx._require(description, "Description"),
            instructions=ctx._require(instructions, "Instructions"),
            overwrite=force,
            metadata=metadata,
        )
        ctx.console.print(f"[green]Skill created:[/green] {name} ({created})")
    except FileExistsError:
        ctx.console.print(
            f"[red]Skill already exists:[/red] {skill_id} (use --force to overwrite)"
        )
        raise SystemExit(1)
    except ValueError as exc:
        ctx.console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)


def delete_skill(ctx: SkillsContext, skill_id: str) -> None:
    """Delete an existing skill."""
    sid, info = ctx._find_or_exit(skill_id)
    msg = (
        f"[green]Skill deleted:[/green] {info.name} ({sid})"
        if ctx.skill_manager.delete(sid)
        else f"[red]Error deleting skill: {sid}[/red]"
    )
    ctx.console.print(msg)
