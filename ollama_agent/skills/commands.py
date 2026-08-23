"""Shared skill management commands used by CLI and REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..i18n import _
from .manager import SkillInfo, SkillManager


class SkillError(Exception):
    """Base exception for skill command failures."""


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be resolved by its ID/prefix."""


class AmbiguousSkillError(SkillError):
    """Raised when a skill ID prefix matches multiple skills."""


class ValidationError(SkillError):
    """Raised when a validation rule fails for skill parameters."""


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
            _("Skill not found: {skill_id}", skill_id=skill_id)
            if not matches
            else _("Ambiguous prefix: {name} -> {matches}", name=skill_id, matches=", ".join(t[0] for t in matches))
        )
        if not matches:
            raise SkillNotFoundError(msg)
        raise AmbiguousSkillError(msg)

    def _require(self, value: str, name: str) -> str:
        if not (cleaned := value.strip()):
            raise ValidationError(_("{name} cannot be empty.", name=name))
        return cleaned


def list_skills(ctx: SkillsContext) -> None:
    """List all skills in the managed directory."""
    if not (skills := ctx.skill_manager.list_all()):
        ctx.console.print(f"[yellow]{_('No skills found.')}[/yellow]")
        return
    table = Table(title=_("Skills"), show_header=True, header_style="bold magenta")
    for col, style in [(_("ID"), "cyan"), (_("Name"), "green"), (_("Description"), "blue")]:
        table.add_column(col, style=style)
    for sid, s in skills:
        table.add_row(sid, s.name, s.description[:80] or "-")
    ctx.console.print(table)


def show_skill(ctx: SkillsContext, skill_id: str) -> None:
    """Display the full contents of a skill's SKILL.md."""
    sid, info = ctx._find_or_exit(skill_id)
    ctx.console.print(
        Panel(Markdown(info.content), title=_("Skill: {sid}", sid=sid), border_style="cyan")
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
            name=ctx._require(name, _("Name")),
            description=ctx._require(description, _("Description")),
            instructions=ctx._require(instructions, _("Instructions")),
            overwrite=force,
            metadata=metadata,
        )
        ctx.console.print(f"[green]✓ {_('Skill created: {name} ({created})', name=name, created=created)}[/green]")
    except FileExistsError as exc:
        ctx.console.print(
            f"[red]{_('Skill already exists: {skill_id} (use --force to overwrite)', skill_id=skill_id)}[/red]"
        )
        raise SkillError(_("Skill already exists: {skill_id}", skill_id=skill_id)) from exc
    except ValueError as exc:
        ctx.console.print(f"[red]{exc}[/red]")
        raise ValidationError(str(exc)) from exc


def delete_skill(ctx: SkillsContext, skill_id: str) -> None:
    """Delete an existing skill."""
    sid, info = ctx._find_or_exit(skill_id)
    if not ctx.skill_manager.delete(sid):
        ctx.console.print(f"[red]{_('Error deleting skill: {sid}', sid=sid)}[/red]")
        raise SkillError(_("Error deleting skill: {sid}", sid=sid))
    ctx.console.print(f"[green]✓ {_('Skill deleted: {name} ({sid})', name=info.name, sid=sid)}[/green]")
