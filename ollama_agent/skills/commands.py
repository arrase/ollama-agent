"""Shared skill management commands used by CLI and REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..core.resource_manager import require_text, resolve_unique_match
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

    def _resolve_skill(self, skill_id: str) -> tuple[str, SkillInfo]:
        try:
            matches = self.skill_manager.find_matches(skill_id)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        except (FileNotFoundError, OSError) as exc:
            raise SkillNotFoundError(
                _("{label} not found: {prefix}", label=_("Skill"), prefix=skill_id)
            ) from exc
        return resolve_unique_match(
            matches,
            skill_id,
            label=_("Skill"),
            not_found_error=SkillNotFoundError,
            ambiguous_error=AmbiguousSkillError,
        )

    def _require(self, value: str, name: str) -> str:
        return require_text(value, name, ValidationError)


def list_skills(ctx: SkillsContext) -> None:
    """List all skills in the managed directory."""
    if not (skills := ctx.skill_manager.list_all()):
        ctx.console.print(f"[yellow]{_('No skills found.')}[/yellow]")
        return
    table = Table(title=_("Skills"), show_header=True, header_style="bold magenta")
    for col, style in [(_("ID"), "cyan"), (_("Name"), "green"), (_("Description"), "blue")]:
        table.add_column(col, style=style)
    for sid, s in skills:
        table.add_row(sid, s.name, s.description[:80])
    ctx.console.print(table)


def show_skill(ctx: SkillsContext, skill_id: str) -> None:
    """Display the full contents of a skill's SKILL.md."""
    sid, info = ctx._resolve_skill(skill_id)
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
    force: bool = False,
) -> None:
    """Create a new skill."""
    name = ctx._require(name, _("Name"))
    description = ctx._require(description, _("Description"))
    instructions = ctx._require(instructions, _("Instructions"))
    try:
        created = ctx.skill_manager.create(
            skill_id,
            name=name,
            description=description,
            instructions=instructions,
            overwrite=force,
        )
    except FileExistsError as exc:
        raise SkillError(
            _("Skill already exists: {skill_id} (use --force to overwrite)", skill_id=skill_id)
        ) from exc
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    ctx.console.print(f"[green]✓ {_('Skill created: {name} ({created})', name=name, created=created)}[/green]")


def delete_skill(ctx: SkillsContext, skill_id: str) -> None:
    """Delete an existing skill."""
    sid, info = ctx._resolve_skill(skill_id)
    try:
        ctx.skill_manager.delete(sid)
    except (FileNotFoundError, OSError) as exc:
        raise SkillNotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise SkillError(str(exc)) from exc
    ctx.console.print(f"[green]✓ {_('Skill deleted: {name} ({sid})', name=info.name, sid=sid)}[/green]")
