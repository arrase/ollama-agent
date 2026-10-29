"""Skill management utilities following the Agent Skills specification."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..core import BaseFileStoreManager, require_text, validate_identifier
from ..i18n import _
from ..settings.paths import BUILTIN_SKILLS_DIR, SKILLS_DIR

logger = logging.getLogger(__name__)

# Maximum SKILL.md size (10 MB) as per spec.
_MAX_SKILL_SIZE = 10 * 1024 * 1024
_SKILL_FILENAME = "SKILL.md"

_FRONTMATTER_CLOSE = re.compile(r"^---\s*$", re.MULTILINE)


@dataclass(slots=True)
class SkillInfo:
    """Parsed metadata and content of a SKILL.md file."""

    name: str
    description: str
    content: str


def _find_skill_file(skill_dir: Path) -> Path:
    """Find SKILL.md (or skill.md) inside skill_dir."""
    upper = skill_dir / _SKILL_FILENAME
    if upper.is_file():
        return upper
    lower = skill_dir / "skill.md"
    if lower.is_file():
        return lower
    raise ValueError(_("Missing SKILL.md: {path}", path=skill_dir))


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into YAML frontmatter dict and body markdown."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    first_line_end = stripped.find("\n")
    if first_line_end == -1 or stripped[:first_line_end].strip() != "---":
        return {}, text
    match = _FRONTMATTER_CLOSE.search(stripped, first_line_end + 1)
    if not match:
        raise ValueError(_("Unclosed YAML frontmatter"))
    yaml_str = stripped[first_line_end + 1 : match.start()]
    try:
        meta = yaml.safe_load(yaml_str)
    except yaml.YAMLError as exc:
        raise ValueError(_("Invalid YAML frontmatter: {exc}", exc=exc)) from exc
    if not isinstance(meta, dict):
        raise ValueError(_("YAML frontmatter must be a mapping"))
    return meta, stripped[match.end() :].lstrip("\n")


def _read_skill(skill_dir: Path) -> SkillInfo:
    """Read and parse the SKILL.md inside *skill_dir*."""
    skill_file = _find_skill_file(skill_dir)
    if skill_file.stat().st_size > _MAX_SKILL_SIZE:
        raise ValueError(_("SKILL.md exceeds 10 MB: {path}", path=skill_file))
    raw = skill_file.read_text(encoding="utf-8")
    meta, _body = _parse_frontmatter(raw)
    if "name" not in meta or "description" not in meta:
        raise ValueError(_("Skill frontmatter must define non-empty 'name' and 'description': {path}", path=skill_file))
    name = str(meta["name"]).strip()
    description = str(meta["description"]).strip()
    if not name or not description:
        raise ValueError(_("Skill frontmatter must define non-empty 'name' and 'description': {path}", path=skill_file))
    return SkillInfo(name=name, description=description, content=raw)


class SkillManager(BaseFileStoreManager[SkillInfo]):
    """Manages skills persisted as subdirectories with SKILL.md files."""

    _ext: str = ""
    not_found_label: str = "skill_id"

    def __init__(
        self,
        skills_dir: Path = SKILLS_DIR,
        builtin_skills_dir: Path = BUILTIN_SKILLS_DIR,
    ) -> None:
        super().__init__(skills_dir)
        self.builtin_dir = builtin_skills_dir.resolve()

    def _collect_skills(self, prefix: str = "") -> dict[str, SkillInfo]:
        """Collect all skills matching *prefix*, allowing user skills to override built-ins.

        A single malformed ``SKILL.md`` (easy to produce via the skill-creator tool)
        is skipped rather than breaking ``/skill list`` for every other skill.
        """
        skills: dict[str, SkillInfo] = {}
        search_dirs = [d for d in (self.builtin_dir, self.base_dir) if d.is_dir()]
        for directory in search_dirs:
            for d in directory.iterdir():
                if not d.is_dir() or not d.name.startswith(prefix):
                    continue
                if not ((d / _SKILL_FILENAME).is_file() or (d / "skill.md").is_file()):
                    continue
                try:
                    skills[d.name] = _read_skill(d)
                except (ValueError, OSError) as exc:
                    logger.warning("Skipping invalid skill %s: %s", d.name, exc)
        return skills

    def get(self, item_id: str) -> SkillInfo:
        """Load a single skill by ID. Raise FileNotFoundError if missing."""
        user_dir = self._path(item_id)
        if user_dir.is_dir():
            return _read_skill(user_dir)
        builtin_skill_dir = self.builtin_dir / user_dir.name
        if builtin_skill_dir.is_dir():
            return _read_skill(builtin_skill_dir)
        raise FileNotFoundError(str(user_dir))

    def find_matches(self, prefix: str) -> list[tuple[str, SkillInfo]]:
        """Return all skills whose id starts with *prefix*."""
        prefix = validate_identifier(prefix, self.not_found_label)
        return sorted(self._collect_skills(prefix).items(), key=lambda x: x[1].name.lower())

    def list_all(self) -> list[tuple[str, SkillInfo]]:
        """List all skills sorted by name."""
        return sorted(self._collect_skills().items(), key=lambda x: x[1].name.lower())

    def create(
        self,
        skill_id: str,
        *,
        name: str,
        description: str,
        instructions: str,
        overwrite: bool = False,
    ) -> str:
        """Create a skill directory with a SKILL.md and return the skill ID."""
        skill_dir = self._path(skill_id)
        clean_name = require_text(name, _("Name"), ValueError)
        clean_desc = require_text(description, _("Description"), ValueError)
        clean_inst = require_text(instructions, _("Instructions"), ValueError)

        if skill_dir.exists() and not overwrite:
            raise FileExistsError(_("Skill already exists: {skill_id}", skill_id=skill_dir.name))
        skill_dir.mkdir(parents=True, exist_ok=True)

        frontmatter = yaml.safe_dump(
            {"name": clean_name, "description": clean_desc},
            allow_unicode=True,
            default_flow_style=False,
        ).strip()

        content = f"---\n{frontmatter}\n---\n\n# {clean_name}\n\n{clean_inst}\n"
        (skill_dir / _SKILL_FILENAME).write_text(content, encoding="utf-8")
        return skill_dir.name

    def delete(self, item_id: str) -> None:
        """Delete a skill directory entirely. Raise FileNotFoundError if missing."""
        user_dir = self._path(item_id)
        if user_dir.is_dir():
            shutil.rmtree(user_dir)
            return
        if (self.builtin_dir / user_dir.name).is_dir():
            raise ValueError(_("Built-in skills cannot be deleted: {name}", name=user_dir.name))
        raise FileNotFoundError(str(user_dir))
