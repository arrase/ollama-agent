"""Skill management utilities following the Agent Skills specification."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..core import BaseFileStoreManager, require_text, validate_identifier
from ..i18n import _
from ..settings.paths import BUILTIN_SKILLS_DIR, SKILLS_DIR

# Maximum SKILL.md size (10 MB) as per spec.
_MAX_SKILL_SIZE = 10 * 1024 * 1024

_FRONTMATTER_CLOSE = re.compile(r"^---\s*$", re.MULTILINE)


@dataclass(slots=True)
class SkillInfo:
    """Parsed metadata and content of a SKILL.md file."""

    name: str
    description: str
    content: str


def _find_skill_file(skill_dir: Path) -> Path | None:
    """Find SKILL.md (or skill.md) inside skill_dir."""
    if not skill_dir.is_dir():
        return None
    if (skill_dir / "SKILL.md").is_file():
        return skill_dir / "SKILL.md"
    if (skill_dir / "skill.md").is_file():
        return skill_dir / "skill.md"
    return None


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into YAML frontmatter dict and body markdown."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    first_line_end = stripped.find("\n")
    if first_line_end == -1 or stripped[:first_line_end].strip() != "---":
        return {}, text
    match = _FRONTMATTER_CLOSE.search(stripped, first_line_end + 1)
    if match is None:
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
    if skill_file is None:
        raise ValueError(_("Missing SKILL.md: {path}", path=skill_dir))
    if skill_file.stat().st_size > _MAX_SKILL_SIZE:
        raise ValueError(_("SKILL.md exceeds 10 MB: {path}", path=skill_file))
    raw = skill_file.read_text(encoding="utf-8")
    meta, _body = _parse_frontmatter(raw)
    if "name" not in meta or "description" not in meta or not meta["name"] or not meta["description"]:
        raise ValueError(
            _("Skill frontmatter must define non-empty 'name' and 'description': {path}", path=skill_file)
        )
    return SkillInfo(name=str(meta["name"]), description=str(meta["description"]), content=raw)


class SkillManager(BaseFileStoreManager[SkillInfo]):
    """Manages skills persisted as subdirectories with SKILL.md files."""

    _ext: str = ""

    def __init__(
        self,
        skills_dir: Path = SKILLS_DIR,
        builtin_skills_dir: Path | None = BUILTIN_SKILLS_DIR,
    ) -> None:
        super().__init__(skills_dir)
        self.builtin_dir = builtin_skills_dir.resolve() if builtin_skills_dir is not None else None

    @staticmethod
    def validate_skill_id(skill_id: str) -> str:
        """Validate skill_id: letters, numbers, underscore, dash only."""
        return validate_identifier(skill_id, "skill_id")

    def _collect_skills(self, prefix: str = "") -> dict[str, SkillInfo]:
        """Collect all skills matching *prefix*, allowing user skills to override built-ins."""
        skills: dict[str, SkillInfo] = {}
        if self.builtin_dir is not None and self.builtin_dir.is_dir():
            for d in self.builtin_dir.iterdir():
                if d.is_dir() and d.name.startswith(prefix) and _find_skill_file(d) is not None:
                    skills[d.name] = _read_skill(d)
        if self.base_dir.is_dir():
            for d in self.base_dir.iterdir():
                if d.is_dir() and d.name.startswith(prefix) and _find_skill_file(d) is not None:
                    skills[d.name] = _read_skill(d)
        return skills

    def get(self, item_id: str) -> SkillInfo:
        """Load a single skill by ID. Raise FileNotFoundError if missing."""
        valid_id = self.validate_skill_id(item_id)
        user_dir = self._path(valid_id)
        if user_dir.is_dir():
            return _read_skill(user_dir)
        if self.builtin_dir is not None and (self.builtin_dir / valid_id).is_dir():
            return _read_skill(self.builtin_dir / valid_id)
        raise FileNotFoundError(str(user_dir))

    def find_matches(self, prefix: str) -> list[tuple[str, SkillInfo]]:
        """Return all skills whose id starts with *prefix*."""
        prefix = self.validate_skill_id(prefix)
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
        skill_id = self.validate_skill_id(skill_id)
        clean_name = require_text(name, _("Name"), ValueError)
        clean_desc = require_text(description, _("Description"), ValueError)
        clean_inst = require_text(instructions, _("Instructions"), ValueError)

        skill_dir = self._path(skill_id)
        if skill_dir.exists() and not overwrite:
            raise FileExistsError(_("Skill already exists: {skill_id}", skill_id=skill_id))
        skill_dir.mkdir(parents=True, exist_ok=True)

        frontmatter = yaml.safe_dump(
            {"name": clean_name, "description": clean_desc},
            allow_unicode=True,
            default_flow_style=False,
        ).strip()

        content = f"---\n{frontmatter}\n---\n\n# {clean_name}\n\n{clean_inst}\n"
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        return skill_id

    def delete(self, item_id: str) -> None:
        """Delete a skill directory entirely. Raise FileNotFoundError if missing."""
        valid_id = self.validate_skill_id(item_id)
        user_dir = self._path(valid_id)
        if user_dir.is_dir():
            shutil.rmtree(user_dir)
            return
        if self.builtin_dir is not None and (self.builtin_dir / valid_id).is_dir():
            raise ValueError(_("Built-in skills cannot be deleted: {name}", name=valid_id))
        raise FileNotFoundError(str(user_dir))
