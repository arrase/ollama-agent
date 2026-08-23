"""Skill management utilities following the Agent Skills specification."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..core import BaseFileStoreManager, validate_identifier
from ..settings.paths import SKILLS_DIR

logger = logging.getLogger(__name__)

# Maximum SKILL.md size (10 MB) as per spec.
_MAX_SKILL_SIZE = 10 * 1024 * 1024


@dataclass(slots=True)
class SkillInfo:
    """Parsed metadata and content of a SKILL.md file."""

    name: str
    description: str
    content: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into YAML frontmatter dict and body markdown."""
    if text.startswith("---") and (end := text.find("---", 3)) != -1:
        try:
            meta = yaml.safe_load(text[3:end])
            if isinstance(meta, dict):
                return meta, text[end + 3 :].lstrip("\n")
        except yaml.YAMLError as exc:
            logger.warning("Invalid YAML frontmatter in skill: %s", exc)
    return {}, text


def _read_skill(skill_dir: Path) -> SkillInfo | None:
    """Read and parse a SKILL.md inside *skill_dir*."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        skill_file = skill_dir / "skill.md"
        if not skill_file.is_file():
            return None
    if skill_file.stat().st_size > _MAX_SKILL_SIZE:
        logger.warning("Skipping skill %s: SKILL.md exceeds 10 MB", skill_dir.name)
        return None
    try:
        raw = skill_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Error reading %s: %s", skill_file, exc)
        return None
    meta, _body = _parse_frontmatter(raw)
    return SkillInfo(
        name=str(meta.get("name", skill_dir.name)),
        description=str(meta.get("description", ""))[:1024],
        content=raw,
        path=skill_dir,
        metadata=meta,
    )


def _remove_readonly(func: Any, path: str, exc: Any) -> None:
    os.chmod(path, 0o777)
    func(path)


class SkillManager(BaseFileStoreManager[SkillInfo]):
    """Manages skills persisted as subdirectories with SKILL.md files."""

    DEFAULT_DIR = SKILLS_DIR
    _ext: str = ""

    def __init__(self, skills_dir: Path = SKILLS_DIR) -> None:
        super().__init__(skills_dir)

    @staticmethod
    def validate_skill_id(skill_id: str) -> str:
        """Validate skill_id: letters, numbers, underscore, dash only."""
        return validate_identifier(skill_id, "skill_id")

    def get(self, item_id: str) -> SkillInfo | None:
        """Load a single skill by ID."""
        d = self._path(item_id)
        if not d.is_dir():
            return None
        return _read_skill(d)

    def find_matches(self, prefix: str) -> list[tuple[str, SkillInfo]]:
        """Return all skills whose id starts with *prefix*."""
        if not (prefix := prefix.strip()):
            return []
        if (skill := self.get(prefix)) is not None:
            return [(prefix, skill)]
        return [
            (d.name, s)
            for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix) and (s := _read_skill(d)) is not None
        ]

    def list_all(self) -> list[tuple[str, SkillInfo]]:
        """List all skills sorted by name."""
        skills = [
            (d.name, s)
            for d in self.base_dir.iterdir()
            if d.is_dir() and (s := _read_skill(d)) is not None
        ]
        return sorted(skills, key=lambda x: x[1].name.lower())

    def create(
        self,
        skill_id: str,
        *,
        name: str,
        description: str,
        instructions: str,
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a skill directory with a SKILL.md and return the skill ID."""
        skill_id = self.validate_skill_id(skill_id)
        skill_dir = self._path(skill_id)
        if skill_dir.exists() and not overwrite:
            raise FileExistsError(f"Skill already exists: {skill_id}")
        skill_dir.mkdir(parents=True, exist_ok=True)

        meta = {"name": name, "description": description}
        if metadata:
            meta.update(metadata)

        frontmatter = yaml.safe_dump(
            meta,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()

        content = f"---\n{frontmatter}\n---\n\n# {name}\n\n{instructions}\n"
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        return skill_id

    def delete(self, item_id: str) -> bool:
        """Delete a skill directory entirely."""
        skill_dir = self._path(item_id)
        if not skill_dir.is_dir():
            return False
        try:
            shutil.rmtree(skill_dir, onexc=_remove_readonly)
            return True
        except OSError as exc:
            logger.error("Error deleting skill %s: %s", item_id, exc)
            return False
