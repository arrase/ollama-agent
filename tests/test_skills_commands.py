from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from ollama_agent.skills.commands import (
    AmbiguousSkillError,
    SkillError,
    SkillNotFoundError,
    SkillsContext,
    ValidationError,
    create_skill,
    delete_skill,
    list_skills,
    show_skill,
)
from ollama_agent.skills.manager import SkillManager


class TestSkillsCommands(unittest.TestCase):
    """Unit tests for skills command operations and resolution."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mgr = SkillManager(skills_dir=Path(self.temp_dir.name), builtin_skills_dir=None)
        self.console = Console(file=io.StringIO(), record=True)
        self.ctx = SkillsContext(console=self.console, skill_manager=self.mgr)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_require_validation(self) -> None:
        self.assertEqual(self.ctx._require("  valid value  ", "Field"), "valid value")
        with self.assertRaises(ValidationError):
            self.ctx._require("   ", "Field")

    def test_create_and_show_skill(self) -> None:
        create_skill(
            self.ctx,
            "git-helper",
            name="Git Helper",
            description="Assists with git",
            instructions="Run git status",
        )
        skill = self.mgr.get("git-helper")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertEqual(skill.name, "Git Helper")

        show_skill(self.ctx, "git-helper")
        out = self.console.export_text()
        self.assertIn("Git Helper", out)
        self.assertIn("Run git status", out)

    def test_list_and_delete_skill(self) -> None:
        create_skill(
            self.ctx,
            "skill-a",
            name="Skill A",
            description="Desc A",
            instructions="Inst A",
        )
        list_skills(self.ctx)
        out = self.console.export_text()
        self.assertIn("Skill A", out)

        delete_skill(self.ctx, "skill-a")
        with self.assertRaises(FileNotFoundError):
            self.mgr.get("skill-a")

    def test_find_or_exit_errors(self) -> None:
        with self.assertRaises(SkillNotFoundError):
            self.ctx._find_or_exit("nonexistent")

        with self.assertRaises(ValidationError):
            self.ctx._find_or_exit("invalid/name")

        create_skill(self.ctx, "test-1", name="T1", description="D1", instructions="I1")
        create_skill(self.ctx, "test-2", name="T2", description="D2", instructions="I2")

        with self.assertRaises(AmbiguousSkillError):
            self.ctx._find_or_exit("test")

    def test_delete_builtin_skill_fails(self) -> None:
        mgr = SkillManager(skills_dir=Path(self.temp_dir.name))
        ctx = SkillsContext(console=self.console, skill_manager=mgr)
        with self.assertRaises(SkillError):
            delete_skill(ctx, "skill-creator")
