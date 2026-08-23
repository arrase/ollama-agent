from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_agent.settings.paths import BUILTIN_SKILLS_DIR
from ollama_agent.skills.manager import (
    SkillManager,
    _parse_frontmatter,
    _read_skill,
)


class TestSkillsManager(unittest.TestCase):
    """Unit tests for SkillInfo and SkillManager persistence."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.skills_dir = Path(self.temp_dir.name)
        self.mgr = SkillManager(self.skills_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parse_frontmatter_valid(self) -> None:
        text = "---\nname: PDF Reader\ndescription: Reads PDF documents\n---\n\n# Instructions\nFollow these steps."
        meta, body = _parse_frontmatter(text)
        self.assertEqual(meta.get("name"), "PDF Reader")
        self.assertEqual(meta.get("description"), "Reads PDF documents")
        self.assertIn("# Instructions", body)

    def test_parse_frontmatter_without_frontmatter(self) -> None:
        text = "# Plain Markdown\nNo frontmatter here."
        meta, body = _parse_frontmatter(text)
        self.assertEqual(meta, {})
        self.assertEqual(body, text)

    def test_create_and_get_skill(self) -> None:
        created_id = self.mgr.create(
            "git-helper",
            name="Git Helper",
            description="Assists with git commands",
            instructions="Use git status before committing.",
        )
        self.assertEqual(created_id, "git-helper")

        loaded = self.mgr.get("git-helper")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.name, "Git Helper")
        self.assertEqual(loaded.description, "Assists with git commands")
        self.assertIn("Use git status", loaded.content)

    def test_create_existing_skill_without_overwrite_raises(self) -> None:
        self.mgr.create("s1", name="N1", description="D1", instructions="I1")
        with self.assertRaises(FileExistsError):
            self.mgr.create("s1", name="N1", description="D1", instructions="I1", overwrite=False)

    def test_find_matches_prefix(self) -> None:
        self.mgr.create("py-lint", name="Python Linter", description="Lints code", instructions="Run ruff")
        self.mgr.create("py-test", name="Python Tester", description="Runs tests", instructions="Run pytest")

        matches = self.mgr.find_matches("py-")
        self.assertEqual(len(matches), 2)

    def test_delete_skill(self) -> None:
        self.mgr.create("temp-skill", name="T", description="D", instructions="I")
        self.assertTrue(self.mgr.delete("temp-skill"))
        self.assertIsNone(self.mgr.get("temp-skill"))
        self.assertFalse(self.mgr.delete("temp-skill"))

    def test_list_all_sorted(self) -> None:
        self.mgr.create("b-skill", name="Beta Skill", description="d", instructions="i")
        self.mgr.create("a-skill", name="Alpha Skill", description="d", instructions="i")

        all_skills = self.mgr.list_all()
        self.assertEqual(len(all_skills), 2)
        self.assertEqual(all_skills[0][1].name, "Alpha Skill")
        self.assertEqual(all_skills[1][1].name, "Beta Skill")

    def test_get_skill_with_lowercase_filename(self) -> None:
        skill_dir = self.skills_dir / "custom-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "skill.md").write_text(
            "---\nname: Lowercase Skill\ndescription: Discovered via lowercase\n---\n# Help",
            encoding="utf-8",
        )
        loaded = self.mgr.get("custom-skill")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.name, "Lowercase Skill")

    def test_builtin_skills_present_and_valid(self) -> None:
        self.assertTrue((BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md").is_file())
        self.assertTrue((BUILTIN_SKILLS_DIR / "task-creator" / "SKILL.md").is_file())

        sc = _read_skill(BUILTIN_SKILLS_DIR / "skill-creator")
        self.assertIsNotNone(sc)
        assert sc is not None
        self.assertEqual(sc.name, "skill-creator")
        self.assertTrue(len(sc.description) > 0)
        self.assertIn("SKILL.md", sc.content)

        tc = _read_skill(BUILTIN_SKILLS_DIR / "task-creator")
        self.assertIsNotNone(tc)
        assert tc is not None
        self.assertEqual(tc.name, "task-creator")
        self.assertTrue(len(tc.description) > 0)
        self.assertIn("task_id", tc.content)


if __name__ == "__main__":
    unittest.main()
