from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.middleware.memory import MemoryMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from ollama_agent.agent import AgentRuntime
from ollama_agent.settings import (
    AGENTS_MD_NAME,
    Settings,
    find_agents_file,
)


class TestAgentsMdSupport(unittest.IsolatedAsyncioTestCase):
    """Unit tests for AGENTS.md specification support and integration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_find_agents_file_in_cwd(self) -> None:
        agents_file = self.base_path / AGENTS_MD_NAME
        agents_file.write_text("# Project Agents\n\nConventions here.\n", encoding="utf-8")

        found = find_agents_file(self.base_path)
        self.assertEqual(found, agents_file)

    def test_find_agents_file_in_parent_directory(self) -> None:
        agents_file = self.base_path / "AGENTS.md"
        agents_file.write_text("# Root Instructions\n", encoding="utf-8")

        # Create git root marker at base_path
        (self.base_path / ".git").mkdir()

        # Create nested subdirectory
        nested_dir = self.base_path / "src" / "deep" / "pkg"
        nested_dir.mkdir(parents=True)

        found = find_agents_file(nested_dir)
        self.assertEqual(found, agents_file)

    def test_find_agents_file_stops_at_git_root(self) -> None:
        # AGENTS.md outside repo
        outside_agents = self.base_path / "AGENTS.md"
        outside_agents.write_text("# Outside\n", encoding="utf-8")

        repo_dir = self.base_path / "my_repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        sub_dir = repo_dir / "src"
        sub_dir.mkdir()

        found = find_agents_file(sub_dir)
        self.assertIsNone(found)

    def test_find_agents_file_name_variations(self) -> None:
        for filename in ("AGENTS.md", "agents.md", ".agents.md"):
            with tempfile.TemporaryDirectory() as td:
                target_dir = Path(td).resolve()
                f = target_dir / filename
                f.write_text("# Variation\n", encoding="utf-8")
                found = find_agents_file(target_dir)
                self.assertEqual(found, f)

    def test_find_agents_file_returns_none_when_missing(self) -> None:
        empty_dir = self.base_path / "empty"
        empty_dir.mkdir()
        self.assertIsNone(find_agents_file(empty_dir))

    def test_memory_middleware_loads_agents_md_into_system_prompt(self) -> None:
        project_dir = self.base_path / "project"
        project_dir.mkdir()
        agents_file = project_dir / "AGENTS.md"
        agents_file.write_text("# Project Standards\n- Use pytest\n- PEP 8\n", encoding="utf-8")

        agent_dir = self.base_path / "agent_home"
        agent_dir.mkdir()
        mem_file = agent_dir / "MEMORY.md"
        mem_file.write_text("# Long-Term Memory\n- Preference: Python\n", encoding="utf-8")

        default_backend = LocalShellBackend(root_dir=project_dir, virtual_mode=True)
        agent_backend = FilesystemBackend(root_dir=agent_dir, virtual_mode=True)
        backend = CompositeBackend(
            default=default_backend,
            routes={"/agent/": agent_backend},
        )

        middleware = MemoryMiddleware(
            backend=backend,
            sources=["/agent/MEMORY.md", "/AGENTS.md"],
        )

        state: dict[str, Any] = {}
        runtime_mock = MagicMock()
        update = middleware.before_agent(state, runtime_mock, {})  # type: ignore[arg-type]
        self.assertIsNotNone(update)
        if update:
            state.update(update)  # type: ignore[arg-type]

        contents = state.get("memory_contents", {})
        self.assertIn("/agent/MEMORY.md", contents)
        self.assertIn("/AGENTS.md", contents)
        self.assertIn("Project Standards", contents["/AGENTS.md"])

        class DummyRequest:
            def __init__(self, state_dict: dict[str, Any]):
                self.state = state_dict
                self.system_message = SystemMessage(content="Base system prompt.")
                self.model = None

            def override(self, **kwargs: Any) -> Any:
                return kwargs

        req = DummyRequest(state)
        modified = middleware.modify_request(req)  # type: ignore[arg-type]
        rendered = modified["system_message"].content  # type: ignore[index]

        if isinstance(rendered, list):
            combined_text = "".join(b.get("text", "") for b in rendered if isinstance(b, dict))
        else:
            combined_text = str(rendered)

        self.assertIn("/agent/MEMORY.md", combined_text)
        self.assertIn("/AGENTS.md", combined_text)
        self.assertIn("Project Standards", combined_text)

    async def test_agent_runtime_routes_ancestor_agents_md(self) -> None:
        repo_root = self.base_path / "my_workspace"
        repo_root.mkdir()
        (repo_root / ".git").mkdir()
        agents_file = repo_root / "AGENTS.md"
        agents_file.write_text("# Repo Wide Rules\n", encoding="utf-8")

        sub_dir = repo_root / "nested" / "subpackage"
        sub_dir.mkdir(parents=True)

        settings = Settings()
        runtime = AgentRuntime(settings=settings)

        mock_deep_agent = MagicMock()
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.profile = None
        mock_model.num_ctx = 8192
        mock_model.effective_params = {}

        with (
            patch("pathlib.Path.cwd", return_value=sub_dir),
            patch("ollama_agent.agent.agent.ensure_model_supports_tools", AsyncMock()),
            patch("ollama_agent.agent.agent.create_ollama_chat_model", AsyncMock(return_value=mock_model)),
            patch("ollama_agent.agent.agent.create_deep_agent", return_value=mock_deep_agent) as mock_create_agent,
        ):
            await runtime._build_graph()

            mock_create_agent.assert_called_once()
            call_kwargs = mock_create_agent.call_args.kwargs

            # Verify memory includes /project/AGENTS.md
            self.assertIn("/agent/MEMORY.md", call_kwargs["memory"])
            self.assertIn("/project/AGENTS.md", call_kwargs["memory"])

            # Verify backend routes has /project/
            backend = call_kwargs["backend"]
            self.assertIn("/project/", backend.routes)


if __name__ == "__main__":
    unittest.main()
