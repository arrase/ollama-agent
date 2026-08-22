"""Application configuration management using YAML-based settings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Self

import yaml  # type: ignore[import-untyped]

from .paths import (
    AGENTS_MD_NAME,
    AGENTS_PATH,
    FS_POLICY_SANDBOXED_PATH,
    FS_POLICY_TRAVERSAL_PATH,
    INSTRUCTIONS_PATH,
    MEMORY_PATH,
    RAG_DIR,
    RAG_POLICY_PATH,
    SETTINGS_PATH,
)


# ---------------------------------------------------------------------------
# Default instructions loader
# ---------------------------------------------------------------------------


def _default_instructions() -> str:
    return (
        resources.files(__package__)
        .joinpath("prompts/default_instructions.md")
        .read_text(encoding="utf-8")
        .strip()
    )


def _default_traversal() -> str:
    return (
        resources.files(__package__)
        .joinpath("prompts/fs_policy_traversal.md")
        .read_text(encoding="utf-8")
        .strip()
    )


def _default_sandboxed() -> str:
    return (
        resources.files(__package__)
        .joinpath("prompts/fs_policy_sandboxed.md")
        .read_text(encoding="utf-8")
        .strip()
    )


def _default_rag_policy() -> str:
    return (
        resources.files(__package__)
        .joinpath("prompts/rag_policy.md")
        .read_text(encoding="utf-8")
        .strip()
    )


# ---------------------------------------------------------------------------
# Settings dataclasses (CUD-inspired)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelSettings:
    name: str = ""
    base_url: str = "http://localhost:11434"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None
    context_window: int = 10000
    reasoning_effort: str = "medium"


@dataclass(slots=True)
class RuntimeSettings:
    allow_traversal: bool = False
    builtin_tool_timeout: int = 30
    collapse_thinking: bool = True
    inherit_env: bool = True


@dataclass(slots=True)
class RAGSettings:
    rag_dir: str = ""
    embedder_model: str = "nomic-embed-text:latest"
    embedder_base_url: str = "http://localhost:11434"
    embedding_dims: int = 768
    default_top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    def __post_init__(self) -> None:
        if not self.rag_dir:
            self.rag_dir = str(RAG_DIR)


@dataclass(slots=True)
class SubAgentMCPServer:
    """An MCP server attached to a subagent."""

    name: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SubAgentSettings:
    """A custom subagent with its own model, skills, and MCP servers."""

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    context_window: int = 0
    mcp_servers: list[SubAgentMCPServer] = field(default_factory=list)


@dataclass(slots=True)
class LangSmithSettings:
    api_key: str = ""
    tracing: str = ""
    project: str = ""
    endpoint: str = ""


@dataclass(slots=True)
class MentionSettings:
    """Configuration for @-mention file/directory context injection."""

    max_file_size: int = 1_048_576       # 1 MB
    max_files: int = 100
    max_total_size: int = 10_485_760     # 10 MB
    max_completions: int = 200


@dataclass(slots=True)
class Settings:
    model: ModelSettings = field(default_factory=ModelSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    mentions: MentionSettings = field(default_factory=MentionSettings)
    subagents: list[SubAgentSettings] = field(default_factory=list)
    langsmith: LangSmithSettings = field(default_factory=LangSmithSettings)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Self:
        raw = raw or {}
        return cls(
            model=_dataclass_from_dict(ModelSettings, raw.get("model")),
            runtime=_dataclass_from_dict(RuntimeSettings, raw.get("runtime")),
            rag=_dataclass_from_dict(RAGSettings, raw.get("rag")),
            mentions=_dataclass_from_dict(MentionSettings, raw.get("mentions")),
            subagents=_subagents_from_list(raw.get("subagents")),
            langsmith=_dataclass_from_dict(LangSmithSettings, raw.get("langsmith")),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"] = {k: v for k, v in d["model"].items() if v is not None}
        ls = d.get("langsmith", {})
        if ls and not any(ls.values()):
            d.pop("langsmith", None)
        return d

    def setup_environment(self) -> None:
        """Inject settings into the environment variables."""
        if self.langsmith.api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith.api_key
        if self.langsmith.tracing:
            os.environ["LANGSMITH_TRACING"] = self.langsmith.tracing
        if self.langsmith.project:
            os.environ["LANGSMITH_PROJECT"] = self.langsmith.project
        if self.langsmith.endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = self.langsmith.endpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataclass_from_dict(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    if raw is None:
        return cls()
    data = dict(raw)
    if "repetition_penalty" in data and "repeat_penalty" not in data:
        data["repeat_penalty"] = data.pop("repetition_penalty")
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})


def _subagents_from_list(
    raw: list[dict[str, Any]] | None,
) -> list[SubAgentSettings]:
    """Parse subagent list from YAML, handling nested mcp_servers."""
    if not raw:
        return []
    subagents: list[SubAgentSettings] = []
    for item in raw:
        mcp_servers = [
            _dataclass_from_dict(SubAgentMCPServer, m)
            for m in (item.get("mcp_servers") or [])
        ]
        sa = _dataclass_from_dict(
            SubAgentSettings,
            {k: v for k, v in item.items() if k != "mcp_servers"},
        )
        sa.mcp_servers = mcp_servers
        subagents.append(sa)
    return subagents


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load_settings(settings_path: Path = SETTINGS_PATH) -> Settings:
    """Load settings from YAML file or create defaults."""
    if not settings_path.exists():
        settings = Settings()
        save_settings(settings, settings_path)
        return settings
    
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    return Settings.from_dict(raw)


def save_settings(settings: Settings, settings_path: Path = SETTINGS_PATH) -> None:
    """Save settings to YAML file atomically."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True)
    tmp_path = settings_path.with_suffix(".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, settings_path)


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


def _load_prompt_file(file_path: Path, default_factory: Callable[[], str]) -> str:
    """Helper to load a prompt file with initial creation from default factory."""
    if not file_path.exists():
        default_text = default_factory()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(default_text + "\n", encoding="utf-8")
        return default_text
    content = file_path.read_text(encoding="utf-8").strip()
    return content if content else default_factory()


def load_instructions(instructions_path: Path = INSTRUCTIONS_PATH) -> str:
    """Load agent instructions from file or return defaults."""
    return _load_prompt_file(instructions_path, _default_instructions)


def load_fs_policy_traversal(policy_path: Path = FS_POLICY_TRAVERSAL_PATH) -> str:
    """Load filesystem traversal policy from file or return defaults."""
    return _load_prompt_file(policy_path, _default_traversal)


def load_fs_policy_sandboxed(policy_path: Path = FS_POLICY_SANDBOXED_PATH) -> str:
    """Load sandboxed filesystem policy from file or return defaults."""
    return _load_prompt_file(policy_path, _default_sandboxed)


def load_rag_policy(policy_path: Path = RAG_POLICY_PATH) -> str:
    """Load RAG policy from file or return defaults."""
    return _load_prompt_file(policy_path, _default_rag_policy)


def ensure_prompt_files(
    *,
    instructions_path: Path = INSTRUCTIONS_PATH,
    traversal_path: Path = FS_POLICY_TRAVERSAL_PATH,
    sandboxed_path: Path = FS_POLICY_SANDBOXED_PATH,
    rag_policy_path: Path = RAG_POLICY_PATH,
) -> None:
    """Ensure all default prompt files exist in the user prompts directory."""
    load_instructions(instructions_path)
    load_fs_policy_traversal(traversal_path)
    load_fs_policy_sandboxed(sandboxed_path)
    load_rag_policy(rag_policy_path)


# ---------------------------------------------------------------------------
# Memory scaffold
# ---------------------------------------------------------------------------


def ensure_memory_file(memory_path: Path = MEMORY_PATH) -> Path:
    """Ensure the MEMORY.md file exists, creating it with defaults if needed."""
    if not memory_path.exists():
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            "# Long-Term Memory\n\nNo persistent memories yet.\n",
            encoding="utf-8",
        )
    return memory_path


def find_agents_file(start_dir: Path | None = None) -> Path | None:
    """Find AGENTS.md in the given directory or its parent hierarchy up to git root."""
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        for candidate in ("AGENTS.md", "agents.md", ".agents.md"):
            target = parent / candidate
            if target.is_file():
                return target
        if (parent / ".git").exists():
            break
    return None


def ensure_agents_file(agents_path: Path = AGENTS_PATH) -> Path:
    """Ensure an AGENTS.md file exists, creating it with defaults if needed."""
    if not agents_path.exists():
        agents_path.parent.mkdir(parents=True, exist_ok=True)
        agents_path.write_text(
            "# Agent Guidelines\n\nProject-specific instructions for AI agents.\n",
            encoding="utf-8",
        )
    return agents_path


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

VALID_RESET_OPTIONS = {"all", "config-file", "system-prompt"}


def reset_config(
    option: str,
    *,
    settings_path: Path = SETTINGS_PATH,
    instructions_path: Path = INSTRUCTIONS_PATH,
    traversal_path: Path = FS_POLICY_TRAVERSAL_PATH,
    sandboxed_path: Path = FS_POLICY_SANDBOXED_PATH,
    rag_policy_path: Path = RAG_POLICY_PATH,
) -> list[str]:
    """Reset configuration or system prompt to defaults."""
    if option not in VALID_RESET_OPTIONS:
        raise ValueError(
            f"Invalid reset option '{option}'. Expected one of: {sorted(VALID_RESET_OPTIONS)}"
        )

    messages: list[str] = []

    if option in ("all", "config-file"):
        if settings_path.exists():
            settings_path.unlink()
        save_settings(Settings(), settings_path)
        messages.append(f"Reset: Restored default configuration at {settings_path}")

    if option in ("all", "system-prompt"):
        if instructions_path.exists():
            instructions_path.unlink()
        if traversal_path.exists():
            traversal_path.unlink()
        if sandboxed_path.exists():
            sandboxed_path.unlink()
        if rag_policy_path.exists():
            rag_policy_path.unlink()
        load_instructions(instructions_path)
        load_fs_policy_traversal(traversal_path)
        load_fs_policy_sandboxed(sandboxed_path)
        load_rag_policy(rag_policy_path)
        messages.append(f"Reset: Restored default system prompt at {instructions_path}")
        messages.append(f"Reset: Restored default traversal policy at {traversal_path}")
        messages.append(f"Reset: Restored default sandboxed policy at {sandboxed_path}")
        messages.append(f"Reset: Restored default RAG policy at {rag_policy_path}")

    return messages

