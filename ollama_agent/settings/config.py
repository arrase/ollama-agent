"""Application configuration management using YAML-based settings."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any, Self

import yaml  # type: ignore[import-untyped]

from .paths import INSTRUCTIONS_PATH, MEMORY_PATH, SETTINGS_PATH, RAG_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default instructions loader
# ---------------------------------------------------------------------------


def _default_instructions() -> str:
    try:
        return (
            resources.files(__package__)
            .joinpath("default_instructions.md")
            .read_text(encoding="utf-8")
            .strip()
        )
    except Exception:
        return "You are an AI Assistant."


# ---------------------------------------------------------------------------
# Settings dataclasses (CUD-inspired)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelSettings:
    name: str = "gemma4:26b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    context_window: int | None = None
    reasoning_effort: str = "medium"


@dataclass(slots=True)
class RuntimeSettings:
    allow_traversal: bool = True
    builtin_tool_timeout: int = 30


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
    langsmith: LangSmithSettings | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Self:
        raw = raw or {}
        langsmith_raw = raw.get("langsmith")
        return cls(
            model=_dataclass_from_dict(ModelSettings, raw.get("model")),
            runtime=_dataclass_from_dict(RuntimeSettings, raw.get("runtime")),
            rag=_dataclass_from_dict(RAGSettings, raw.get("rag")),
            mentions=_dataclass_from_dict(MentionSettings, raw.get("mentions")),
            subagents=_subagents_from_list(raw.get("subagents")),
            langsmith=_dataclass_from_dict(LangSmithSettings, langsmith_raw) if langsmith_raw else None,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("langsmith") is None:
            d.pop("langsmith", None)
        return d

    def setup_environment(self) -> None:
        """Inject settings into the environment variables."""
        if self.langsmith:
            import os
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
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in valid})


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
    """Save settings to YAML file."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True)
    settings_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


def load_instructions(instructions_path: Path = INSTRUCTIONS_PATH) -> str:
    """Load agent instructions from file or return defaults."""
    if not instructions_path.exists():
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        instructions_path.write_text(_default_instructions() + "\n", encoding="utf-8")
        return _default_instructions()
    try:
        return (
            instructions_path.read_text(encoding="utf-8").strip()
            or _default_instructions()
        )
    except Exception:
        return _default_instructions()


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


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def reset_config(option: str) -> None:
    """Reset configuration or system prompt to defaults."""
    if option in ("all", "config-file"):
        if SETTINGS_PATH.exists():
            SETTINGS_PATH.unlink()
        save_settings(Settings())
        print(f"Reset: Restored default configuration at {SETTINGS_PATH}")

    if option in ("all", "system-prompt"):
        if INSTRUCTIONS_PATH.exists():
            INSTRUCTIONS_PATH.unlink()
        load_instructions()
        print(f"Reset: Restored default system prompt at {INSTRUCTIONS_PATH}")
