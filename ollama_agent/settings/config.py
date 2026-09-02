"""Application configuration management using YAML-based settings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any, Self, TypeVar

import yaml  # type: ignore[import-untyped]
from jinja2 import Environment, StrictUndefined

from ..core import atomic_write_text
from ..i18n import _
from .paths import (
    INSTRUCTIONS_PATH,
    MEMORY_PATH,
    RAG_DIR,
    SETTINGS_PATH,
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
    context_window: int | str = 10000
    reasoning_effort: str = "medium"


@dataclass(slots=True)
class RuntimeSettings:
    allow_traversal: bool = False
    builtin_tool_timeout: int = 30
    collapse_thinking: bool = True
    inherit_env: bool = True
    language: str = ""


@dataclass(slots=True)
class RAGSettings:
    rag_dir: str = str(RAG_DIR)
    embedder_model: str = "nomic-embed-text:latest"
    embedder_base_url: str = "http://localhost:11434"
    embedding_dims: int = 768
    default_top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    def __post_init__(self) -> None:
        if not self.rag_dir.strip():
            raise ValueError(_("Setting 'rag.rag_dir' must be a non-empty string"))


@dataclass(slots=True)
class SubAgentMCPServer:
    """An MCP server attached to a subagent."""

    name: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "args": self.args,
            "env": self.env,
        }


@dataclass(slots=True)
class SubAgentSettings:
    """A custom subagent with its own model, skills, and MCP servers."""

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    context_window: int | str = 0
    mcp_servers: list[SubAgentMCPServer] = field(default_factory=list)


@dataclass(slots=True)
class LangSmithSettings:
    api_key: str = ""
    tracing: bool = False
    project: str = ""
    endpoint: str = ""


@dataclass(slots=True)
class MentionSettings:
    """Configuration for @-mention file/directory context injection."""

    max_file_size: int = 1_048_576  # 1 MB
    max_files: int = 100
    max_total_size: int = 10_485_760  # 10 MB
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
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        if not isinstance(raw, dict):
            raise ValueError(_("Expected mapping for Settings, got {type_name}", type_name=type(raw).__name__))
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(_("Unknown setting keys: {keys}", keys=sorted(unknown)))
        kwargs: dict[str, Any] = {}
        if "model" in raw:
            kwargs["model"] = _dataclass_from_dict(ModelSettings, raw["model"])
        if "runtime" in raw:
            kwargs["runtime"] = _dataclass_from_dict(RuntimeSettings, raw["runtime"])
        if "rag" in raw:
            kwargs["rag"] = _dataclass_from_dict(RAGSettings, raw["rag"])
        if "mentions" in raw:
            kwargs["mentions"] = _dataclass_from_dict(MentionSettings, raw["mentions"])
        if "subagents" in raw:
            kwargs["subagents"] = _subagents_from_list(raw["subagents"])
        if "langsmith" in raw:
            kwargs["langsmith"] = _dataclass_from_dict(LangSmithSettings, raw["langsmith"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"] = {k: v for k, v in d["model"].items() if v is not None}
        ls = d["langsmith"]
        if not any(ls.values()):
            d.pop("langsmith")
        return d

    def setup_environment(self) -> None:
        """Inject settings into the environment variables."""
        if self.langsmith.api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith.api_key
        if self.langsmith.tracing:
            os.environ["LANGSMITH_TRACING"] = "true"
        if self.langsmith.project:
            os.environ["LANGSMITH_PROJECT"] = self.langsmith.project
        if self.langsmith.endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = self.langsmith.endpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], raw: Any) -> T:
    if not isinstance(raw, dict):
        raise ValueError(
            _("Expected mapping for '{name}', got {type_name}", name=cls.__name__, type_name=type(raw).__name__)
        )
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(_("Unknown setting keys: {keys}", keys=sorted(unknown)))
    return cls(**raw)


def _subagents_from_list(
    raw: Any,
) -> list[SubAgentSettings]:
    """Parse subagent list from YAML, handling nested mcp_servers."""
    if not isinstance(raw, list):
        raise ValueError(_("Expected list for subagents, got {type_name}", type_name=type(raw).__name__))
    subagents: list[SubAgentSettings] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(_("Expected mapping for subagent, got {type_name}", type_name=type(item).__name__))
        data = dict(item)
        if "mcp_servers" in item:
            mcp_servers_raw = item["mcp_servers"]
            if not isinstance(mcp_servers_raw, list):
                raise ValueError(
                    _("Expected list for mcp_servers, got {type_name}", type_name=type(mcp_servers_raw).__name__)
                )
            data["mcp_servers"] = [_dataclass_from_dict(SubAgentMCPServer, m) for m in mcp_servers_raw]
        subagents.append(_dataclass_from_dict(SubAgentSettings, data))
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

    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(_("Settings file must contain a YAML mapping: {path}", path=settings_path))
    return Settings.from_dict(raw)


def save_settings(settings: Settings, settings_path: Path = SETTINGS_PATH) -> None:
    """Save settings to YAML file atomically."""
    text = yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True)
    atomic_write_text(settings_path, text)


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


def render_prompt_template(template_str: str, context: dict[str, Any]) -> str:
    """Render a Jinja2 template string with the provided context dictionary."""
    env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_str)
    return template.render(context)


def load_instructions(instructions_path: Path = INSTRUCTIONS_PATH) -> str:
    """Load agent instructions from file or create it from bundled defaults."""
    if not instructions_path.exists():
        default_text = (
            resources.files(__package__).joinpath("prompts/default_instructions.md").read_text(encoding="utf-8").strip()
        )
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        instructions_path.write_text(default_text + "\n", encoding="utf-8")
        return default_text
    return instructions_path.read_text(encoding="utf-8").strip()


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

VALID_RESET_OPTIONS = {"all", "config-file", "system-prompt"}


def reset_config(
    option: str,
    *,
    settings_path: Path = SETTINGS_PATH,
    instructions_path: Path = INSTRUCTIONS_PATH,
) -> list[str]:
    """Reset configuration or system prompt to defaults."""
    if option not in VALID_RESET_OPTIONS:
        raise ValueError(
            _(
                "Invalid reset option '{option}'. Expected one of: {valid}",
                option=option,
                valid=sorted(VALID_RESET_OPTIONS),
            )
        )

    messages: list[str] = []

    if option in ("all", "config-file"):
        save_settings(Settings(), settings_path)
        messages.append(_("Reset: Restored default configuration at {path}", path=settings_path))

    if option in ("all", "system-prompt"):
        instructions_path.unlink(missing_ok=True)
        load_instructions(instructions_path)
        messages.append(_("Reset: Restored default system prompt at {path}", path=instructions_path))

    return messages
