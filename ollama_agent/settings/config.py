"""Application configuration management using YAML-based settings."""

from __future__ import annotations

import os
import weakref
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Self, TypeVar

import yaml  # type: ignore[import-untyped]
from pydantic import (
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    with_config,
)

from ..core import (
    DEFAULT_REASONING_EFFORT,
    atomic_write_text,
)
from ..i18n import _
from .paths import (
    INSTRUCTIONS_PATH,
    MEMORY_PATH,
    RAG_DIR,
    SETTINGS_PATH,
)

_FORBID_EXTRA = ConfigDict(extra="forbid")
_task_model_overrides: dict[int, tuple[weakref.ref[ModelSettings], str, str, str, str]] = {}

# ---------------------------------------------------------------------------
# Settings dataclasses (CUD-inspired)
# ---------------------------------------------------------------------------


@with_config(_FORBID_EXTRA)
@dataclass(slots=True, weakref_slot=True)
class ModelSettings:
    name: StrictStr = ""
    base_url: StrictStr = "http://localhost:11434"
    temperature: StrictFloat | StrictInt | None = None
    top_p: StrictFloat | StrictInt | None = None
    top_k: StrictInt | None = None
    min_p: StrictFloat | StrictInt | None = None
    presence_penalty: StrictFloat | StrictInt | None = None
    repeat_penalty: StrictFloat | StrictInt | None = None
    context_window: StrictInt | StrictStr = 10000
    reasoning_effort: StrictStr = DEFAULT_REASONING_EFFORT


@with_config(_FORBID_EXTRA)
@dataclass(slots=True)
class RuntimeSettings:
    allow_traversal: StrictBool = False
    builtin_tool_timeout: StrictInt = 30
    collapse_thinking: StrictBool = True
    inherit_env: StrictBool = True
    language: StrictStr = ""


@with_config(_FORBID_EXTRA)
@dataclass(slots=True)
class RAGSettings:
    rag_dir: StrictStr = str(RAG_DIR)
    embedder_model: StrictStr = "embeddinggemma-2:740m"
    embedder_base_url: StrictStr = "http://localhost:11434"
    embedding_dims: StrictInt = 768
    default_top_k: StrictInt = 5
    chunk_size: StrictInt = 500
    chunk_overlap: StrictInt = 50

    def __post_init__(self) -> None:
        if not self.rag_dir.strip():
            raise ValueError(_("Setting 'rag.rag_dir' must be a non-empty string"))


@with_config(_FORBID_EXTRA)
@dataclass(slots=True)
class SubAgentMCPServer:
    """An MCP server attached to a subagent."""

    name: StrictStr = ""
    command: StrictStr = ""
    args: list[StrictStr] = field(default_factory=list)
    env: dict[StrictStr, StrictStr] = field(default_factory=dict)

    def to_connection_config(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "args": self.args,
            "env": self.env,
        }


@with_config(_FORBID_EXTRA)
@dataclass(slots=True)
class SubAgentSettings:
    """A custom subagent with its own model, skills, and MCP servers."""

    name: StrictStr = ""
    description: StrictStr = ""
    system_prompt: StrictStr = ""
    model: StrictStr = ""
    context_window: StrictInt | StrictStr = 0
    mcp_servers: list[SubAgentMCPServer] = field(default_factory=list)


@with_config(_FORBID_EXTRA)
@dataclass(slots=True)
class LangSmithSettings:
    api_key: StrictStr = ""
    tracing: StrictBool = False
    project: StrictStr = ""
    endpoint: StrictStr = ""


@with_config(_FORBID_EXTRA)
@dataclass(slots=True)
class MentionSettings:
    """Configuration for @-mention file/directory context injection."""

    max_file_size: StrictInt = 1_048_576  # 1 MB
    max_files: StrictInt = 100
    max_total_size: StrictInt = 10_485_760  # 10 MB
    max_completions: StrictInt = 200


@with_config(_FORBID_EXTRA)
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
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        key = id(self.model)
        if key in _task_model_overrides:
            ref, orig_name, orig_effort, applied_name, applied_effort = _task_model_overrides[key]
            if ref() is self.model:
                if self.model.name == applied_name:
                    d["model"]["name"] = orig_name
                if self.model.reasoning_effort == applied_effort:
                    d["model"]["reasoning_effort"] = orig_effort
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
    try:
        return TypeAdapter(cls).validate_python(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


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
