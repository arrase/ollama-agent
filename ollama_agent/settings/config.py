"""Application configuration management using YAML-based settings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from importlib import resources
from pathlib import Path
from types import UnionType
from typing import Any, Self, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml  # type: ignore[import-untyped]

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
    reasoning_effort: str = DEFAULT_REASONING_EFFORT


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
    embedder_model: str = "embeddinggemma-2:740m"
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

    def to_connection_config(self) -> dict[str, Any]:
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
        return _dataclass_from_dict(cls, raw)

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


def _coerce_setting(name: str, value: Any, hint: Any) -> Any:
    """Validate one untrusted YAML value against its dataclass field annotation.

    Without this, ``context_window: true`` reaches Ollama as ``num_ctx=1`` and
    ``temperature: "hot"`` surfaces much later as a pydantic error from deep inside
    the model constructor.
    """
    origin = get_origin(hint)
    if origin is Union or origin is UnionType:
        return _coerce_union(name, value, hint)
    return _coerce_value(value, hint)


def _coerce_union(name: str, value: Any, hint: Any) -> Any:
    """Validate against each union member in turn; a union is accepted if any member validates."""
    errors: list[str] = []
    for arg in get_args(hint):
        if arg is type(None):
            if value is None:
                return None
            continue
        try:
            return _coerce_value(value, arg)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(_("Invalid value for '{name}': {detail}", name=name, detail="; ".join(errors) or _("wrong type")))


def _coerce_value(value: Any, hint: Any) -> Any:
    origin = get_origin(hint) or hint
    if is_dataclass(origin):
        return _dataclass_from_dict(cast(type[Any], origin), value)
    if origin is bool:
        if not isinstance(value, bool):
            raise ValueError(_("expected a boolean, got {t}", t=type(value).__name__))
        return value
    if origin is int:
        # bool is a subclass of int; context_window: true must not become num_ctx=1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(_("expected an integer, got {t}", t=type(value).__name__))
        return value
    if origin is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(_("expected a number, got {t}", t=type(value).__name__))
        return float(value)
    if origin is str:
        if not isinstance(value, str):
            raise ValueError(_("expected a string, got {t}", t=type(value).__name__))
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"expected a list, got {type(value).__name__}")
        (item_hint,) = get_args(hint)
        return [_coerce_value(item, item_hint) for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"expected a dict, got {type(value).__name__}")
        key_hint, val_hint = get_args(hint)
        return {_coerce_value(k, key_hint): _coerce_value(v, val_hint) for k, v in value.items()}
    return value


def _dataclass_from_dict(cls: type[T], raw: Any) -> T:
    if not isinstance(raw, dict):
        raise ValueError(
            _("Expected mapping for '{name}', got {type_name}", name=cls.__name__, type_name=type(raw).__name__)
        )
    dataclass_fields = fields(cast(Any, cls))
    valid = {f.name for f in dataclass_fields}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(_("Unknown setting keys: {keys}", keys=sorted(unknown)))

    hints = get_type_hints(cast(Any, cls))
    kwargs: dict[str, Any] = {}
    for f in dataclass_fields:
        if f.name not in raw:
            continue
        try:
            kwargs[f.name] = _coerce_setting(f.name, raw[f.name], hints[f.name])
        except ValueError as exc:
            raise ValueError(_("Invalid value for '{name}.{key}': {exc}", name=cls.__name__, key=f.name, exc=exc))
    return cls(**kwargs)


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
