"""Application configuration management."""

from __future__ import annotations

import configparser
import logging
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from ..memory.settings import Mem0Settings

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_DIR = Path.home() / ".ollama-agent"
DEFAULT_DATABASE_PATH = DEFAULT_CONFIG_DIR / "sessions.db"
DEFAULT_MCP_CONFIG_PATH = DEFAULT_CONFIG_DIR / "mcp_servers.json"
DEFAULT_INSTRUCTIONS_PATH = DEFAULT_CONFIG_DIR / "instructions.md"


def _default_instructions() -> str:
    """Return built-in default instructions shipped with the package."""
    try:
        return (
            resources.files(__package__).joinpath("default_instructions.md").read_text(encoding="utf-8").strip()
        )
    except Exception:
        return "You are an AI Assistant."


# Backwards-compatible alias (previously a large inline constant).
DEFAULT_INSTRUCTIONS = _default_instructions()


@dataclass
class Config:
    """Application configuration."""

    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: str = "medium"
    database_path: Path = field(default_factory=lambda: DEFAULT_DATABASE_PATH)
    builtin_tool_timeout: int = 30
    mcp_config_path: Path = field(default_factory=lambda: DEFAULT_MCP_CONFIG_PATH)
    mem0: Mem0Settings = field(default_factory=Mem0Settings)


def _safe_cast(value: Any, caster: type, default: Any) -> Any:
    """Safely cast a value with fallback to default."""
    if value is None:
        return default
    try:
        return caster(value)
    except (TypeError, ValueError):
        return default


def _write_default_config(path: Path, defaults: Config) -> None:
    """Write default configuration to file."""
    parser = configparser.ConfigParser()
    parser["default"] = {
        "model": defaults.model,
        "base_url": defaults.base_url,
        "api_key": defaults.api_key,
        "reasoning_effort": defaults.reasoning_effort,
        "database_path": str(defaults.database_path),
        "builtin_tool_timeout": str(defaults.builtin_tool_timeout),
        "mcp_config_path": str(defaults.mcp_config_path),
    }
    parser["mem0"] = {k: str(v) for k, v in asdict(defaults.mem0).items()}

    with path.open("w", encoding="utf-8") as f:
        parser.write(f)


def _load_mem0(section: dict[str, str]) -> Mem0Settings:
    """Load Mem0 settings from config section."""
    d = Mem0Settings()
    g, c = section.get, _safe_cast
    return Mem0Settings(
        collection_name=g("collection_name", d.collection_name), host=g("host", d.host),
        port=c(g("port"), int, d.port), embedding_model_dims=c(g("embedding_model_dims"), int, d.embedding_model_dims),
        llm_model=g("llm_model", d.llm_model), llm_temperature=c(g("llm_temperature"), float, d.llm_temperature),
        llm_max_tokens=c(g("llm_max_tokens"), int, d.llm_max_tokens), ollama_base_url=g("ollama_base_url", d.ollama_base_url),
        embedder_model=g("embedder_model", d.embedder_model), embedder_base_url=g("embedder_base_url", d.embedder_base_url),
        user_id=g("user_id", d.user_id))


def get_config(config_dir: Path | None = None) -> Config:
    """Load configuration from file or create defaults."""
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    config_path = config_dir / "config.ini"
    config_dir.mkdir(parents=True, exist_ok=True)

    defaults = Config()
    if not config_path.exists():
        _write_default_config(config_path, defaults)
        return defaults

    parser = configparser.ConfigParser()
    parser.read(config_path)

    section = dict(parser["default"]) if parser.has_section("default") else {}
    mem0_section = dict(parser["mem0"]) if parser.has_section("mem0") else {}

    return Config(
        model=section.get("model", defaults.model),
        base_url=section.get("base_url", defaults.base_url),
        api_key=section.get("api_key", defaults.api_key),
        reasoning_effort=section.get("reasoning_effort", defaults.reasoning_effort),
        database_path=Path(section.get("database_path", str(defaults.database_path))),
        builtin_tool_timeout=_safe_cast(
            section.get("builtin_tool_timeout"), int, defaults.builtin_tool_timeout
        ),
        mcp_config_path=Path(
            section.get("mcp_config_path", str(defaults.mcp_config_path))
        ),
        mem0=_load_mem0(mem0_section),
    )


def load_instructions(instructions_path: Path = DEFAULT_INSTRUCTIONS_PATH) -> str:
    """Load agent instructions from file or return defaults."""
    if not instructions_path.exists():
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        defaults = _default_instructions()
        instructions_path.write_text(defaults + "\n", encoding="utf-8")
        return defaults

    try:
        return instructions_path.read_text(encoding="utf-8").strip() or _default_instructions()
    except Exception:
        return _default_instructions()
