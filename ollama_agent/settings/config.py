"""Application configuration management."""

from __future__ import annotations

import configparser
import logging
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from ..memory.settings import Mem0Settings
from ..rag.settings import RAGSettings

from .paths import APP_DIR, DATABASE_PATH, INSTRUCTIONS_PATH, MCP_SERVERS_PATH

logger = logging.getLogger(__name__)


def _default_instructions() -> str:
    try:
        return resources.files(__package__).joinpath("default_instructions.md").read_text(encoding="utf-8").strip()
    except Exception:
        return "You are an AI Assistant."


@dataclass
class Config:
    """Application configuration."""
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434"
    api_key: str = "ollama"
    reasoning_effort: str = "medium"
    context_window: int | None = None
    database_path: Path = field(default_factory=lambda: DATABASE_PATH)
    builtin_tool_timeout: int = 30
    mcp_config_path: Path = field(default_factory=lambda: MCP_SERVERS_PATH)
    mem0: Mem0Settings = field(default_factory=Mem0Settings)
    rag: RAGSettings = field(default_factory=RAGSettings)


def _safe_cast(value: Any, caster: type, default: Any) -> Any:
    if value is None:
        return default
    try:
        return caster(value)
    except (TypeError, ValueError):
        return default


def _load_mem0(section: dict[str, str]) -> Mem0Settings:
    d = Mem0Settings()
    g = section.get
    c = _safe_cast
    return Mem0Settings(
        collection_name=g("collection_name", d.collection_name),
        qdrant_path=g("qdrant_path", d.qdrant_path),
        embedding_model_dims=c(g("embedding_model_dims"), int, d.embedding_model_dims),
        llm_model=g("llm_model", d.llm_model),
        llm_temperature=c(g("llm_temperature"), float, d.llm_temperature),
        llm_max_tokens=c(g("llm_max_tokens"), int, d.llm_max_tokens),
        ollama_base_url=g("ollama_base_url", d.ollama_base_url),
        embedder_model=g("embedder_model", d.embedder_model),
        embedder_base_url=g("embedder_base_url", d.embedder_base_url),
        user_id=g("user_id", d.user_id),
    )


def _load_rag(section: dict[str, str]) -> RAGSettings:
    d, g, c = RAGSettings(), section.get, _safe_cast
    return RAGSettings(
        rag_dir=g("rag_dir", d.rag_dir),
        embedder_model=g("embedder_model", d.embedder_model),
        embedder_base_url=g("embedder_base_url", d.embedder_base_url),
        embedding_dims=c(g("embedding_dims"), int, d.embedding_dims),
        default_top_k=c(g("default_top_k"), int, d.default_top_k),
        chunk_size=c(g("chunk_size"), int, d.chunk_size),
        chunk_overlap=c(g("chunk_overlap"), int, d.chunk_overlap),
    )


def get_config(config_dir: Path | None = None) -> Config:
    """Load configuration from file or create defaults."""
    config_dir = config_dir or APP_DIR
    config_path = config_dir / "config.ini"
    config_dir.mkdir(parents=True, exist_ok=True)
    defaults = Config()

    if not config_path.exists():
        parser = configparser.ConfigParser()
        parser["default"] = {
            "model": defaults.model,
            "base_url": defaults.base_url,
            "api_key": defaults.api_key,
            "reasoning_effort": defaults.reasoning_effort,
            "context_window": "" if defaults.context_window is None else str(defaults.context_window),
            "database_path": str(defaults.database_path),
            "builtin_tool_timeout": str(defaults.builtin_tool_timeout),
            "mcp_config_path": str(defaults.mcp_config_path),
        }
        parser["mem0"] = {k: str(v) for k, v in asdict(
            defaults.mem0).items() if not k.startswith("_")}
        parser["rag"] = {k: str(v) for k, v in asdict(defaults.rag).items()}
        with config_path.open("w", encoding="utf-8") as f:
            parser.write(f)
        return defaults

    parser = configparser.ConfigParser()
    parser.read(config_path)
    section = dict(parser["default"]) if parser.has_section("default") else {}
    mem0_section = dict(parser["mem0"]) if parser.has_section("mem0") else {}
    rag_section = dict(parser["rag"]) if parser.has_section("rag") else {}

    base_url = section.get("base_url", defaults.base_url).rstrip("/")
    if base_url.endswith("/v1"):
        raise ValueError(
            f"base_url '{base_url}' contains an '/v1' path from the old OpenAI-compatible "
            "configuration. Update base_url to the native Ollama host "
            f"(e.g. 'http://localhost:11434') in {config_path}."
        )

    return Config(
        model=section.get("model", defaults.model),
        base_url=base_url,
        api_key=section.get("api_key", defaults.api_key),
        reasoning_effort=section.get("reasoning_effort", defaults.reasoning_effort),
        context_window=_safe_cast(section.get("context_window"), int, defaults.context_window),
        database_path=Path(section.get("database_path", str(defaults.database_path))),
        builtin_tool_timeout=_safe_cast(section.get("builtin_tool_timeout"), int, defaults.builtin_tool_timeout),
        mcp_config_path=Path(section.get("mcp_config_path", str(defaults.mcp_config_path))),
        mem0=_load_mem0(mem0_section),
        rag=_load_rag(rag_section),
    )


def load_instructions(instructions_path: Path = INSTRUCTIONS_PATH) -> str:
    """Load agent instructions from file or return defaults."""
    if not instructions_path.exists():
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        instructions_path.write_text(
            _default_instructions() + "\n", encoding="utf-8")
        return _default_instructions()
    try:
        return instructions_path.read_text(encoding="utf-8").strip() or _default_instructions()
    except Exception:
        return _default_instructions()


def reset_config(option: str) -> None:
    """Reset configuration or system prompt to defaults."""
    if option in ("all", "config-file"):
        config_path = APP_DIR / "config.ini"
        if config_path.exists():
            config_path.unlink()
        
        get_config()
        print(f"Reset: Restored default configuration at {config_path}")

    if option in ("all", "system-prompt"):
        if INSTRUCTIONS_PATH.exists():
            INSTRUCTIONS_PATH.unlink()
        
        load_instructions()
        print(f"Reset: Restored default system prompt at {INSTRUCTIONS_PATH}")

