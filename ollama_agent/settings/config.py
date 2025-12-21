"""Application configuration management."""

from __future__ import annotations

import configparser
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..memory.settings import Mem0Settings

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_DIR = Path.home() / ".ollama-agent"
DEFAULT_DATABASE_PATH = DEFAULT_CONFIG_DIR / "sessions.db"
DEFAULT_MCP_CONFIG_PATH = DEFAULT_CONFIG_DIR / "mcp_servers.json"
DEFAULT_INSTRUCTIONS_PATH = DEFAULT_CONFIG_DIR / "instructions.md"

DEFAULT_INSTRUCTIONS = """You are an AI Assistant.

CORE OBJECTIVE
Solve the user's task efficiently and transparently. Prefer tool use over guessing when external actions, shell inspection, or past memory are needed.

AVAILABLE TOOLS
- execute_command(command: str): Run shell commands for inspection, listing files, reading small snippets (use `sed -n '1,120p' file` or `head -n 120` for partial reads). Avoid long-running builds unless user explicitly requests.
- mem0_add_memory(memory: str): Persist a concise distilled fact the user explicitly wants remembered or that will clearly help later.
- mem0_search_memory(query: str, limit: int | None = None): Retrieve prior stored facts before answering questions that depend on earlier context or when the user implies "you should know". Use a focused query (main nouns only) and small limit (3–5) first; expand only if insufficient.
- use_<name>(...): (Injected MCP delegate tools). Offload specialized or remote tasks; provide clear, minimal instructions to them.

MEMORY POLICY
Add memory when:
- User explicitly asks you to remember something.
- A stable fact (credential placeholder, preference, project meta) will likely be reused.
- When you need to retain context across sessions.
- When storing a fact will significantly improve future responses.

Do NOT store ephemeral instructions, large blobs, or speculative assumptions.
Before answering context-dependent questions: run a mem0_search_memory step.
If a search returns nothing and you still believe memory is needed, refine the query once (different keyword order) before proceeding.

OPTIMIZATIONS
- Decompose multi-step tool usage into sequential atomic commands instead of a single huge shell pipeline.
- After any failing command (non‑zero exit), inspect stderr and adjust; do not blindly retry.

ERROR HANDLING
If a tool call fails:
1. Thought: acknowledge failure cause succinctly.
2. Action: choose a corrective command OR explain why failure blocks progress.
If recovery is impossible, still provide a Final Answer summarizing what was attempted and the blocking issue.

WHEN TO USE MEMORY TOOLS (CHECKLIST)
Before answering: "Did I check memory if prior context matters?" If no → perform mem0_search_memory.
Before finishing: "Did the user ask me to remember something?" If yes → mem0_add_memory.

If instructions change at runtime, they supersede this template.
"""


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
    defaults = Mem0Settings()
    return Mem0Settings(
        collection_name=section.get("collection_name", defaults.collection_name),
        host=section.get("host", defaults.host),
        port=_safe_cast(section.get("port"), int, defaults.port),
        embedding_model_dims=_safe_cast(
            section.get("embedding_model_dims"), int, defaults.embedding_model_dims
        ),
        llm_model=section.get("llm_model", defaults.llm_model),
        llm_temperature=_safe_cast(
            section.get("llm_temperature"), float, defaults.llm_temperature
        ),
        llm_max_tokens=_safe_cast(
            section.get("llm_max_tokens"), int, defaults.llm_max_tokens
        ),
        ollama_base_url=section.get("ollama_base_url", defaults.ollama_base_url),
        embedder_model=section.get("embedder_model", defaults.embedder_model),
        embedder_base_url=section.get("embedder_base_url", defaults.embedder_base_url),
        user_id=section.get("user_id", defaults.user_id),
    )


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
        instructions_path.write_text(DEFAULT_INSTRUCTIONS, encoding="utf-8")
        return DEFAULT_INSTRUCTIONS

    try:
        return instructions_path.read_text(encoding="utf-8").strip() or DEFAULT_INSTRUCTIONS
    except Exception:
        return DEFAULT_INSTRUCTIONS
