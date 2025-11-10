"""Application configuration management."""

from __future__ import annotations

import configparser
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_DIR = Path.home() / ".ollama-agent"
DEFAULT_DATABASE_PATH = DEFAULT_CONFIG_DIR / "sessions.db"
DEFAULT_MCP_CONFIG_PATH = DEFAULT_CONFIG_DIR / "mcp_servers.json"
DEFAULT_INSTRUCTIONS_PATH = DEFAULT_CONFIG_DIR / "instructions.md"

# Default agent instructions
DEFAULT_INSTRUCTIONS = """You are an AI Assistant.

CORE OBJECTIVE
Solve the user's task efficiently and transparently. Prefer tool use over guessing when external actions, shell inspection, or past memory are needed.

AVAILABLE TOOLS
- execute_command(command: str): Run shell commands for inspection, listing files, reading small snippets (use `sed -n '1,120p' file` or `head -n 120` for partial reads). Avoid long-running builds unless user explicitly requests.
- mem0_add_memory(memory: str): Persist a concise distilled fact the user explicitly wants remembered or that will clearly help later.
- mem0_search_memory(query: str, limit: int | None = None): Retrieve prior stored facts before answering questions that depend on earlier context or when the user implies “you should know”. Use a focused query (main nouns only) and small limit (3–5) first; expand only if insufficient.
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
Before answering: “Did I check memory if prior context matters?” If no → perform mem0_search_memory.
Before finishing: “Did the user ask me to remember something?” If yes → mem0_add_memory.

If instructions change at runtime, they supersede this template.
"""


@dataclass(eq=True)
class Mem0Settings:
    collection_name: str = "ollama-agent"
    host: str = "localhost"
    port: int = 63333
    embedding_model_dims: int = 768
    llm_model: str = "llama3.1:latest"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    ollama_base_url: str = "http://localhost:11434"
    embedder_model: str = "nomic-embed-text:latest"
    embedder_base_url: str = "http://localhost:11434"
    user_id: str = "default"


@dataclass
class Config:
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: str = "medium"
    database_path: Path = field(default_factory=lambda: DEFAULT_DATABASE_PATH)
    builtin_tool_timeout: int = 30
    mcp_config_path: Path = field(default_factory=lambda: DEFAULT_MCP_CONFIG_PATH)
    mem0: Mem0Settings = field(default_factory=Mem0Settings)


def _coerce(value: str | None, cast, default, label: str):
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%s, using %s", label, value, default)
        return default


def _write_default_config(path: Path, defaults: Config) -> None:
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

    mem0_defaults = {k: str(v) for k, v in asdict(defaults.mem0).items()}
    parser["mem0"] = mem0_defaults

    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)


def _load_mem0(parser: configparser.ConfigParser) -> Mem0Settings:
    base_defaults = Mem0Settings()
    defaults = asdict(base_defaults)
    if parser.has_section("mem0"):
        defaults.update({k: v for k, v in parser.items("mem0")})
        if "enabled" in defaults and defaults.get("enabled") not in {True, "true", "True", "1"}:
            logger.warning("mem0.enabled is no longer supported; Mem0 is always enabled")

    return Mem0Settings(
        collection_name=str(defaults["collection_name"]),
        host=str(defaults["host"]),
        port=_coerce(defaults.get("port"), int, base_defaults.port, "mem0.port"),
        embedding_model_dims=_coerce(
            defaults.get("embedding_model_dims"),
            int,
            base_defaults.embedding_model_dims,
            "mem0.embedding_model_dims",
        ),
        llm_model=str(defaults["llm_model"]),
        llm_temperature=_coerce(
            defaults.get("llm_temperature"),
            float,
            base_defaults.llm_temperature,
            "mem0.llm_temperature",
        ),
        llm_max_tokens=_coerce(
            defaults.get("llm_max_tokens"),
            int,
            base_defaults.llm_max_tokens,
            "mem0.llm_max_tokens",
        ),
        ollama_base_url=str(defaults["ollama_base_url"]),
        embedder_model=str(defaults["embedder_model"]),
        embedder_base_url=str(defaults["embedder_base_url"]),
        user_id=str(defaults["user_id"]),
    )


def get_config(config_dir: Path | None = None) -> Config:
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    config_path = config_dir / "config.ini"
    config_dir.mkdir(parents=True, exist_ok=True)

    defaults = Config()
    if not config_path.exists():
        _write_default_config(config_path, defaults)
        return defaults

    parser = configparser.ConfigParser()
    parser.read(config_path)

    mem0 = _load_mem0(parser)

    get_default = parser.defaults().get
    section = parser["default"] if parser.has_section("default") else {}

    def _get(option: str, fallback: str) -> str:
        return section.get(option, get_default(option, fallback))

    return Config(
        model=_get("model", defaults.model),
        base_url=_get("base_url", defaults.base_url),
        api_key=_get("api_key", defaults.api_key),
        reasoning_effort=_get("reasoning_effort", defaults.reasoning_effort),
        database_path=Path(_get("database_path", str(defaults.database_path))),
        builtin_tool_timeout=_coerce(
            _get("builtin_tool_timeout", str(defaults.builtin_tool_timeout)),
            int,
            defaults.builtin_tool_timeout,
            "default.builtin_tool_timeout",
        ),
        mcp_config_path=Path(_get("mcp_config_path", str(defaults.mcp_config_path))),
        mem0=mem0,
    )


def load_instructions(instructions_path: Path = DEFAULT_INSTRUCTIONS_PATH) -> str:
    if not instructions_path.exists():
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        instructions_path.write_text(DEFAULT_INSTRUCTIONS, encoding="utf-8")
        logger.info("Created instructions file at %s", instructions_path)
        return DEFAULT_INSTRUCTIONS

    try:
        content = instructions_path.read_text(encoding="utf-8").strip()
        return content or DEFAULT_INSTRUCTIONS
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error reading instructions %s: %s", instructions_path, exc)
        return DEFAULT_INSTRUCTIONS
