"""Application configuration management."""

import configparser
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_DIR = Path.home() / ".ollama-agent"
DEFAULT_DATABASE_PATH = DEFAULT_CONFIG_DIR / "sessions.db"
DEFAULT_MCP_CONFIG_PATH = DEFAULT_CONFIG_DIR / "mcp_servers.json"
DEFAULT_INSTRUCTIONS_PATH = DEFAULT_CONFIG_DIR / "instructions.md"

# Default agent instructions
DEFAULT_INSTRUCTIONS = """As an expert assistant your primary goal is to solve user tasks, using the available tools if needed.

When using tools you must strictly follow the Thought, Action (function call), and Observation (tool result) sequence until you have a Final Answer.

Thought:
[your reasoning here]

Action:
[function call here]

Observation:
[tool result here]
... (repeat Thought, Action, Observation)

Thought:
[your reasoning here]

Final Answer:
[your final answer here]

Memory management:
- When the user explicitly asks you to remember something or you encounter information that could be useful later, call `mem0_add_memory(memory: str)` with a concise summary before finalizing your reply.
- Before answering questions that might rely on prior context, when the user hints you should already know something, or whenever you believe past context would help, call `mem0_search_memory(query: str, limit: int | None = None)` to retrieve relevant memories.
- When using memory tools, continue the Thought → Action → Observation loop until you have the information you need.
"""


@dataclass(eq=True)
class Mem0Settings:
    """Configuration for Mem0-backed persistent memory."""

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
    """Application configuration with defaults."""
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: str = "medium"
    database_path: Path = field(default_factory=lambda: DEFAULT_DATABASE_PATH)
    builtin_tool_timeout: int = 30
    mcp_config_path: Path = field(
        default_factory=lambda: DEFAULT_MCP_CONFIG_PATH)
    mem0: Mem0Settings = field(default_factory=Mem0Settings)


def get_config(config_dir: Path | None = None) -> Config:
    """
    Load configuration from file or create with defaults.

    Args:
        config_dir: Directory for config file. Defaults to ~/.ollama-agent

    Returns:
        Config object with values from file or defaults.
    """
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    config_file = config_dir / "config.ini"
    config_dir.mkdir(parents=True, exist_ok=True)

    defaults = Config()

    if not config_file.exists():
        # Create default config file
        parser = configparser.ConfigParser()
        parser.add_section("default")
        parser.set("default", "model", defaults.model)
        parser.set("default", "base_url", defaults.base_url)
        parser.set("default", "api_key", defaults.api_key)
        parser.set("default", "reasoning_effort", defaults.reasoning_effort)
        parser.set("default", "database_path", str(defaults.database_path))
        parser.set("default", "builtin_tool_timeout",
                   str(defaults.builtin_tool_timeout))
        parser.set("default", "mcp_config_path", str(defaults.mcp_config_path))

        parser.add_section("mem0")
        parser.set("mem0", "collection_name", defaults.mem0.collection_name)
        parser.set("mem0", "host", defaults.mem0.host)
        parser.set("mem0", "port", str(defaults.mem0.port))
        parser.set("mem0", "embedding_model_dims",
                   str(defaults.mem0.embedding_model_dims))
        parser.set("mem0", "llm_model", defaults.mem0.llm_model)
        parser.set("mem0", "llm_temperature",
                   str(defaults.mem0.llm_temperature))
        parser.set("mem0", "llm_max_tokens",
                   str(defaults.mem0.llm_max_tokens))
        parser.set("mem0", "ollama_base_url", defaults.mem0.ollama_base_url)
        parser.set("mem0", "embedder_model", defaults.mem0.embedder_model)
        parser.set("mem0", "embedder_base_url",
                   defaults.mem0.embedder_base_url)
        parser.set("mem0", "user_id", defaults.mem0.user_id)

        with open(config_file, 'w', encoding='utf-8') as f:
            parser.write(f)

        return defaults

    # Load from existing file
    parser = configparser.ConfigParser()
    parser.read(config_file)

    def _getint(section: str, option: str, fallback: int) -> int:
        try:
            return parser.getint(section, option, fallback=fallback)
        except ValueError:
            logger.warning(
                "Invalid integer for %s.%s, using fallback %s",
                section,
                option,
                fallback,
            )
            return fallback

    def _getfloat(section: str, option: str, fallback: float) -> float:
        try:
            return parser.getfloat(section, option, fallback=fallback)
        except ValueError:
            logger.warning(
                "Invalid float for %s.%s, using fallback %s",
                section,
                option,
                fallback,
            )
            return fallback

    mem0_defaults = Mem0Settings()
    mem0_section = "mem0"
    if parser.has_section(mem0_section):
        if parser.has_option(mem0_section, "enabled") and not parser.getboolean(mem0_section, "enabled", fallback=True):
            logger.warning("mem0.enabled is no longer supported; Mem0 is always enabled")
        mem0 = Mem0Settings(
            collection_name=parser.get(
                mem0_section,
                "collection_name",
                fallback=mem0_defaults.collection_name,
            ),
            host=parser.get(mem0_section, "host", fallback=mem0_defaults.host),
            port=_getint(mem0_section, "port", mem0_defaults.port),
            embedding_model_dims=_getint(
                mem0_section,
                "embedding_model_dims",
                mem0_defaults.embedding_model_dims,
            ),
            llm_model=parser.get(
                mem0_section,
                "llm_model",
                fallback=mem0_defaults.llm_model,
            ),
            llm_temperature=_getfloat(
                mem0_section,
                "llm_temperature",
                mem0_defaults.llm_temperature,
            ),
            llm_max_tokens=_getint(
                mem0_section,
                "llm_max_tokens",
                mem0_defaults.llm_max_tokens,
            ),
            ollama_base_url=parser.get(
                mem0_section,
                "ollama_base_url",
                fallback=mem0_defaults.ollama_base_url,
            ),
            embedder_model=parser.get(
                mem0_section,
                "embedder_model",
                fallback=mem0_defaults.embedder_model,
            ),
            embedder_base_url=parser.get(
                mem0_section,
                "embedder_base_url",
                fallback=mem0_defaults.embedder_base_url,
            ),
            user_id=parser.get(
                mem0_section,
                "user_id",
                fallback=mem0_defaults.user_id,
            ),
        )
    else:
        mem0 = mem0_defaults

    config = Config(
        model=parser.get("default", "model", fallback=defaults.model),
        base_url=parser.get("default", "base_url", fallback=defaults.base_url),
        api_key=parser.get("default", "api_key", fallback=defaults.api_key),
        reasoning_effort=parser.get(
            "default", "reasoning_effort", fallback=defaults.reasoning_effort),
        database_path=Path(parser.get(
            "default", "database_path", fallback=str(defaults.database_path))),
        builtin_tool_timeout=int(parser.get(
            "default", "builtin_tool_timeout", fallback=str(defaults.builtin_tool_timeout))),
        mcp_config_path=Path(parser.get(
            "default", "mcp_config_path", fallback=str(defaults.mcp_config_path))),
        mem0=mem0,
    )

    return config


def load_instructions(instructions_path: Path = DEFAULT_INSTRUCTIONS_PATH) -> str:
    """
    Load agent instructions from file, creating it with defaults if needed.

    Args:
        instructions_path: Path to the instructions file.

    Returns:
        Instructions content as string.
    """
    if not instructions_path.exists():
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        instructions_path.write_text(DEFAULT_INSTRUCTIONS, encoding='utf-8')
        logger.info(f"Created instructions file at {instructions_path}")
        return DEFAULT_INSTRUCTIONS

    try:
        content = instructions_path.read_text(encoding='utf-8').strip()
        return content if content else DEFAULT_INSTRUCTIONS
    except Exception as e:
        logger.warning(f"Error reading instructions: {e}")
        return DEFAULT_INSTRUCTIONS
