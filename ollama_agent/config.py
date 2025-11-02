"""Application configuration management."""

import configparser
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_DIR = Path.home() / ".ollama-agent"
DEFAULT_DATABASE_PATH = DEFAULT_CONFIG_DIR / "sessions.db"


@dataclass
class Config:
    """Application configuration with defaults."""
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: str = "medium"
    database_path: Path = field(default_factory=lambda: DEFAULT_DATABASE_PATH)
    builtin_tool_timeout: int = 30


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
        parser.set("default", "builtin_tool_timeout", str(defaults.builtin_tool_timeout))
        
        with open(config_file, 'w', encoding='utf-8') as f:
            parser.write(f)
        return defaults
    
    # Load from existing file
    parser = configparser.ConfigParser()
    parser.read(config_file)
    
    return Config(
        model=parser.get("default", "model", fallback=defaults.model),
        base_url=parser.get("default", "base_url", fallback=defaults.base_url),
        api_key=parser.get("default", "api_key", fallback=defaults.api_key),
        reasoning_effort=parser.get("default", "reasoning_effort", fallback=defaults.reasoning_effort),
        database_path=Path(parser.get("default", "database_path", fallback=str(defaults.database_path))),
        builtin_tool_timeout=int(parser.get("default", "builtin_tool_timeout", fallback=str(defaults.builtin_tool_timeout))),
    )
