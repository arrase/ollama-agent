"""Application configuration management."""

import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Application configuration with defaults."""
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: str = "medium"


def get_config(config_dir: Path | None = None) -> Config:
    """
    Load configuration from file or create with defaults.
    
    Args:
        config_dir: Directory for config file. Defaults to ~/.ollama-agent
        
    Returns:
        Config object with values from file or defaults.
    """
    config_dir = config_dir or Path.home() / ".ollama-agent"
    config_file = config_dir / "config.ini"
    
    # Default values
    defaults = Config()
    
    # If the file doesn't exist, create it with default values
    if not config_file.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
        parser = configparser.ConfigParser()
        parser.add_section("default")
        parser.set("default", "model", defaults.model)
        parser.set("default", "base_url", defaults.base_url)
        parser.set("default", "api_key", defaults.api_key)
        parser.set("default", "reasoning_effort", defaults.reasoning_effort)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            parser.write(f)
        
        return defaults
    
    # If it exists, read the values
    parser = configparser.ConfigParser()
    parser.read(config_file)
    
    return Config(
        model=parser.get("default", "model", fallback=defaults.model),
        base_url=parser.get("default", "base_url", fallback=defaults.base_url),
        api_key=parser.get("default", "api_key", fallback=defaults.api_key),
        reasoning_effort=parser.get("default", "reasoning_effort", fallback=defaults.reasoning_effort),
    )
