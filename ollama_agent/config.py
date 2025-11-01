"""Application configuration management."""

import configparser
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG = {
    "default": {
        "model": "gpt-oss:20b",
        "base_url": "http://localhost:11434/v1/",
        "api_key": "ollama",
        "reasoning_effort": "medium"
    }
}

_config = configparser.ConfigParser()
_config_dir = Path.home() / ".ollama-agent"
_config_file = _config_dir / "config.ini"

def _initialize():
    """Initializes the configuration, creating the file if it doesn't exist."""
    global _config
    _config_dir.mkdir(parents=True, exist_ok=True)
    
    if not _config_file.exists():
        _config.read_dict(_DEFAULT_CONFIG)
        with open(_config_file, 'w', encoding='utf-8') as f:
            _config.write(f)
    else:
        _config.read(_config_file)

    if 'default' not in _config:
        _config.add_section('default')

    for key, value in _DEFAULT_CONFIG['default'].items():
        if not _config.has_option('default', key):
            _config.set('default', key, value)

def get(key: str, default: Any = None) -> Any:
    """
    Gets a configuration value from the 'default' section.
    
    Args:
        key: Configuration key.
        default: Default value if the key does not exist.
        
    Returns:
        The configuration value or the default value.
    """
    return _config.get('default', key, fallback=default)

def set(key: str, value: Any) -> bool:
    """
    Sets a configuration value in the 'default' section and saves it.
    
    Args:
        key: Configuration key.
        value: Value to set.
        
    Returns:
        True if saved successfully, False otherwise.
    """
    _config.set('default', key, str(value))
    return save()

def save() -> bool:
    """
    Saves the current configuration to the file.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        with open(_config_file, 'w', encoding='utf-8') as f:
            _config.write(f)
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

# Initialize on module import
_initialize()
