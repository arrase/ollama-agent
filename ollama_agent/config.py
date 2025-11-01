"""Application configuration management."""

import json
from pathlib import Path
from typing import Any


class Config:
    """Manages application configuration."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".ollama-agent"
        self.config_file = self.config_dir / "config.json"
        self._ensure_config_dir()
        
    def _ensure_config_dir(self):
        """Creates the configuration directory if it does not exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def load(self) -> dict[str, Any]:
        """
        Loads the configuration from the file.
        
        Returns:
            Dictionary with the configuration, or default values if it does not exist.
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self._default_config()
        return self._default_config()
    
    def save(self, config: dict[str, Any]):
        """
        Saves the configuration to the file.
        
        Args:
            config: Dictionary with the configuration to save.
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def _default_config(self) -> dict[str, Any]:
        """Returns the default configuration."""
        return {
            "model": "gpt-oss:20b",
            "base_url": "http://localhost:11434/v1/",
            "api_key": "ollama"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a configuration value.
        
        Args:
            key: Configuration key.
            default: Default value if the key does not exist.
        
        Returns:
            The configuration value or the default value.
        """
        config = self.load()
        return config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Sets a configuration value.
        
        Args:
            key: Configuration key.
            value: Value to set.
        """
        config = self.load()
        config[key] = value
        self.save(config)
