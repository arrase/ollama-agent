"""Application configuration management."""

import json
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG = {
    "model": "gpt-oss:20b",
    "base_url": "http://localhost:11434/v1/",
    "api_key": "ollama",
    "reasoning_effort": "medium"
}


class Config:
    """Manages application configuration."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".ollama-agent"
        self.config_file = self.config_dir / "config.json"
        self._cache: dict[str, Any] | None = None
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
        if self._cache is not None:
            return self._cache
            
        if not self.config_file.exists():
            self._cache = _DEFAULT_CONFIG.copy()
            self.save(self._cache)
            return self._cache
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                self._cache = {**_DEFAULT_CONFIG, **config}
                return self._cache
        except (json.JSONDecodeError, OSError) as e:
            # On error, use defaults and inform user
            print(f"Warning: Could not load config from '{self.config_file}': {e}")
            print("Using default configuration.")
            self._cache = _DEFAULT_CONFIG.copy()
            return self._cache
    
    def save(self, config: dict[str, Any]) -> bool:
        """
        Saves the configuration to the file.
        
        Args:
            config: Dictionary with the configuration to save.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self._cache = config
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
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
    
    def set(self, key: str, value: Any) -> bool:
        """
        Sets a configuration value.
        
        Args:
            key: Configuration key.
            value: Value to set.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        config = self.load()
        config[key] = value
        return self.save(config)
