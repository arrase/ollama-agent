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

class Config:
    """Manages application configuration."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".ollama-agent"
        self.config_file = self.config_dir / "config.ini"
        self._config = configparser.ConfigParser()
        self._ensure_config_dir_and_load()
        
    def _ensure_config_dir_and_load(self):
        """Creates the configuration directory if it does not exist and loads config."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not self.config_file.exists():
            self._config.read_dict(_DEFAULT_CONFIG)
            self.save()
        else:
            self._config.read(self.config_file)

        if 'default' not in self._config:
            self._config['default'] = {}

        for key, value in _DEFAULT_CONFIG['default'].items():
            if key not in self._config['default']:
                self._config['default'][key] = value

    def load(self) -> dict[str, Any]:
        """
        Returns the configuration as a dictionary.
        
        Returns:
            Dictionary with the configuration.
        """
        return dict(self._config.items('default'))
    
    def save(self) -> bool:
        """
        Saves the current configuration to the file.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self._config.write(f)
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
        return self._config.get('default', key, fallback=default)
    
    def set(self, key: str, value: Any) -> bool:
        """
        Sets a configuration value.
        
        Args:
            key: Configuration key.
            value: Value to set.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        self._config.set('default', key, str(value))
        return self.save()
