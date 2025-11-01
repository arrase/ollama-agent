"""Application configuration management."""

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    """Application configuration with defaults."""
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "ollama"
    reasoning_effort: str = "medium"


class ConfigManager:
    """Manages application configuration with file persistence."""
    
    SECTION = "default"
    
    def __init__(self, config_dir: Path | None = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for config file. Defaults to ~/.ollama-agent
        """
        self.config_dir = config_dir or Path.home() / ".ollama-agent"
        self.config_file = self.config_dir / "config.ini"
        self._parser = configparser.ConfigParser()
        self._defaults = Config()
        self._ensure_config()
    
    def _ensure_config(self) -> None:
        """Ensure config directory and file exist with defaults."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config_file.exists():
            self._parser.read(self.config_file)
        
        if self.SECTION not in self._parser:
            self._parser.add_section(self.SECTION)
        
        # Merge defaults with existing config
        changed = False
        for key, value in vars(self._defaults).items():
            if not self._parser.has_option(self.SECTION, key):
                self._parser.set(self.SECTION, key, str(value))
                changed = True
        
        if changed:
            self._save()
    
    def get_value(self, key: str, default: Any = None) -> str:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key.
            default: Default value if key doesn't exist.
            
        Returns:
            Configuration value or default.
        """
        return self._parser.get(self.SECTION, key, fallback=default)
    
    def set_value(self, key: str, value: Any) -> bool:
        """
        Set a configuration value and save.
        
        Args:
            key: Configuration key.
            value: Value to set.
            
        Returns:
            True if saved successfully.
        """
        self._parser.set(self.SECTION, key, str(value))
        return self._save()
    
    def _save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if saved successfully.
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self._parser.write(f)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def as_config(self) -> Config:
        """
        Get configuration as a Config dataclass.
        
        Returns:
            Config object with current values.
        """
        return Config(
            model=self.get_value("model", self._defaults.model),
            base_url=self.get_value("base_url", self._defaults.base_url),
            api_key=self.get_value("api_key", self._defaults.api_key),
            reasoning_effort=self.get_value("reasoning_effort", self._defaults.reasoning_effort),
        )


# Global instance for convenience
_manager = ConfigManager()


def get_value(key: str, default: Any = None) -> str:
    """Get configuration value from global manager."""
    return _manager.get_value(key, default)


def set_value(key: str, value: Any) -> bool:
    """Set configuration value in global manager."""
    return _manager.set_value(key, value)


def get_config() -> Config:
    """Get configuration as Config object."""
    return _manager.as_config()
