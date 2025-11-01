"""Gestión de configuración de la aplicación."""

import json
from pathlib import Path
from typing import Any


class Config:
    """Gestiona la configuración de la aplicación."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".ollama-agent"
        self.config_file = self.config_dir / "config.json"
        self._ensure_config_dir()
        
    def _ensure_config_dir(self):
        """Crea el directorio de configuración si no existe."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def load(self) -> dict[str, Any]:
        """
        Carga la configuración desde el archivo.
        
        Returns:
            Diccionario con la configuración, o valores por defecto si no existe.
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
        Guarda la configuración en el archivo.
        
        Args:
            config: Diccionario con la configuración a guardar.
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error al guardar configuración: {e}")
    
    def _default_config(self) -> dict[str, Any]:
        """Retorna la configuración por defecto."""
        return {
            "model": "gpt-oss:20b",
            "base_url": "http://localhost:11434/v1/",
            "api_key": "ollama"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración.
        
        Args:
            key: Clave de configuración.
            default: Valor por defecto si la clave no existe.
            
        Returns:
            El valor de configuración o el valor por defecto.
        """
        config = self.load()
        return config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Establece un valor de configuración.
        
        Args:
            key: Clave de configuración.
            value: Valor a establecer.
        """
        config = self.load()
        config[key] = value
        self.save(config)
