"""Settings package for application configuration."""

from .config import (
    MentionSettings,
    ModelSettings,
    RAGSettings,
    RuntimeSettings,
    Settings,
    SubAgentMCPServer,
    SubAgentSettings,
    ensure_memory_file,
    load_instructions,
    load_settings,
    reset_config,
    save_settings,
)
from .paths import (
    APP_DIR,
    HISTORY_DB_PATH,
    INSTRUCTIONS_PATH,
    MCP_SERVERS_PATH,
    MEMORY_PATH,
    RAG_DIR,
    SETTINGS_PATH,
    SKILLS_DIR,
    TASKS_DIR,
)

__all__ = [
    # Paths
    "APP_DIR",
    "HISTORY_DB_PATH",
    "INSTRUCTIONS_PATH",
    "MCP_SERVERS_PATH",
    "MEMORY_PATH",
    "RAG_DIR",
    "SETTINGS_PATH",
    "SKILLS_DIR",
    "TASKS_DIR",
    # Settings
    "MentionSettings",
    "ModelSettings",
    "RAGSettings",
    "RuntimeSettings",
    "Settings",
    "SubAgentMCPServer",
    "SubAgentSettings",
    "ensure_memory_file",
    "load_instructions",
    "load_settings",
    "reset_config",
    "save_settings",
]
