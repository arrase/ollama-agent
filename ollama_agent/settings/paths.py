"""Default filesystem locations for ollama-agent.

These constants are intentionally centralized to keep persistence paths stable
across modules (config, tasks, RAG, MCP).
"""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path.home() / ".ollama-agent"

SETTINGS_PATH = APP_DIR / "settings.yaml"
INSTRUCTIONS_PATH = APP_DIR / "instructions.md"
HISTORY_DB_PATH = APP_DIR / "history.db"
MEMORY_PATH = APP_DIR / "MEMORY.md"
MCP_SERVERS_PATH = APP_DIR / "mcp_servers.json"

TASKS_DIR = APP_DIR / "tasks"
RAG_DIR = APP_DIR / "rag"
SKILLS_DIR = APP_DIR / "skills"
