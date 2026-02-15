"""Default filesystem locations for ollama-agent.

These constants are intentionally centralized to keep persistence paths stable
across modules (config, tasks, RAG, Mem0, MCP).
"""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path.home() / ".ollama-agent"

INSTRUCTIONS_PATH = APP_DIR / "instructions.md"
DATABASE_PATH = APP_DIR / "sessions.db"
MCP_SERVERS_PATH = APP_DIR / "mcp_servers.json"

TASKS_DIR = APP_DIR / "tasks"
RAG_DIR = APP_DIR / "rag"
MEMORY_DIR = APP_DIR / "memory"
SKILLS_DIR = APP_DIR / "skills"
