"""Default filesystem locations for ollama-agent.

These constants are intentionally centralized to keep persistence paths stable
across modules (config, tasks, RAG, MCP).
"""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path.home() / ".ollama-agent"

SETTINGS_PATH = APP_DIR / "settings.yaml"
PROMPTS_DIR = APP_DIR / "prompts"
INSTRUCTIONS_PATH = PROMPTS_DIR / "instructions.md"
FS_POLICY_TRAVERSAL_PATH = PROMPTS_DIR / "fs_policy_traversal.md"
FS_POLICY_SANDBOXED_PATH = PROMPTS_DIR / "fs_policy_sandboxed.md"
RAG_POLICY_PATH = PROMPTS_DIR / "rag_policy.md"
HISTORY_DB_PATH = APP_DIR / "history.db"
MEMORY_PATH = APP_DIR / "MEMORY.md"
AGENTS_MD_NAME = "AGENTS.md"
AGENTS_PATH = APP_DIR / AGENTS_MD_NAME
MCP_PATH = APP_DIR / "mcp.json"
MCP_LOG_PATH = APP_DIR / "mcp.log"

TASKS_DIR = APP_DIR / "tasks"
RAG_DIR = APP_DIR / "rag"
SKILLS_DIR = APP_DIR / "skills"
BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills" / "builtin"

