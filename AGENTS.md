# Ollama Agent - AI Agent Guidelines

This repository contains the `ollama-agent` CLI and interactive REPL application built on top of DeepAgents, LangChain, and LangGraph.

## Development & Test Commands

- **Run Unit Tests**:
  ```bash
  .venv/bin/python -m unittest discover -s tests
  ```
- **Install in Editable Mode**:
  ```bash
  .venv/bin/pip install -e .
  ```

## Coding Conventions & Engineering Rules

- **KISS & Zero Defensive Bloat**: Write minimal, straightforward code. Avoid artificial abstractions, unnecessary wrappers, or unsolicited fallbacks. Fail fast and fail loud when invariants are violated.
- **Top-Level Imports Only**: All `import` and `from ... import` statements must reside at the very top of each Python file (PEP 8 standard). Never use function-level or inline imports.
- **Virtual Environment**: Always execute Python scripts, tools, and test suites using the project's virtual environment (`.venv/bin/python`).
- **Dependency Management**: Dependencies are declared strictly in `pyproject.toml`.

## Architecture & Context

- **Agent Runtime**: Stateful graph orchestration using `deepagents.create_deep_agent` and `langgraph-checkpoint-sqlite` checkpointing.
- **Memory & Guidelines**:
  - `~/.ollama-agent/MEMORY.md`: Long-term persistent user memory across sessions.
  - `AGENTS.md`: Project-specific instructions loaded into memory at startup.
- **REPL & TUI**: Interactive Textual-based interface in `ollama_agent/interfaces/repl.py`.
- **RAG Engine**: Vector store queries via Qdrant and Ollama embeddings in `ollama_agent/rag/`.
- **MCP & Skills**: Subagent tools via Model Context Protocol and the Agent Skills specification.
