# Ollama Agent - AI Coding Assistant Guide

## Architecture Overview

This is a **Python CLI/TUI application** that wraps local AI models (via Ollama) with tool execution capabilities and persistent session management. The current branch (`feat-streaming-response`) is focused on adding streaming response capabilities.

### Core Components

1. **OllamaAgent** (`agent.py`): Main orchestrator wrapping `openai-agents` library
   - Manages AsyncOpenAI client connected to Ollama-compatible API
   - Handles session persistence via SQLite (using `SQLiteSession` from `openai-agents`)
   - Lazy-initializes MCP servers on first agent run
   - Agent lifecycle: `__init__` → `_ensure_mcp_servers_initialized` → `run_async` → `cleanup`

2. **Entry Points** (`main.py`): Two execution modes
   - **TUI mode** (default): Textual-based interactive chat with keybindings
   - **CLI mode** (`--prompt`): Single-shot execution with markdown-rendered output

3. **Tool System** (`tools.py`): Shell command execution via `@function_tool` decorator
   - Global timeout controlled by `set_builtin_tool_timeout()` (default: 30s)
   - Uses `subprocess.run(shell=True)` with structured `CommandResult` return type

4. **Task Management** (`tasks.py`): Save/run reusable prompts as YAML files
   - Task IDs computed using Blake2s hash (8 chars) of title
   - Stored in `~/.ollama-agent/tasks/*.yaml`

5. **MCP Integration** (`settings/mcp.py`): Model Context Protocol server support
   - Supports stdio (`MCPServerStdio`) and HTTP (`MCPServerSse`, `MCPServerStreamableHttp`)
   - Config: `~/.ollama-agent/mcp_servers.json` with `mcpServers` object

## Key Patterns & Conventions

### Async/Await Usage
- All agent execution methods are async: `run_async()`, `get_session_history()`, `_ensure_mcp_servers_initialized()`
- MCP server lifecycle managed via async context managers (`__aenter__`/`__aexit__`)
- Always call `await agent.cleanup()` in `finally` blocks for non-interactive mode

### Session Management
- Sessions stored in `~/.ollama-agent/sessions.db` (SQLite)
- Session ID is UUID4, loaded via `load_session(id)` or created via `reset_session()`
- Preview text extracted from first message (max 50 chars) using `_extract_preview_text()`

### Configuration Hierarchy
1. CLI args (`--model`, `--effort`, `--builtin-tool-timeout`) override
2. Config file (`~/.ollama-agent/config.ini`)
3. Hardcoded defaults in `Config` dataclass

### Reasoning Effort
- Valid values: `"low"`, `"medium"`, `"high"` (enforced by `ReasoningEffortValue` Literal type)
- Passed to `ModelSettings(reasoning=Reasoning(effort=...))` for OpenAI API
- Validated via `validate_reasoning_effort()` with fallback to `"medium"`

### File Organization
```
~/.ollama-agent/
├── config.ini              # User config (auto-created)
├── sessions.db             # SQLite conversation history
├── instructions.md         # Agent system prompt (auto-created from DEFAULT_INSTRUCTIONS)
├── mcp_servers.json        # MCP server definitions
└── tasks/*.yaml            # Saved task files (Blake2s hash names)
```

## Development Workflows

### Running the Application
```bash
# Install in editable mode (for development)
pip install -e .

# Interactive TUI
ollama-agent

# Non-interactive CLI
ollama-agent -p "Your prompt here" -m "model-name" -e "high"

# Task management
ollama-agent task-list
ollama-agent task-run <id>
ollama-agent task-delete <id>
```

### Testing Considerations
- Requires running Ollama instance at `http://localhost:11434/v1/` (configurable)
- Test with different models using `-m` flag
- MCP servers are optional; agent works without them
- Session DB can be deleted for fresh state: `rm ~/.ollama-agent/sessions.db`

### Adding New Tools
1. Define function in `tools.py` with `@function_tool` decorator
2. Add to `OllamaAgent._create_agent()` tools list
3. Return structured TypedDict for consistent error handling (see `CommandResult`)

### Adding MCP Servers
Edit `~/.ollama-agent/mcp_servers.json`:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_mcp_server"],
      "cache_tools_list": true,
      "max_retry_attempts": 3
    }
  }
}
```

## Common Pitfalls

- **Don't forget `await agent.cleanup()`** - MCP connections must be explicitly closed
- **Task IDs are derived from title hash** - changing title changes ID, creates new task
- **Session loading doesn't validate existence** - check `list_sessions()` first
- **Agent instructions loaded once at init** - changes to `instructions.md` require restart
- **Reasoning effort validation is silent** - invalid values fall back to "medium" with warning

## Textual TUI Structure

Located in `ollama_agent/tui/`:
- `app.py`: Main `ChatInterface` app with keybindings (Ctrl+R/S/L/T/C)
- `session_list_screen.py`: Modal screen for session selection
- `task_list_screen.py`: Modal screen for task management
- `create_task_screen.py`: Modal form for creating new tasks

TUI uses `RichLog` for chat display with markdown rendering via `Rich` library.
