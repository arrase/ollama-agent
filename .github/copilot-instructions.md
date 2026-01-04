# Ollama Agent - Copilot Instructions

## Architecture Overview

This is a Python CLI/REPL agent that connects to Ollama-compatible APIs, uses **openai-agents SDK** for orchestration, and supports tool execution, MCP servers, and persistent memory via Mem0+Qdrant.

### Core Components Flow
```
main.py → CLI/REPL interface → OllamaAgent → openai-agents Runner → Ollama API
                                    ↓
                            SessionManager (SQLite)
                            MemoryManager (Mem0 + Qdrant/Docker)
                            MCP Servers (optional delegates)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `agent/agent.py` | `OllamaAgent` dataclass - main orchestrator with `lifespan()` context manager |
| `agent/factory.py` | `create_agent()` factory that merges config with runtime overrides |
| `agent/builtin_tools.py` | `@function_tool` decorated tools: `execute_command`, `mem0_*` |
| `settings/config.py` | `Config` dataclass, reads `~/.ollama-agent/config.ini` |
| `settings/mcp/lifecycle.py` | MCP server init/cleanup, creates delegate agents per server |
| `streaming/events.py` | Event dispatch: `event_payloads()` extracts typed dicts from SDK events |
| `interfaces/repl.py` | Interactive REPL with slash commands, uses `prompt_toolkit` + `rich` |
| `execution/runner.py` | Non-interactive one-shot execution |

## Development Workflow

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run
ollama-agent              # REPL mode
ollama-agent -p "query"   # Non-interactive

# Clean pycache
find . -type d -name __pycache__ -exec rm -rf {} +
```

No test suite exists yet. Manual testing via REPL/CLI.

## Patterns & Conventions

### Tool Definition Pattern
Use `@function_tool` decorator from `agents` SDK. Tools return typed dicts (`CommandResult`, `Mem0ToolResult`):
```python
@function_tool
def execute_command(command: str) -> CommandResult:
    # Returns {"success": bool, "stdout": str, "stderr": str, "exit_code": int}
```

### Context Variables for Cross-Cutting State
Uses `contextvars.ContextVar` for thread-safe tool config:
```python
_tool_timeout: ContextVar[int] = ContextVar("tool_timeout", default=30)
set_tool_timeout = _tool_timeout.set
```

### Async Lifecycle Pattern
`OllamaAgent` uses async context manager for resource cleanup:
```python
async with agent.lifespan():
    # MCP servers initialized, memory ready
    await agent.run_async_streamed(prompt)
# Cleanup guaranteed
```

### Streaming Events
Agent emits typed payloads: `text_delta`, `reasoning_delta`, `tool_call`, `tool_output`, `error`. Renderers implement `StreamingRenderer` ABC.

### MCP Server Delegation
Each MCP server becomes a delegate agent exposed as a tool (`use_<name>`). Config in `~/.ollama-agent/mcp_servers.json`:
```json
{"mcpServers": {"filesystem": {"command": "npx", "args": [...]}}}
```

## Configuration Locations

- `~/.ollama-agent/config.ini` - Main config (model, base_url, reasoning_effort)
- `~/.ollama-agent/instructions.md` - Agent system prompt (user-editable)
- `~/.ollama-agent/mcp_servers.json` - MCP server definitions
- `~/.ollama-agent/sessions.db` - SQLite session persistence

## Important Implementation Details

1. **Model Validation**: `core/models.py` checks Ollama capabilities via `ollama.show()` before runs
2. **Screen Vision**: `@dpN` tokens in prompts trigger screenshot capture via `mss` library
3. **Mem0 Bootstrap**: `memory/bootstrap.py` auto-starts Qdrant Docker container if missing
4. **Reasoning Effort**: Passed to SDK as `ModelSettings(reasoning=Reasoning(effort=...))`, values: `low|medium|high|disabled`

## Adding New Built-in Tools

1. Define function in `agent/builtin_tools.py` with `@function_tool` decorator
2. Add to `BUILTIN_TOOLS` list
3. Use type hints - SDK generates schema from them
4. Return a TypedDict for structured output
