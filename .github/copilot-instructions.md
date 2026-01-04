
# Instructions for code agents (ollama-agent)

## Quick Overview
- CLI/REPL packaged as `ollama-agent` (entrypoint: `ollama_agent/main.py`, defined in `pyproject.toml`).
- The core is `OllamaAgent` in `ollama_agent/agent/agent.py`: it creates an `agents.Agent` (openai-agents) using `AsyncOpenAI` and points to an Ollama-compatible endpoint (default `base_url` is `http://localhost:11434/v1/`).
- Persistent state:
  - Sessions/chats: SQLite via `agents.SQLiteSession` at `~/.ollama-agent/sessions.db` (`ollama_agent/agent/session_manager.py`).
  - Config: `~/.ollama-agent/config.ini` (created on first run) (`ollama_agent/settings/config.py`).
  - Instructions: `~/.ollama-agent/instructions.md` (created on first run; defaults from `ollama_agent/settings/default_instructions.md`).
  - Tasks: YAML files in `~/.ollama-agent/tasks/*.yaml` (`ollama_agent/tasks/manager.py`).

## Main Flows
- REPL: `ollama_agent/interfaces/repl.py` (slash commands) → `OllamaAgent.run_async_streamed()` → incremental rendering.
- Non-interactive CLI: `ollama_agent/execution/runner.py` uses `stream_agent_events_with_renderer()`.
- Streaming: events from openai-agents are normalized to payloads `{type: ...}` in `ollama_agent/streaming/events.py`.

## Key Integrations (and where to touch them)
- Built-in tools:
  - Defined with `@agents.function_tool` in `ollama_agent/agent/builtin_tools.py`.
  - Exposed by adding them to `BUILTIN_TOOLS`.
  - Tool timeouts: `set_tool_timeout()` (ContextVar) is configured in `ollama_agent/main.py`.
- Model capability gate: before creating the agent the support for tools is validated with `ensure_model_supports_tools()` (`ollama_agent/core/models.py`, it calls `ollama.show(model)` and looks for the `"tools"` capability).
- Screen vision: the `@dpN` token in prompts → captures a screenshot and converts it to a multimodal input in the Responses style (`_maybe_attach_screen_context()` in `ollama_agent/agent/agent.py`, helpers in `ollama_agent/vision/screen.py`). Note: on Linux this requires `DISPLAY`/`WAYLAND_DISPLAY`.
- Persistent memory (Mem0 + Qdrant): `MemoryManager` in `ollama_agent/memory/memory_manager.py`.
  - On initialization, it starts/ensures Qdrant via Docker (`ollama_agent/memory/bootstrap.py`, container `ollama-agent-qdrant-<port>`).
  - Memory tools: `mem0_add_memory`, `mem0_search_memory`.
- MCP (optional): servers declared in `~/.ollama-agent/mcp_servers.json`.
  - Lifecycle: `ollama_agent/settings/mcp/lifecycle.py`.
  - Builders: `ollama_agent/settings/mcp/builders.py` (stdio/sse/streamable_http) and it creates a "delegated agent" that is exposed as a tool (named `use_<name>` by default).

## Project Conventions
- Prefer `prompt: object` (string or multimodal list). If you include multimodal messages, preserve the callback pattern `RunConfig(session_input_callback=...)` in `OllamaAgent._prepare_input()`.
- Agent runtime errors: `run_async()` returns a string `"Error: ..."` (it does not raise) and streaming emits a payload `{type:"error"}`.
- Task IDs: only `[A-Za-z0-9_-]+` (validated by `TaskManager.validate_task_id`).

## Real-world Dev Workflows
- Local setup:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e .`
- Run:
  - REPL: `ollama-agent`
  - Single prompt: `ollama-agent -p "..."`
  - Override model/effort/timeout: `ollama-agent -m <model> -e <low|medium|high|disabled> -t <seconds> -p "..."`

## When Making Changes
- If you change CLI options: edit `ollama_agent/interfaces/cli.py` and maintain compatibility with `README.md`.
- If you add new streaming payload types: update `ollama_agent/streaming/events.py` and the renderer(s) (`ollama_agent/streaming/console_renderer.py` and the REPL).
