
# Copilot instructions (ollama-agent)

## Big picture
- Package entrypoint: `ollama-agent` → `ollama_agent/main.py` (declared in `pyproject.toml`).
- Core runtime: `OllamaAgent` in `ollama_agent/agent/agent.py` builds a DeepAgents graph (`deepagents.create_deep_agent`) backed by `langchain-ollama` (`ChatOllama`).
- User state lives in `~/.ollama-agent/`: `config.ini`, `sessions.db`, `instructions.md`, `tasks/*.yaml`, and optional `mcp_servers.json`.

## Main flows (debug here first)
- Non-interactive CLI: `ollama_agent/interfaces/cli.py` → `ollama_agent/execution/runner.py:run_non_interactive()` → `ollama_agent/streaming/events.py:stream_agent_events_with_renderer()`.
- REPL: `ollama_agent/interfaces/repl.py` → `OllamaAgent.run_async_streamed()`; slash commands reuse task helpers from `ollama_agent/tasks/commands.py`.
- Streaming: `OllamaAgent.run_async_streamed()` yields normalized `{type: ...}` payloads consumed by renderers.

## Extension points & integrations
- Built-in tools: add `@langchain.tools.tool` functions in `ollama_agent/agent/builtin_tools.py`, then include them in `BUILTIN_TOOLS`.
- Tool timeout: controlled via a `ContextVar` (`set_tool_timeout()`), configured from config/CLI in `ollama_agent/main.py`.
- Model capability gate: `ollama_agent/core/models.py:ensure_model_supports_tools()` checks `ollama.show(model)` for capability `"tools"` before agent creation.
- Screen vision: `@dpN` tokens in string prompts trigger screenshot capture and multimodal conversion in `_maybe_attach_screen_context()` (`ollama_agent/vision/screen.py`). On Linux requires `DISPLAY` or `WAYLAND_DISPLAY`.
- Mem0 memory: `ollama_agent/memory/memory_manager.py` uses Mem0 + embedded/local Qdrant persistence via `mem0.qdrant_path`. Tools are `mem0_add_memory` / `mem0_search_memory`.
- MCP servers (optional): loaded from `~/.ollama-agent/mcp_servers.json` in `OllamaAgent.initialize()` via `ollama_agent/settings/mcp/lifecycle.py`; each server becomes a delegated tool named `use_<name>` by default.

## Project-specific conventions
- `prompt` is intentionally typed as `object`: typically `str`; multimodal prompts are represented using standard content blocks.
- Error behavior is non-raising by design: `run_async()` returns `"Error: ..."`; streaming yields `{type:"error"}`.
- Tasks: YAML schema is `title`, `prompt`, `model`, optional `reasoning_effort` (see `ollama_agent/tasks/manager.py`). Task IDs must match `[A-Za-z0-9_-]+`.

## Dev workflow
- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`
- Run: `ollama-agent` or `ollama-agent -p "..."`.
- Common overrides: `ollama-agent -m <model> -e <low|medium|high|disabled> -t <seconds> -p "..."`.
- Note (README): `--effort` currently only affects `gpt-oss` models; use `disabled` for other models to avoid surprises.
- If you change CLI flags/help, update both `ollama_agent/interfaces/cli.py` and `README.md`.
