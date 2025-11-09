## Project Snapshot
- CLI entrypoint `ollama_agent/main.py` selects between streamed CLI commands and the Textual TUI; treat CLI args (model, effort, timeout) as authoritative overrides on top of config defaults.
- Core agent runs on `openai-agents` with `AsyncOpenAI` pointed at Ollama; always verify models via `ensure_model_supports_tools` so tool calls stay available.
- Textual app in `ollama_agent/tui/app.py` streams responses through custom renderers, so preserve event ordering and markdown formatting contracts.
- Persistence, config, and instructions live under `~/.ollama-agent`; keep backwards compatibility when touching paths or defaults.

## Core Modules & Contracts
- `create_agent` centralizes `OllamaAgent` construction; pass overrides into it instead of mutating global state so caches and MCP init stay correct.
- `OllamaAgent` caches agents keyed by `(model, effort)`; clear `_agent_cache` whenever instructions or MCP servers change to avoid stale tool lists.
- Streamed runs emit `text_delta`, `reasoning_delta`, `reasoning_summary`, `tool_call`, `tool_output`, `agent_update`; new features should extend handlers rather than changing these keys.
- Built-ins live in `agent/tools.py`; respect `set_builtin_tool_timeout` when adding tools and return dicts compatible with `CommandResult`.

## State & Persistence
- `SessionManager` wraps `agents.SQLiteSession` storing data at `~/.ollama-agent/sessions.db`; avoid schema changes unless mirrored in existing tables (`agent_sessions`, `agent_messages`).
- Call `reset_session` when starting fresh conversations to keep UI state aligned with the backing SQLite session ID.
- `TaskManager` stores YAML per BLAKE2 hash of the title (first 8 chars); modifying `compute_task_id` can orphan saved tasks.
- `settings/configini.load_instructions` backfills `instructions.md`; reuse it so user overrides survive onboarding flows.
- MCP servers load from `~/.ollama-agent/mcp_servers.json`; initialization happens lazily, so update flows should re-run `initialize_mcp_servers` or trigger `cleanup_mcp_servers` on shutdown.

## Developer Workflow
- Install locally with `pip install -e .` (Python ≥3.9) and ensure an Ollama server exposing tool-capable models is running before manual tests.
- Use `ollama-agent -p "prompt"` to exercise the streamed CLI path and bare `ollama-agent` to validate TUI interactions after changes.
- Validate persistence via CLI task commands (`task-list`, `task-run`, `task-delete`) and TUI shortcuts (Ctrl+R/Ctrl+S/Ctrl+L/Ctrl+T).
- Enable verbose logging with `LOGLEVEL=DEBUG` when chasing MCP or session issues; logging already routes through the stdlib logger.
- With no automated tests, sanity-check both `run_async` and `run_async_streamed` flows whenever touching agent execution or event handling.

## Extension Points
- New agent tools should use `@function_tool` and return JSON-serializable payloads that mirror `CommandResult`; keep outputs concise for both CLI and TUI previews.
- Extend the TUI by adding screens under `ollama_agent/tui`, following existing CSS/binding patterns so shortcuts remain consistent.
- If you tweak streaming UX, update `StreamingMarkdownRenderer` and `ReasoningRenderer` together to avoid RichLog flicker and dangling reasoning banners.
- When introducing config knobs, add fields to the `Config` dataclass, write defaults into `config.ini`, and wire them through CLI flags if user-facing.
- Document new commands or behaviors in `README.md` to keep feature discoverability in sync with CLI/TUI affordances.
