## Project Focus
- Local-first assistant wrapping Ollama-compatible models; entry is `ollama_agent/main.py`, which dispatches to CLI commands or launches the Textual TUI.
- Packaging is configured through `pyproject.toml`; the `ollama-agent` console script boots `main()`.

## Architecture Notes
- CLI flows live in `ollama_agent/cli.py`; subcommands (`task-list`, `task-run`, `task-delete`) short-circuit `main()` before the TUI starts.
- The TUI (`ollama_agent/tui/app.py`) renders chat via Textual `RichLog`, with modal screens in `tui/*.py` for sessions and tasks; bindings (Ctrl+R/S/L/T) call `ChatInterface.action_*` helpers.
- `OllamaAgent` (`agent/agent.py`) wraps `openai-agents` `Runner`; `run_async_streamed` yields `text_delta`, `reasoning_delta`, `tool_call`, `tool_output`, `agent_update`, and `error` payloads—reuse these event names when extending stream consumers.
- `SessionManager` (`agent/session_manager.py`) persists history with `agents.SQLiteSession` in `~/.ollama-agent/sessions.db`; resetting returns a fresh UUID and reinitialises the backing session.
- Task storage (`tasks.py`) hashes titles with BLAKE2s to form 8-char IDs and serialises YAML under `~/.ollama-agent/tasks`.

## Configuration & Conventions
- User config lives in `~/.ollama-agent/config.ini`; `settings/configini.get_config()` returns defaults (`gpt-oss:20b`, `http://localhost:11434/v1/`, timeout 30s). Calls to `create_agent()` should accept optional overrides instead of reading env vars directly.
- Agent instructions are loaded from `~/.ollama-agent/instructions.md`; `load_instructions()` auto-creates the file with default tool-usage scaffolding. Keep edits idempotent—empty files fall back to defaults.
- Global built-in tool timeout is mutated through `agent.tools.set_builtin_tool_timeout()`. Always set this before invoking `execute_command` to keep CLI and TUI consistent.
- Reasoning effort strings must pass through `utils.validate_reasoning_effort`; invalid values downgrade to `"medium"` while logging a warning.

## External Integrations
- `agent/tools.execute_command` is the only built-in tool and executes `subprocess.run(shell=True)`; adhere to timeout and capture semantics when adding new tools.
- MCP support (`settings/mcp.py`) loads `mcp_servers.json` with `mcpServers` entries; supported transports are `stdio`, `streamable_http`, and `sse`. `initialize_mcp_servers()` returns `RunningMCPServer` entries whose `cleanup()` must be awaited on shutdown.
- Streaming relies on `openai` >= 1.0 events; tests or mocks should mirror `ResponseTextDeltaEvent` / `ResponseReasoningTextDeltaEvent` behaviour solved in `_raw_event_payloads`.

## Developer Workflow
- Recommended setup: `python -m venv .venv && source .venv/bin/activate`, then `pip install -e .` (optionally `pip install "textual[dev]"`).
- Launch interactive UI with `ollama-agent` (respects config overrides) or `python -m ollama_agent.main`.
- Non-interactive runs use `ollama-agent -p "..."`; streaming output handled by `run_non_interactive()` with Rich live updates.
- No automated tests ship with `tests/`; when adding tests, prefer async-friendly patterns for agent streaming and stub filesystem paths to avoid touching `~/.ollama-agent`.

## Extension Tips
- When extending session/task dialogs, refresh views via `refresh(recompose=True)` after mutations—current screens depend on that pattern.
- Preserve Rich/Textual rendering cadence: `StreamingMarkdownRenderer` flushes every 5 tokens and wipes `RichLog._line_cache`; avoid removing this to keep markdown diff-free.
- Any new CLI command should be routed through `handle_cli_commands()` and return `True` once handled to prevent the TUI from launching inadvertently.
- Keep new config knobs mirrored across CLI args, `Config` dataclass defaults, and the generated `config.ini` skeleton.
