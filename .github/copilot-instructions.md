# Ollama Agent – AI Coding Assistant Guide

## Core Architecture
- `OllamaAgent` in `ollama_agent/agent.py` wraps `openai-agents` with an `AsyncOpenAI` client pointed at the Ollama-compatible `/v1/` endpoint and injects the `execute_command` tool plus any configured MCP servers.
- `run_async_streamed` is the hot path; it yields `text_delta`, `reasoning_delta`, `reasoning_summary`, `tool_call`, `tool_output`, `agent_update`, and `error` payloads that both CLI streaming (`main.run_non_interactive`) and the Textual TUI consume.
- Sessions persist via `agents.SQLiteSession` stored at `~/.ollama-agent/sessions.db`; `_extract_preview_text` expects chat items to stay JSON-serializable and keeps previews under 50 characters.
- MCP descriptors in `~/.ollama-agent/mcp_servers.json` are lazy-loaded; `RunningMCPServer` entries cache async closers, so always hold onto the list until `cleanup()` runs.

## Execution Modes
- `ollama_agent/main.py` owns CLI parsing, task subcommands, timeout overrides, and launches either the streamed console run or the Textual `ChatInterface`.
- `run_non_interactive` uses `rich.Live` to display markdown increments; it pauses the live view for reasoning/tool segments, so keep buffer state (`agent_shown`, `in_reasoning`) in sync with new event types.
- The TUI relies on `StreamingMarkdownRenderer` and `ReasoningRenderer` in `tui/app.py`; both mutate `RichLog.lines` and `_line_cache`, so any rendering changes must preserve the clear/update cycle to avoid duplicate lines.
- TUI key bindings (Ctrl+R/S/T/L) call directly into `OllamaAgent` session/task APIs; remember to propagate new agent behaviors through these bindings.

## Tasks, Tools, Sessions
- `TaskManager` stores YAML tasks in `~/.ollama-agent/tasks/` with IDs `blake2s(title)[:8]`; `find_task_by_prefix` returns `None` when a prefix matches 0 or many files—surface ambiguity to users.
- The sole built-in tool `execute_command` (`tools.py`) shells out with a global timeout; the CLI flag `--builtin-tool-timeout` and TUI constructor both call `set_builtin_tool_timeout`, so keep timeout state process-global.
- Session management commands (`list_sessions`, `get_session_history`, `delete_session`) hit SQLite directly; never bypass `OllamaAgent` when mutating session tables to keep `session_id` in sync.

## Configuration & Instructions
- Config precedence is CLI args → `~/.ollama-agent/config.ini` → defaults from `settings/configini.py`; new options must be added in all three places to stay consistent.
- `load_instructions` seeds `~/.ollama-agent/instructions.md`; edits only apply to new `OllamaAgent` instances, so recreate or `reset_session()` after changing instructions.
- Default connection targets `http://localhost:11434/v1/` with API key `ollama`; reuse these defaults unless the user explicitly overrides them.

## Developer Workflow
- Typical setup: `python -m venv .venv && source .venv/bin/activate && pip install -e .`; install `textual[dev]` if you need Textual tooling.
- Run the TUI with `ollama-agent`; fire one-off prompts via `ollama-agent -p "..."` and optionally `-m/-e/-t` overrides for model, reasoning effort, and command timeout.
- Automated tests are currently empty (`tests/`); manual verification against a live Ollama endpoint is expected when introducing behavior changes.

## Pitfalls & Tips
- Always `await agent.cleanup()` in async entrypoints to shut down MCP processes; both CLI and TUI already do this—mirror that pattern in new flows.
- Streaming handlers assume the event schema from `openai-agents`; expand the `match`/`if` blocks wherever you introduce new event types to avoid silent drops.
- `validate_reasoning_effort` falls back to `"medium"` and logs; validate earlier if you need strict enforcement or user feedback.
- The renderers expect incremental tokens and call `RichLog._line_cache.clear()`; Textual lacks a public API, so keep this workaround until upstream exposes one.
