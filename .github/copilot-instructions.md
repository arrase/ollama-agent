# Copilot Instructions for ollama-agent

## Big Picture
- `ollama_agent` wraps `openai-agents` primitives so a local Ollama instance can be driven through an OpenAI-compatible API (`AsyncOpenAI` against `http://localhost:11434/v1/`).
- `main.py` starts either a one-off CLI exchange (`--prompt`) or a Textual-powered chat UI; both funnel user input into `OllamaAgent.run_async`.
- Persistent settings live in `~/.ollama-agent/config.json`; defaults prefer the `gpt-oss:20b` model, the local Ollama base URL, and `reasoning_effort="medium"`.
- The runtime agent exposes a single tool (`execute_command`) that lets the model run shell commands; additional tools follow the same pattern and must stay JSON-serializable.

## Key Modules & Patterns
- `ollama_agent/agent.py`: constructs `OllamaAgent` and performs global `openai-agents` setup (`set_tracing_disabled`, `set_default_openai_api`, `set_default_openai_client`). Reapply the same sequence when introducing new agent variants or tests to avoid OpenAI cloud calls.
- `ollama_agent/tools.py`: `@function_tool` exports must handle their own timeouts/errors because failures are surfaced in chat transcripts; shell execution is capped at 30s via `subprocess.run(..., timeout=30)`.
- `ollama_agent/config.py`: always modify config through `Config.load/save/get/set`; malformed JSON is silently replaced with `_default_config`, so guard expensive migrations with explicit validation if required.
- `ollama_agent/tui.py`: asynchronous Textual `App`; extend interactions with `await`-friendly routines and update the `RichLog` after clearing the "thinking..." placeholder to keep the transcript readable.
- `ollama_agent/main.py`: CLI flags override config values; remember to pass `choices=list(ALLOWED_REASONING_EFFORTS)` whenever you surface new knobs so help text stays aligned.

## Workflows & Commands
- Install locally with `pip install -e .`; entry point `ollama-agent` is registered via `pyproject.toml`.
- Ensure Ollama serves an OpenAI-compatible endpoint (`ollama serve --api`); override `base_url`, `model`, or `api_key` either via CLI flags or by editing `~/.ollama-agent/config.json`.
- Non-interactive call: `ollama-agent --prompt "hola" --effort low` prints both user and agent messages; logs "Agent: thinking..." while awaiting the async run.
- Interactive TUI: `ollama-agent` with no flags; `Ctrl+C` quits, `Ctrl+L` wipes the log, and the footer shows active key bindings.
- When scripting, prefer `OllamaAgent.run_async` inside async code (avoid nested `asyncio.run` calls from existing event loops such as Textual callbacks).

## Extending the Project
- Add tools by decorating callables with `@function_tool` and returning dictionaries containing only JSON-safe values; include friendly error messages so the agent can reason about failures.
- New configuration keys belong in `_default_config` and should respect UTF-8 encoding (`json.dump(..., ensure_ascii=False)`); document them in help text for discoverability.
- If you introduce streaming or partial responses, keep the TUI update pattern that clears the placeholder line and scrolls to the end (`chat_log.scroll_end`).
- Tests are currently absent; if you add them, emulate existing async flows by faking `Runner.run` or by substituting `execute_command` to keep runs deterministic.

## External Dependencies
- `openai-agents>=0.1.0` provides the `Agent`, `Runner`, and decorator utilities; many behaviors (e.g., global client state) are library-driven, so read its docs before bypassing helpers.
- `textual>=0.47.0` powers the TUI; follow its async/await lifecycle (`compose`, `on_mount`, `on_input_submitted`) when adding widgets.
- `openai>=1.0.0` supplies `AsyncOpenAI`; the client works offline as long as the `base_url` points at Ollama.
