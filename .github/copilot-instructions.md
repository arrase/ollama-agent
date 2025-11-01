# Copilot Instructions for ollama-agent

## Big Picture
- `ollama-agent` wraps `openai-agents` primitives to talk to a local Ollama server exposed as OpenAI-compatible REST (`AsyncOpenAI` with `base_url` defaulting to `http://localhost:11434/v1/`).
- The CLI (`ollama_agent.main:main`) chooses between a non-interactive prompt mode and a Textual-powered TUI for conversational loops.
- Persistent configuration lives in `~/.ollama-agent/config.json`; defaults favour running against an Ollama instance with API key `ollama`.

## Key Modules
- `ollama_agent/agent.py` builds a single `Agent` with one tool (`execute_command`) and sets global client/tracing knobs via `set_default_openai_client`, `set_default_openai_api`, and `set_tracing_disabled`.
- `ollama_agent/tools.py` demonstrates how tools must be decorated with `@function_tool` and return JSON-serializable dictionaries; commands run via `subprocess.run` with a 30s timeout and shell execution.
- `ollama_agent/config.py` encapsulates load/save of config; always go through `Config.get/set` so defaults stay consistent and files are UTF-8 encoded.
- `ollama_agent/tui.py` drives interactive chat using Textual widgets; responses come from `await self.agent.run_async(message)`, so new UI features should stick to async coroutines and update the `RichLog` with `Text` styles.

## Workflows
- Install for development with `pip install -e .`; dependencies are `openai>=1.0.0`, `openai-agents>=0.1.0`, and `textual>=0.47.0`.
- Ensure an Ollama server exposes the OpenAI API-compatible endpoint (`ollama serve --api`) or adjust `base_url` in config.
- Non-interactive run: `ollama-agent --prompt "Mi primer mensaje"` prints both sides of the exchange in the terminal.
- Interactive TUI: run `ollama-agent` (no args); exit with `Ctrl+C`, clear log with `Ctrl+L`.

## Patterns & Gotchas
- `OllamaAgent` must be instantiated before any `Runner.run` calls; if you create new agent instances, repeat the `set_default_*` calls to keep the global openai-agents state aligned.
- New tools should mirror `execute_command`: decorate with `@function_tool`, return structured data, and handle exceptions gracefully since tool errors propagate into chat transcripts.
- When extending config, add fields to `_default_config`, and expect `Config.load` to swallow JSON parsing errors by falling back to defaults.
- Textual message ordering relies on appending to `RichLog`; when adding streaming responses, remember to clear the "thinking..." placeholder similar to the existing pattern.
- `OllamaAgent.run` wraps `asyncio.run`, so avoid calling it from an already running event loop (prefer `run_async` in async contexts).
