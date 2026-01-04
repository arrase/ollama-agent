# Instrucciones para agentes de código (ollama-agent)

## Panorama rápido
- CLI/REPL empaquetado como `ollama-agent` (entrypoint: `ollama_agent/main.py`, definido en `pyproject.toml`).
- El “core” es `OllamaAgent` en `ollama_agent/agent/agent.py`: crea un `agents.Agent` (openai-agents) sobre `AsyncOpenAI` apuntando a un endpoint compatible con Ollama (`base_url` por defecto `http://localhost:11434/v1/`).
- Estado persistente:
  - Sesiones/chat: SQLite via `agents.SQLiteSession` en `~/.ollama-agent/sessions.db` (`ollama_agent/agent/session_manager.py`).
  - Config: `~/.ollama-agent/config.ini` (se crea al primer run) (`ollama_agent/settings/config.py`).
  - Instrucciones: `~/.ollama-agent/instructions.md` (se crea al primer run; defaults desde `ollama_agent/settings/default_instructions.md`).
  - Tasks: YAML en `~/.ollama-agent/tasks/*.yaml` (`ollama_agent/tasks/manager.py`).

## Flujos principales
- REPL: `ollama_agent/interfaces/repl.py` (slash commands) → `OllamaAgent.run_async_streamed()` → renderizado incremental.
- CLI no interactivo: `ollama_agent/execution/runner.py` usa `stream_agent_events_with_renderer()`.
- Streaming: eventos de openai-agents se normalizan a payloads `{type: ...}` en `ollama_agent/streaming/events.py`.

## Integraciones clave (y dónde tocarlas)
- Tools built-in:
  - Definidas con `@agents.function_tool` en `ollama_agent/agent/builtin_tools.py`.
  - Se exponen añadiéndolas a `BUILTIN_TOOLS`.
  - Timeout de tools: `set_tool_timeout()` (ContextVar) se configura en `ollama_agent/main.py`.
- Model capability gate: antes de crear el agente se valida soporte de tools con `ensure_model_supports_tools()` (`ollama_agent/core/models.py`, usa `ollama.show(model)` y busca capability `"tools"`).
- Screen vision: token `@dpN` en prompts → captura screenshot y lo convierte a input multimodal estilo Responses (`_maybe_attach_screen_context()` en `ollama_agent/agent/agent.py`, helpers en `ollama_agent/vision/screen.py`). Nota: en Linux requiere `DISPLAY`/`WAYLAND_DISPLAY`.
- Memoria persistente (Mem0 + Qdrant): `MemoryManager` en `ollama_agent/memory/memory_manager.py`.
  - Al inicializar, arranca/asegura Qdrant por Docker (`ollama_agent/memory/bootstrap.py`, contenedor `ollama-agent-qdrant-<port>`).
  - Tools de memoria: `mem0_add_memory`, `mem0_search_memory`.
- MCP (opcional): servidores declarados en `~/.ollama-agent/mcp_servers.json`.
  - Lifecycle: `ollama_agent/settings/mcp/lifecycle.py`.
  - Builders: `ollama_agent/settings/mcp/builders.py` (stdio/sse/streamable_http) y crea un “delegated agent” que se expone como tool (`use_<name>` por defecto).

## Convenciones del proyecto
- Preferir `prompt: object` (string o lista multimodal). Si introduces mensajes multimodales, conserva el patrón de callback `RunConfig(session_input_callback=...)` en `OllamaAgent._prepare_input()`.
- Errores en ejecución del agente: `run_async()` retorna un string `"Error: ..."` (no lanza) y el streaming emite payload `{type:"error"}`.
- Task IDs: solo `[A-Za-z0-9_-]+` (valida `TaskManager.validate_task_id`).

## Workflows de dev (reales)
- Setup local:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e .`
- Ejecutar:
  - REPL: `ollama-agent`
  - Prompt único: `ollama-agent -p "..."`
  - Forzar model/effort/timeout: `ollama-agent -m <model> -e <low|medium|high|disabled> -t <segundos> -p "..."`

## Al hacer cambios
- Si cambias opciones/CLI: tocar `ollama_agent/interfaces/cli.py` y mantener compatibilidad con `README.md`.
- Si añades nuevos payload types de streaming: actualizar `ollama_agent/streaming/events.py` + renderer(s) (`ollama_agent/streaming/console_renderer.py` y REPL).
