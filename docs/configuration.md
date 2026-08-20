# Configuration Reference

`ollama-agent` uses a centralized YAML configuration file stored at `~/.ollama-agent/settings.yaml` to manage model parameters, runtime security policies, document retrieval settings, context loading limits, telemetry tracing, and subagent declarations.

---

## Complete `settings.yaml` Reference

```yaml
# ==============================================================================
# Ollama Agent Configuration File
# Location: ~/.ollama-agent/settings.yaml
# ==============================================================================

# Primary LLM Model Settings
model:
  name: "gemma4:26b"                     # Default Ollama model tag
  base_url: "http://localhost:11434"     # Ollama API server endpoint
  temperature: 0.0                       # Sampling temperature (0.0 for deterministic outputs)
  context_window: 10000                  # Context window size in tokens (num_ctx)
  reasoning_effort: "medium"             # Default thinking effort: low, medium, high, disabled, hide, enabled

# Agent Runtime Behavior & Security Policies
runtime:
  allow_traversal: false                 # Allow agent filesystem traversal outside the working directory
  builtin_tool_timeout: 30               # Tool execution timeout in seconds
  collapse_thinking: true                # Automatically collapse thinking blocks in REPL TUI
  inherit_env: false                     # Inherit shell environment variables for executed commands

# RAG (Retrieval-Augmented Generation) Vector Database Settings
rag:
  rag_dir: "~/.ollama-agent/rag"         # Storage directory for RAG databases and vector indices
  embedder_model: "nomic-embed-text:latest" # Embeddings model tag in Ollama
  embedder_base_url: "http://localhost:11434" # Base URL for embeddings server
  embedding_dims: 768                    # Vector dimensionality of the embedding model
  default_top_k: 5                       # Default number of documents retrieved per query
  chunk_size: 500                        # Text chunk size in characters
  chunk_overlap: 50                      # Overlap between consecutive text chunks in characters

# Context Injection Limits (@-mentions)
mentions:
  max_file_size: 1048576                 # Maximum individual file size in bytes (1 MB)
  max_files: 100                          # Maximum number of files attached per directory mention
  max_total_size: 10485760                # Maximum total context payload size in bytes (10 MB)
  max_completions: 200                    # Maximum autocompletion candidates displayed in REPL

# Telemetry & Tracing via LangSmith
langsmith:
  api_key: ""                            # LangSmith API key (e.g. "lsv2_pt_...")
  tracing: "true"                        # Enable LangChain / LangGraph tracing ("true" / "false")
  project: "ollama-agent"                # LangSmith project name
  endpoint: "https://api.smith.langchain.com" # LangSmith API endpoint URL

# Specialized Subagents Configuration
subagents:
  - name: "code_reviewer"
    description: "Specialized subagent for code review and security auditing"
    system_prompt: "You are an expert code reviewer focused on security and clean architecture."
    model: "qwen2.5-coder:32b"
    context_window: 32768
    mcp_servers:
      - name: "git_mcp"
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-git"]
        env:
          PATH: "/usr/bin:/bin"
```

---

## Context Window (`num_ctx`) Resolution Hierarchy

To guarantee optimal context utilization without exceeding model memory boundaries, `ollama-agent` resolves the effective context window size (`num_ctx`) at application startup using a strict priority hierarchy:

```mermaid
flowchart TD
    A[Start Context Resolution] --> B{Explicit Config Override?}
    B -- Yes (`model.context_window`) --> C[Use Configured Value]
    B -- No (`null`) --> D[Fetch Model Metadata via `ollama.show()`]
    D --> E{Structured `model_info` Key?}
    E -- Found `*.context_length` --> F[Use `context_length` Metadata]
    E -- Not Found --> G{Modelfile / Parameter `num_ctx`?}
    G -- Matched via Regex --> H[Use Parsed `num_ctx` Value]
    G -- Not Found --> I[Raise `ModelContextWindowError`]
```

1. **Explicit Configuration Override**: If `model.context_window` in `settings.yaml` (or CLI argument) is explicitly defined, its value is used directly.
2. **Structured Model Metadata (`model_info`)**: Queries Ollama's `AsyncClient.show()` endpoint for modern model metadata keys ending in `.context_length` (e.g., `llama.context_length`, `qwen2.context_length`).
3. **Modelfile Parameter Parsing**: Scans raw Modelfile `parameters` or string fields using regex matching (`^\s*(?:PARAMETER\s+)?num_ctx\s+(\d+)\s*$`) to extract declared `num_ctx` values.
4. **Error Handling**: If resolution fails across all stages, `ollama-agent` halts startup and raises a `ModelContextWindowError`, prompting the user to specify `context_window` in `settings.yaml`.

---

## Model Tool-Calling Capability Verification

`ollama-agent` requires a model capable of native function/tool calling. At initialization, the application verifies model capabilities before starting the runtime:

1. Asynchronously queries `ollama.AsyncClient.show(model)`.
2. Inspects returned model capabilities payload for the `"tools"` tag.
3. If `"tools"` is missing from the capability list, startup terminates immediately with a `ModelCapabilityError`:
   ```
   ModelCapabilityError: Model 'llama2:latest' does not support tools.
   ```

---

## LangSmith Tracing Setup & Environment Injection

`ollama-agent` natively supports LangSmith tracing for monitoring agent workflows, tool execution paths, and LLM latency.

### Setup

Add your credentials to the `langsmith` section in `~/.ollama-agent/settings.yaml`:

```yaml
langsmith:
  api_key: "lsv2_pt_your_api_key_here"
  tracing: "true"
  project: "ollama-agent"
  endpoint: "https://api.smith.langchain.com"
```

### Environment Injection

At runtime, `Settings.setup_environment()` automatically injects non-empty LangSmith settings directly into standard system environment variables:

* `LANGSMITH_API_KEY`
* `LANGSMITH_TRACING`
* `LANGSMITH_PROJECT`
* `LANGSMITH_ENDPOINT`

LangChain and LangGraph automatically pick up these environment variables to send execution traces to your LangSmith dashboard.

---

## System Prompt Customization & Configuration Reset

### System Prompt Files

Agent prompt instructions are managed via Markdown files located in `~/.ollama-agent/`:

* `instructions.md`: Main system instructions governing agent identity, tone, tool usage guidelines, and operational constraints.
* `fs_policy_traversal.md`: Operational policy injected when `--allow-traversal` is enabled (unrestricted filesystem access).
* `fs_policy_sandboxed.md`: Operational policy injected when sandboxed to project boundaries (`--no-allow-traversal`).
* `rag_policy.md`: Operational policy injected dynamically when a RAG database is loaded and active.

If any of these files do not exist during application startup, `ollama-agent` automatically creates them pre-populated with built-in default templates.

### Configuration Reset Options (`--config-reset`)

To restore default configurations or prompts, use the `--config-reset` flag on the CLI:

```bash
ollama-agent --config-reset <option>
```

| Reset Option | Actions Performed |
| :--- | :--- |
| `config-file` | Unlinks `~/.ollama-agent/settings.yaml` and re-initializes it with default settings. |
| `system-prompt` | Unlinks `instructions.md`, `fs_policy_traversal.md`, `fs_policy_sandboxed.md`, and `rag_policy.md` and restores default system prompt templates. |
| `all` | Performs a complete factory reset of both `settings.yaml` and all system prompt policy files. |
