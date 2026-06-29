# Ollama Agent

Ollama Agent is a powerful command-line tool (CLI and REPL) that allows you to interact with local AI models. Built on [DeepAgents](https://docs.langchain.com/oss/python/deepagents/overview) and [LangChain](https://github.com/langchain-ai/langchain), it provides a persistent chat experience, session management, and the ability to execute local shell commands, turning your local models into helpful assistants for your daily tasks.

## Features

- **Interactive REPL**: A modern, terminal-based chat interface with Markdown rendering and slash commands.
- **Non-Interactive CLI**: Execute single prompts directly from your command line for quick queries.
- **Native Ollama Integration**: Connects directly to Ollama's native API (via `langchain-ollama`), no OpenAI compatibility layer needed.
- **Thinking / Reasoning**: Leverages Ollama's native [thinking capability](https://docs.ollama.com/capabilities/thinking) to expose model reasoning traces. Configurable per model via `--effort`.
- **Automatic Context Window**: Resolves the model's effective context window (`num_ctx`) automatically from Ollama metadata, or allows manual override in config.
- **Per-session Model Switching**: Change the model mid-conversation and continue from that point with the new model (context preserved). The change is not permanent and only affects the current session.
- **Tool-Powered**: The agent can execute shell commands via an integrated shell backend, allowing it to interact with your local environment to perform tasks.
- **MCP Integration**: Extend the main agent with [Model Context Protocol](https://modelcontextprotocol.io/) servers (`mcp_servers.json`) that provide additional tools as isolated subagents.
- **Custom Subagents**: Define specialized subagents in `settings.yaml` with their own model, skills, and MCP servers — each with isolated context for clean delegation.
- **Session Management**: Conversations are automatically saved and can be reloaded, deleted, or switched between.
- **Task Management**: Save frequently used prompts as "tasks" and execute them with a simple command.
- **Configurable**: Easily configure the model, Ollama host, context window, and reasoning effort.
- **Persistent Memory**: Native memory layer backed by `MEMORY.md`, allowing the agent to persist long-term context across sessions.
- **RAG (Retrieval Augmented Generation)**: Create and manage document databases for context-aware responses using local embeddings and Qdrant.
- **Skills**: Extend the agent with reusable, on-demand capabilities via the [Agent Skills specification](https://agentskills.io/specification). Skills provide task-specific instructions and context through progressive disclosure.
- **File/Directory Context (@-mentions)**: Reference local files or directories directly in your prompts (e.g. `@src/main.py` or `@.`). Type `@` in the REPL to interactively autocomplete file and folder paths in the terminal!

## Prerequisites (Important)

Before installing/running the app, make sure you have:

- **Ollama (or compatible API) running**.
- **A model that supports tool calling** (required). If the selected model does not support tools/function-calling, the app will exit.
- **The embeddings model downloaded in Ollama**. By default, RAG uses `nomic-embed-text:latest`.

```bash
# Required embeddings model (default for RAG)
ollama pull nomic-embed-text:latest
```

## Installation

For end-users, the recommended way to install `ollama-agent` is using `pipx`, which installs the application in an isolated environment.

```bash
# Install from GitHub
pipx install git+https://github.com/arrase/ollama-agent.git
```

## Quick Start

Start the interactive REPL:

```bash
ollama-agent
```

Or run a single prompt (non-interactive):

```bash
ollama-agent -p "List all files in the current directory as JSON."
```

## Usage

### Interactive Mode (REPL)

To start the chat interface, simply run:

```bash
ollama-agent
```

The REPL provides a persistent chat session. You can use slash commands to manage the session:

- `/help`: Show available commands.
- `/new`: Start a new chat session (clears context).
- `/clear`: Clear the screen.
- `/models`: List available Ollama models (shows tool support).
- `/model-set <model>`: Switch to a different model (conversation preserved).
- `/sessions`: List saved sessions.
- `/session-load <id>`: Load a saved session.
- `/session-delete <id>`: Delete a saved session.
- `/tasks`: List saved tasks.
- `/task-run <id>`: Run a specific task.
- `/task-delete <id>`: Delete a specific task.
- `/rag`: Show current RAG database status.
- `/rag-list`: List available RAG databases.
- `/rag-create <name>`: Create a new RAG database.
- `/rag-load <name>`: Load a RAG database for the session.
- `/rag-unload`: Unload the current RAG database.
- `/rag-add <path>`: Add a file to the loaded RAG database.
- `/rag-add <path> --dir`: Add all files from a directory.
- `/rag-delete <name>`: Delete a RAG database.
- `/skills`: List all skills.
- `/skill-show <id>`: Show skill details.
- `/skill-create <id>`: Create a skill interactively.
- `/skill-delete <id>`: Delete a skill.
- `/exit`: Quit the application.

### Non-Interactive Mode

You can run a single prompt directly from the command line:

```bash
ollama-agent --prompt "List all files in the current directory as JSON."
# Or using the short form:
ollama-agent -p "List all files in the current directory as JSON."
```

### File / Directory Context (@-mentions)

Ollama Agent supports referencing files or directories directly in your prompt using the `@` symbol, automatically loading their content into the agent's context.

*   **Single File**: `@filename.txt` or `@"file name with spaces.txt"`
*   **Directory Traversal**: `@src` or `@.` (recursively reads all text files in the directory).
*   **Autocompletion**: In the REPL, type `@` and hit `Tab` to interactively autocomplete file and folder paths in the terminal!

#### Ignored Paths and Formats
To protect the model's context window, directory traversal automatically filters out system and build directories (such as `.git`, `.venv`, `node_modules`, `build`, etc.) and skips binary files.

#### Safety Limits
The following default safety limits are enforced to avoid overloaded contexts. They can be customized in `~/.ollama-agent/settings.yaml` under the `mentions` key:

```yaml
mentions:
  max_file_size: 1048576      # Max individual file size in bytes (default: 1 MB)
  max_files: 100               # Max files in a directory mention (default: 100)
  max_total_size: 10485760     # Max total attached context in bytes (default: 10 MB)
  max_completions: 200         # Max autocompletion candidates (default: 200)
```

#### Decorator & Mention Safety
To avoid false positives, words starting with `@` that do not exist (like Python's `@decorator` syntax or `@staticmethod`) are ignored and treated as literal text. However, if a nonexistent path contains path separators (e.g. `@src/mainn.py`) or standard file extensions (e.g. `@file.py`), the agent will halt and report a `File or directory not found` error so you can correct it.

### Common Options

You can override the configured model, reasoning effort, or tool execution timeout:

```bash
ollama-agent --model "gpt-oss:20b" --effort "high" --prompt "What is the current date?"
# Or using short forms:
ollama-agent -m "gpt-oss:20b" -e "high" -p "What is the current date?"
```

**Thinking / Reasoning effort** — the `--effort` flag maps to Ollama's native [`think` parameter](https://docs.ollama.com/capabilities/thinking). Thinking-capable models emit a `thinking` field that separates their reasoning trace from the final answer.

| Model family | `--effort` value | Ollama `think` value | Behaviour |
|---|---|---|---|
| **GPT-OSS** | `low` / `medium` / `high` | `"low"` / `"medium"` / `"high"` | Sets the thinking trace length. GPT-OSS only accepts these levels; `true`/`false` is ignored. |
| **GPT-OSS** | `disabled` / `hide` | *(not sent)* | GPT-OSS cannot fully disable thinking. For `disabled`, a warning is emitted and the default level is used. In both cases, the thinking trace is hidden from the UI. |
| **GPT-OSS** | `enabled` | `"medium"` | Enables thinking using the default `medium` effort level. |
| **Other thinking models** (Qwen 3, DeepSeek R1, DeepSeek-v3.1, …) | `low` / `medium` / `high` / `enabled` | `true` | Enables thinking. The specific levels are ignored by Ollama but turn on thinking. |
| **Other thinking models** | `hide` | `true` | The model generates the reasoning trace, but it is hidden from the UI output. |
| **Other thinking models** | `disabled` | `false` | Disables thinking at the model level. |
| **Non-thinking models** | *(any)* | *(not sent)* | Setting is ignored. |

Thinking is enabled by default in Ollama for supported models. See the [Ollama thinking docs](https://docs.ollama.com/capabilities/thinking) for the full list of supported models and API details.

```bash
ollama-agent --builtin-tool-timeout 60 --prompt "Run a long-running task"
# Or using short forms:
ollama-agent -t 60 -p "Run a long-running task"
```

**Available Parameters:**

- `-m`, `--model`: Specify the AI model to use
- `-p`, `--prompt`: Provide a prompt for non-interactive mode
- `-e`, `--effort`: Set reasoning effort level (low, medium, high, disabled, hide, enabled)
- `-t`, `--builtin-tool-timeout`: Set tool-call timeout in seconds (applies to tool executions, including shell backend and built-in tools). Overrides `builtin_tool_timeout` from `config.ini` for the current run.
- `--rag <database>`: Load a RAG database for the session
- `--skills-dir <dir>`: Additional skills directory (can be repeated to add multiple sources)

## Tasks

Tasks are saved prompts that can be executed repeatedly.

**Create a Task (CLI):**

```bash
ollama-agent task-create <task_id> \
    --title "My task title" \
    --task-prompt "Do the thing" \
    --task-model "gpt-oss:20b" \
    --task-effort "medium"
```

- Use `--force` to overwrite an existing task.
- `task_id` must be filesystem-safe (letters, numbers, `_`, `-`).

**Create a Task (REPL):**

Inside the REPL:

```text
/task-create <task_id>
```

The REPL will prompt you for title/model/effort and then lets you enter a **multiline** prompt (finish with Esc+Enter).

**Create a Task (manual YAML):**

Tasks are stored as YAML files in `~/.ollama-agent/tasks/`. To create one, add a new file named `<task_id>.yaml` in that directory.

- `<task_id>` can be any filesystem-safe ID (it will show up in `task-list` and is what you pass to `task-run`).
- The YAML supports: `title`, `prompt`, `model`, and (optionally) `reasoning_effort`.

Example:

```yaml
title: "List repo tree"
prompt: "List all files in this repository as a tree."
model: "gpt-oss:20b"
reasoning_effort: "medium"  # low|medium|high|disabled|hide|enabled
```

**List Tasks:**

```bash
ollama-agent task-list
# or inside REPL: /tasks
```

**Run a Task:**

Use the task ID (or a unique prefix) from the list to run it.

```bash
ollama-agent task-run <task_id>
# or inside REPL: /task-run <task_id>
```

**Delete a Task:**

```bash
ollama-agent task-delete <task_id>
# or inside REPL: /task-delete <task_id>
```

## Configuration

On the first run, the application will create a default configuration file at `~/.ollama-agent/settings.yaml`. You can edit this file to permanently change the default model, Ollama host, and other settings.

Example default `settings.yaml`:

```yaml
model:
  name: qwen3.5:9b
  base_url: http://localhost:11434
  temperature: 0.0
  context_window: null
  reasoning_effort: medium
runtime:
  allow_traversal: true
  builtin_tool_timeout: 30
rag:
  rag_dir: /home/arrase/.ollama-agent/rag
  embedder_model: nomic-embed-text:latest
  embedder_base_url: http://localhost:11434
  embedding_dims: 768
  default_top_k: 5
  chunk_size: 500
  chunk_overlap: 50
```

| Key | Description |
|---|---|
| `model.name` | Default Ollama model. Must support tool calling. |
| `model.base_url` | Native Ollama host (e.g. `http://localhost:11434`). Must **not** contain an `/v1` path. |
| `model.reasoning_effort` | Default thinking level: `low`, `medium`, `high`, `disabled`, `hide`, or `enabled`. See [Thinking / Reasoning](#common-options) above. |
| `model.context_window` | If set, forces the runtime `num_ctx` for the selected model. Leave `null` to let the app resolve it automatically. |
| `runtime.builtin_tool_timeout` | Timeout in seconds for tool executions. |

### Context Window Resolution

Ollama Agent needs to know the effective context window (`num_ctx`) for every model. The runtime resolves it in this order:

1. `context_window` from `config.ini`, if defined.
2. `PARAMETER num_ctx` from `ollama show <model>` (the model's Modelfile).
3. The model's reported `*.context_length` metadata from `ollama show <model>`.

If none of those sources provides a value, the app exits with a clear error asking you to set `model.context_window` in `settings.yaml`.

### Configuration Reset

If you need to reset the configuration or system prompt to their default values, you can use the `--config-reset` flag:

```bash
# Reset all configuration files
ollama-agent --config-reset all

# Reset only the system prompt (instructions.md)
ollama-agent --config-reset system-prompt

# Reset only the settings (settings.yaml)
ollama-agent --config-reset config-file
```

> **Note**: When upgrading from v0.1 to v0.2, it is recommended to reset the system prompt to ensure compatibility with new features:
> `ollama-agent --config-reset system-prompt`

## LangSmith Tracing

Ollama Agent supports native tracing via [LangSmith](https://docs.smith.langchain.com/). To enable it, simply add the `langsmith` section to your `~/.ollama-agent/settings.yaml`:

```yaml
langsmith:
  api_key: "your-api-key"
  tracing: "true"
  project: "your-project-name"
  endpoint: "https://api.smith.langchain.com" # Optional, useful for EU or specific regions
```

When configured, the agent will automatically inject these values into the environment upon startup, enabling deep tracing of tool executions, reasoning steps, and agent workflows. If omitted, no environment variables will be injected and tracing will remain disabled.

## Persistent Memory

The agent manages its own long-term memory via a file located at `~/.ollama-agent/MEMORY.md`. The agent uses built-in tools to read, update, and persist facts, preferences, and context across sessions automatically.

## RAG (Retrieval Augmented Generation)

RAG allows the agent to search through your documents and use relevant context when answering questions. Documents are chunked, embedded using Ollama, and stored in local Qdrant databases.

### RAG Databases

RAG databases are stored in `~/.ollama-agent/rag/<name>/`. Each database is independent and can contain documents from different sources.

**Create a Database (CLI):**

```bash
ollama-agent rag-create my-docs
```

**Create a Database (REPL):**

```text
/rag-create my-docs
```

**List Databases:**

```bash
ollama-agent rag-list
# or inside REPL: /rag-list
```

**Delete a Database:**

```bash
ollama-agent rag-delete my-docs
# or inside REPL: /rag-delete my-docs
```

### Adding Documents

Before adding documents, you need to load a database (in REPL) or specify it in the command (CLI).

**Add a Single File (CLI):**

```bash
ollama-agent rag-add my-docs /path/to/document.md
```

**Add a Directory (CLI):**

```bash
ollama-agent rag-add my-docs /path/to/folder --dir
```

**Add Files (REPL):**

First load the database, then add files:

```text
/rag-load my-docs
/rag-add /path/to/document.md
/rag-add /path/to/folder --dir
```

Supported file types include: `.txt`, `.md`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.yml`, `.html`, `.css`, `.xml`, `.csv`, `.rst`, `.ini`, `.cfg`, `.sh`

### Searching Documents

Manual query commands have been removed from both CLI and REPL. Load a RAG database and ask your question normally — the agent will use the `rag_search` tool automatically when it needs document context.

### Using RAG with Prompts

Once a RAG database is loaded, the agent can automatically search it using the `rag_search` tool, which returns both formatted context and detailed results with relevance scores.

**Start REPL with RAG:**

```bash
ollama-agent --rag my-docs
```

**Use RAG in Non-Interactive Mode:**

```bash
ollama-agent --rag my-docs -p "What does the documentation say about configuration?"
```

**Switch RAG Database (REPL):**

```text
/rag-load another-db
```

### Configure RAG

RAG settings are located in `~/.ollama-agent/settings.yaml` under the `rag` section:

```yaml
rag:
  rag_dir: /home/user/.ollama-agent/rag
  embedder_model: nomic-embed-text:latest
  embedder_base_url: http://localhost:11434
  embedding_dims: 768
  default_top_k: 5
  chunk_size: 500
  chunk_overlap: 50
```

- `rag_dir`: Directory where RAG databases are stored
- `embedder_model`: Ollama model used for generating embeddings
- `embedding_dims`: Dimension of the embedding vectors (must match the model)
- `default_top_k`: Default number of results to return in searches
- `chunk_size`: Maximum size of text chunks (in characters)
- `chunk_overlap`: Overlap between consecutive chunks

## Skills

Skills are reusable agent capabilities that provide specialized workflows and domain knowledge. They follow the [Agent Skills specification](https://agentskills.io/specification) and are powered by [DeepAgents skills](https://docs.langchain.com/oss/python/deepagents/skills).

When a prompt arrives, the agent checks skill descriptions to find relevant ones. Only when a skill matches does the agent read the full instructions — this pattern is called *progressive disclosure* and keeps the system prompt lean.

### Skill Structure

Each skill is a directory containing at least a `SKILL.md` file with YAML frontmatter:

```text
~/.ollama-agent/skills/
├── langgraph-docs/
│   └── SKILL.md
└── arxiv-search/
    ├── SKILL.md
    └── arxiv_search.py
```

Example `SKILL.md`:

```markdown
---
name: langgraph-docs
description: Use this skill for requests related to LangGraph in order to fetch relevant documentation to provide accurate, up-to-date guidance.
---

# langgraph-docs

## Overview

This skill explains how to access LangGraph Python documentation.

## Instructions

1. Fetch the documentation index using the fetch_url tool.
2. Select 2-4 most relevant documentation URLs.
3. Fetch selected documentation.
4. Provide accurate guidance based on the docs.
```

Additional files (scripts, templates, docs) can be placed alongside `SKILL.md` — just reference them in the instructions so the agent knows when and how to use them.

### Skill Sources and Precedence

Skills are loaded from multiple directories in order (last wins for same-name skills):

1. **Global**: `~/.ollama-agent/skills/` — user-level skills available in every session.
2. **Project**: `./skills/` — project-specific skills in the current working directory.
3. **CLI extra**: directories passed via `--skills-dir`.

```bash
# Load additional skill sources
ollama-agent --skills-dir /path/to/team-skills --skills-dir /path/to/project-skills -p "Help me with LangGraph"
```

### Managing Skills (CLI)

**Create a Skill:**

```bash
ollama-agent skill-create langgraph-docs \
    --name "LangGraph Docs" \
    --description "Fetch relevant LangGraph documentation" \
    --instructions "Use fetch_url to read https://docs.langchain.com/llms.txt and select relevant pages."
```

Use `--force` to overwrite an existing skill.

**List Skills:**

```bash
ollama-agent skill-list
# or inside REPL: /skills
```

**Show Skill Details:**

```bash
ollama-agent skill-show langgraph-docs
# or inside REPL: /skill-show langgraph-docs
```

**Delete a Skill:**

```bash
ollama-agent skill-delete langgraph-docs
# or inside REPL: /skill-delete langgraph-docs
```

### Managing Skills (REPL)

Inside the REPL you can create skills interactively:

```text
/skill-create my-skill
```

The REPL will prompt for name, description, and then open a multiline editor for instructions (finish with Esc+Enter).

### Creating Skills Manually

You can also create skills by hand — just create a directory under `~/.ollama-agent/skills/` with a `SKILL.md` file:

```bash
mkdir -p ~/.ollama-agent/skills/my-skill
cat > ~/.ollama-agent/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: A custom skill that does something useful.
---

# my-skill

## Instructions

Your instructions here.
EOF
```

### Tips

- Write clear, specific descriptions — the agent decides whether to use a skill based on the description alone.
- `SKILL.md` files must be under 10 MB; larger files are skipped.
- Descriptions longer than 1024 characters are truncated.
- Skills directories that don't exist are silently ignored.

## Agent Instructions

You can customize the agent's behavior by editing the instructions file at `~/.ollama-agent/instructions.md`. This file is automatically created on first use with default instructions.

## MCP Servers (Main Agent)

The main agent supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) to extend its capabilities with additional tools. MCP servers configured in `~/.ollama-agent/mcp_servers.json` provide their tools **directly to the main agent** — no subagent wrapping.

### MCP Configuration

Create `~/.ollama-agent/mcp_servers.json`:

```json
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"]
        },
        "brave-search": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {
                "BRAVE_API_KEY": "your-key-here"
            }
        },
        "remote-api": {
            "url": "http://localhost:8000/mcp"
        }
    }
}
```

Supported transports:
- **stdio**: Set `command` (and optionally `args`, `env`) to launch a subprocess.
- **http**: Set `url` to connect to a remote MCP server.

All tools from all configured servers are loaded and made directly available to the main agent. If a server fails to connect, it is skipped and the agent continues normally.

## 🤖 Custom Subagents

Define specialized subagents that your main agent can delegate tasks to. Each subagent has its own isolated context, model, skills, and MCP servers — keeping the orchestrator's context clean and focused.

Configure them in `~/.ollama-agent/settings.yaml`:

```yaml
subagents:
  - name: "research-agent"
    description: "Delegate here for complex research or web searches."
    system_prompt: "You are a research specialist. Search thoroughly and return concise summaries."
    model: "gemma4:26b"          # Optional, inherits from main agent
    context_window: 65536        # Optional, inherits from main agent
    skills_paths:
      - "./skills/research"
    mcp_servers:
      - name: "brave-search"
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-brave-search"]
        env:
          BRAVE_API_KEY: "${BRAVE_API_KEY}"

  - name: "database-agent"
    description: "Delegate here when the user asks about customer or sales data."
    system_prompt: "You are a database analyst. Query the database and summarize results."
    skills_paths:
      - "./skills/database"
    mcp_servers:
      - name: "sqlite-server"
        command: "uvx"
        args: ["mcp-server-sqlite", "--db-path", "./data/ventas.db"]
```

- **Context isolation**: Subagent tool calls don't bloat the main agent's context — only the final result is returned.
- **Environment injection**: Use `${VAR_NAME}` in MCP `env` fields to inject secrets from the host environment.
- **Skills & MCP per subagent**: Each subagent can have its own skills directories and MCP server connections, completely independent from the main agent.
- **Graceful failures**: If a subagent's MCP server fails to load (e.g., missing env vars), it is skipped and the agent continues normally.

## For Developers

Interested in contributing? Great! Here’s how to get started.

### Project Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/arrase/ollama-agent.git
    cd ollama-agent
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. **Install in editable mode:**

    This will install the project and its dependencies. The `-e` flag allows you to make changes to the source code and have them immediately reflected.

    ```bash
    pip install -e .
    ```

### Project Structure

- `ollama_agent/main.py`: Main application entry point.
- `ollama_agent/interfaces/`: CLI and REPL interface implementations.
- `ollama_agent/agent/`: Core agent logic (DeepAgents graph), session management, and built-in tools.
- `ollama_agent/core/`: Shared types, model capability checks, and common utilities.
- `ollama_agent/tasks/`: Task management system.
- `ollama_agent/skills/`: Skills management and DeepAgents skills integration.
- `ollama_agent/rag/`: RAG implementation for context retrieval.
- `ollama_agent/mcp/`: MCP server lifecycle and integration helpers.
- `ollama_agent/streaming/`: Console output streaming, rendering, and non-interactive runner.
- `ollama_agent/settings/`: Application configuration and centralized filesystem paths.
