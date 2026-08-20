# Ollama Agent

Ollama Agent is a powerful command-line tool (CLI and REPL) that allows you to interact with local AI models. Built on [DeepAgents](https://docs.langchain.com/oss/python/deepagents/overview) and [LangChain](https://github.com/langchain-ai/langchain), it provides a persistent chat experience, session management, and the ability to execute local shell commands, turning your local models into helpful assistants for your daily tasks.

## Features

- **Interactive REPL**: A modern, terminal-based chat interface with Markdown rendering, multiline input support (`\ + Enter`), live context token metrics, and slash commands.
- **Non-Interactive CLI**: Execute single prompts directly from your command line for quick queries.
- **Native Ollama Integration**: Connects directly to Ollama's native API (via `langchain-ollama`), no OpenAI compatibility layer needed.
- **Thinking / Reasoning**: Leverages Ollama's native [thinking capability](https://docs.ollama.com/capabilities/thinking) to expose model reasoning traces. Configurable per model via `--effort`.
- **Automatic Context Window & Live Monitor**: Resolves effective context window (`num_ctx`) automatically from Ollama metadata and displays real-time token usage and percentage gauge directly in the TUI header.
- **Per-session Model Switching**: Change the model mid-conversation and continue from that point with the new model (context preserved). The change is not permanent and only affects the current session.
- **Tool-Powered**: The agent can execute shell commands via an integrated shell backend, with human-in-the-loop confirmation before running commands and editing files.
- **MCP Integration**: Extend the main agent with [Model Context Protocol](https://modelcontextprotocol.io/) servers (`mcp_servers.json`) that provide additional tools directly to the agent.
- **Custom Subagents**: Define specialized subagents in `settings.yaml` with their own model and MCP servers — each with isolated context for clean delegation.
- **Session Management**: Persistent SQLite-backed session history with full session restoration (`/session resume <id>`), markdown export (`/session export`), listing, and creation of fresh sessions (`/session new`).
- **Task Management**: Save frequently used prompts as "tasks" and execute them with a simple command.
- **Configurable**: Easily configure the model, Ollama host, context window, and reasoning effort.
- **Persistent Memory & Project Guidelines**: Native memory layer backed by `MEMORY.md` and repository-level `AGENTS.md` standard support, allowing the agent to persist long-term context and follow project-specific conventions.
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
- `/model [list | set <model>]`: Manage models (list available models, switch active model).
- `/session [list | resume <id> | new | export [path] | delete <id>]`: Manage chat sessions (list past sessions, resume previous conversation, export to Markdown, delete).
- `/task [list | create <id> | run <id> | delete <id>]`: Manage saved prompt tasks.
- `/skill [list | show <id> | create <id> | delete <id>]`: Manage agent skills.
- `/rag [status | list | create <name> | load <name> | unload | add <path> | delete <name>]`: Manage RAG document databases.
- `/yolo [on|off]`: Toggle YOLO mode or set it explicitly (on/off).
- `/new`: Start a new chat session (clears context).
- `/clear`: Clear the screen.
- `/exit` (or `/quit`): Quit the application.

#### Multiline Input

To enter multiline text or code blocks:
- **Add a new line**: End the line with a backslash `\` and press `Enter` (`\ + Enter`). The trailing backslash is automatically removed and a new line is inserted into the prompt editor.
- **Submit prompt**: Press `Enter` without a trailing backslash to send the message.
- **Navigation**: Use the `↑` and `↓` arrow keys to move the cursor freely between lines. History navigation is triggered only when pressing `↑` at the beginning `(0, 0)` or `↓` at the end of the text.

### Non-Interactive Mode

You can run a single prompt directly from the command line:

```bash
ollama-agent --prompt "List all files in the current directory as JSON."
# Or using the short form:
ollama-agent -p "List all files in the current directory as JSON."
```

### Human-in-the-Loop & YOLO Mode

For security and control, Ollama Agent includes a **Human-in-the-Loop (HITL)** approval flow. By default, before running shell commands (`execute`) or writing/modifying local files (`write_file`, `edit_file`), the agent pauses execution and displays an inline confirmation widget in the terminal:

- **Approve**: Authorize this single tool execution.
- **Reject**: Block the tool execution and send feedback to the agent so it can attempt an alternative approach.
- **Allow Session**: Authorize this execution and automatically approve all future calls for this specific tool type (e.g., all file writes) for the remainder of the current session.
- **Cancel**: Completely abort the tool call and stop agent execution, returning control to the REPL input so you can type new instructions.

#### YOLO Mode

If you trust the agent and want to run tasks without any confirmation prompts, you can enable **YOLO Mode**:

- **CLI Flag**: Start the agent with `-y` or `--yolo` (e.g. `ollama-agent -y` or `ollama-agent --yolo`).
- **Slash Command**: Toggle it dynamically inside the REPL using `/yolo`, or explicitly set it with `/yolo on` and `/yolo off`.

When YOLO mode is active:
1. Confirmations are bypassed entirely.
2. The REPL status bar shows `YOLO: On` (in red).
3. The prompt symbol changes color to **red** (`❯❯ `) to make it highly visible.

### Live Context Usage & Token Monitoring

The REPL dynamic header displays the current model, active RAG database, YOLO status, and live context window consumption in real-time:

```text
● ollama-agent │ Model: gemma4:26b │ Context: 2.1k/16.4k (12%) │ Effort: medium │ YOLO: OFF
```

- **Tokens & Percentage**: Shows current tokens consumed versus effective context window limit (`num_ctx`).
- **Dynamic Color Alerts**:
  - 🔵 **Blue/Cyan**: Healthy context usage (<75%).
  - 🟡 **Yellow**: Elevated context warning (>75%).
  - 🔴 **Red**: Critical context limit proximity (>90%).

### File / Directory Context (@-mentions)

Ollama Agent supports referencing files or directories directly in your prompt using the `@` symbol, automatically loading their content into the agent's context.

*   **Single File**: `@filename.txt` or `@"file name with spaces.txt"`
*   **Directory Traversal**: `@src` or `@.` (recursively reads all text files in the directory).
*   **Autocompletion**: In the REPL, type `@` and hit `Tab` to interactively autocomplete file and folder paths in the terminal!

#### File Formats
Text files are attached as raw text blocks, and multimodal files (such as images and audio) are base64-encoded and attached as native multimodal inputs. Other binary files (such as `.zip` or executables) are automatically skipped during directory traversal.

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
- `-e`, `--effort`: Set reasoning effort level (`low`, `medium`, `high`, `disabled`, `hide`, `enabled`)
- `-t`, `--builtin-tool-timeout`: Set tool-call timeout in seconds (applies to tool executions, including shell backend and built-in tools). Overrides `builtin_tool_timeout` from `settings.yaml` for the current run.
- `-y`, `--yolo`: Enable YOLO mode (bypasses all tool execution confirmation prompts)
- `--rag <database>`: Load a RAG database for the session
- `--allow-traversal`: Allow virtual filesystem traversal to OS directories outside the project root
- `--no-allow-traversal`: Sandbox agent to project directory (default)
- `--config-reset <all|system-prompt|config-file>`: Reset configuration or system prompts to defaults

## Session Management

`ollama-agent` features complete SQLite-backed persistent session management across conversations.

### Commands

| Command (REPL) | Command (CLI) | Description |
| :--- | :--- | :--- |
| `/session list` (or `/session`) | `ollama-agent session-list` | List all saved chat sessions with step counts and active session indicator. |
| `/session resume <id>` | — | Resume a past session by ID or prefix, restoring conversation messages directly in the terminal viewport. |
| `/session new` (or `/new`) | — | Start a clean new session with a new thread ID and fresh context. |
| `/session export [path]` | `ollama-agent session-export <id> [-o path]` | Export the entire conversation history to a structured Markdown file. |
| `/session delete <id>` | `ollama-agent session-delete <id>` | Delete a session and its saved checkpoints from history. |

### Interactive Autocompletion

Inside the REPL, typing `/session resume ` or `/session delete ` dynamically lists and autocompletes available session IDs in real-time.

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
/task create <task_id>
```

The REPL will open an interactive modal dialog with fields for Task ID, Title, Model, Reasoning Effort, and a multiline prompt editor with Cancel/Create buttons.

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
# or inside REPL: /task list (or /task)
```

**Run a Task:**

Use the task ID (or a unique prefix) from the list to run it.

```bash
ollama-agent task-run <task_id>
# or inside REPL: /task run <task_id>
```

**Delete a Task:**

```bash
ollama-agent task-delete <task_id>
# or inside REPL: /task delete <task_id>
```

## Configuration

On the first run, the application will create a default configuration file at `~/.ollama-agent/settings.yaml`. You can edit this file to permanently change the default model, Ollama host, and other settings.

Example default `settings.yaml`:

```yaml
model:
  name: gemma4:26b
  base_url: http://localhost:11434
  temperature: 0.0
  context_window: 10000
  reasoning_effort: medium
runtime:
  allow_traversal: false
  builtin_tool_timeout: 30
  collapse_thinking: true
  inherit_env: false
rag:
  rag_dir: /home/user/.ollama-agent/rag
  embedder_model: nomic-embed-text:latest
  embedder_base_url: http://localhost:11434
  embedding_dims: 768
  default_top_k: 5
  chunk_size: 500
  chunk_overlap: 50
mentions:
  max_file_size: 1048576
  max_files: 100
  max_total_size: 10485760
  max_completions: 200
subagents: []
```

| Key | Description |
|---|---|
| `model.name` | Default Ollama model. Must support tool calling (default: `gemma4:26b`). |
| `model.base_url` | Native Ollama host (e.g. `http://localhost:11434`). Must **not** contain an `/v1` path. |
| `model.reasoning_effort` | Default thinking level: `low`, `medium`, `high`, `disabled`, `hide`, or `enabled`. See [Thinking / Reasoning](#common-options) above. |
| `model.context_window` | Context window size in tokens (`num_ctx`) (default: `10000`). |
| `runtime.allow_traversal` | If true, permits virtual filesystem traversal outside the working directory (default: `false`). |
| `runtime.builtin_tool_timeout` | Timeout in seconds for tool executions (default: `30`). |
| `runtime.collapse_thinking` | If true, collapses model reasoning/thinking blocks in REPL output by default (default: `true`). |
| `runtime.inherit_env` | If true, local shell commands execute with the parent's full environment variables (e.g. PATH). |

### Context Window Resolution

Ollama Agent needs to know the effective context window (`num_ctx`) for every model. The runtime resolves it in this order:

1. `model.context_window` from `settings.yaml` (or CLI override), if defined.
2. The model's reported `*.context_length` metadata from `ollama show <model>` (e.g. `llama.context_length`, `qwen2.context_length`).
3. `PARAMETER num_ctx` regex parsed from the model's Modelfile / parameters via `ollama show <model>`.

If none of those sources provides a value, the app exits with a clear error asking you to set `model.context_window` in `settings.yaml`.

### Configuration Reset

If you need to reset the configuration or system prompt files to their default values, you can use the `--config-reset` flag:

```bash
# Reset all configuration files and prompt files
ollama-agent --config-reset all

# Reset system prompts (instructions.md, fs_policy_traversal.md, fs_policy_sandboxed.md)
ollama-agent --config-reset system-prompt

# Reset only the settings (settings.yaml)
ollama-agent --config-reset config-file
```

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

## Persistent Memory & Project Guidelines (`AGENTS.md`)

`ollama-agent` incorporates both persistent user memory and repository-level project instructions:

### 1. Project Guidelines (`AGENTS.md`)
The open **`AGENTS.md`** standard provides repository-specific guidelines, testing commands, and coding conventions directly to AI coding agents without manual prompt repetition.

- **Automatic Discovery**: When starting a session or executing commands, `ollama-agent` searches for `AGENTS.md` (or `agents.md`, `.agents.md`) in the current working directory. If not found, it ascends through parent directories up to the repository root (marked by `.git`).
- **Native Context Injection**: Discovered guidelines are automatically injected into the agent's memory context at session startup.
- **Example `AGENTS.md`**:

```markdown
# Project Guidelines

## Development & Test Commands
- Run unit tests: `pytest`
- Run linter: `flake8`

## Coding Conventions
- Strictly follow PEP 8 standards with top-level imports.
- Keep functions small and focused on a single responsibility.
- Do not use fallback chains to hide missing state.
```

### 2. Long-Term User Memory (`~/.ollama-agent/MEMORY.md`)
The agent manages its own cross-session memory file at `~/.ollama-agent/MEMORY.md`. The agent reads and updates this file automatically to persist user preferences, architectural decisions, and context across sessions.

### 3. Global Agent Guidelines (`~/.ollama-agent/AGENTS.md`)
You can optionally place an `AGENTS.md` file in `~/.ollama-agent/AGENTS.md` to define user-level global guidelines that are loaded across all repositories.




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
/rag create my-docs
```

**List Databases:**

```bash
ollama-agent rag-list
# or inside REPL: /rag list
```

**Delete a Database:**

```bash
ollama-agent rag-delete my-docs
# or inside REPL: /rag delete my-docs
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
/rag load my-docs
/rag add /path/to/document.md
/rag add /path/to/folder --dir
```

Supported file types include: `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.sh`, `.yaml`, `.yml`, `.json`, `.xml`, `.md`, `.txt`, `.toml`, `.c`, `.cpp`, `.h`, `.hpp`, `.go`, `.rs`, `.css`, `.html`, `.sql`, `.ini`, `.cfg`, `.properties`, `.java`, `.kt`, `.gradle`, `.bat`, `.ps1`, `.csv`, `.rst`

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
/rag load another-db
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

- `rag_dir`: Directory where RAG databases are stored (defaults to `~/.ollama-agent/rag`)
- `embedder_model`: Ollama model used for generating embeddings (default: `nomic-embed-text:latest`)
- `embedder_base_url`: Ollama host for generating embeddings (default: `http://localhost:11434`)
- `embedding_dims`: Dimension of the embedding vectors (must match the model, default: `768`)
- `default_top_k`: Default number of results to return in searches (default: `5`)
- `chunk_size`: Maximum size of text chunks in characters (default: `500`)
- `chunk_overlap`: Overlap between consecutive chunks in characters (default: `50`)

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

### Skill Sources

Skills are loaded from the global skills directory:

- **Global Skills Directory**: `~/.ollama-agent/skills/` — user-level skills available across sessions.

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
# or inside REPL: /skill list (or /skill)
```

**Show Skill Details:**

```bash
ollama-agent skill-show langgraph-docs
# or inside REPL: /skill show langgraph-docs
```

**Delete a Skill:**

```bash
ollama-agent skill-delete langgraph-docs
# or inside REPL: /skill delete langgraph-docs
```

### Managing Skills (REPL)

Inside the REPL you can create skills interactively:

```text
/skill create my-skill
```

The REPL will open an interactive modal dialog with fields for Skill ID, Name, Description, and a multiline markdown instructions editor with Cancel/Create buttons.

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

You can customize the agent's behavior by editing the instructions file at `~/.ollama-agent/prompts/instructions.md`. This file is automatically created on first use with default instructions.

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

Define specialized subagents that your main agent can delegate tasks to. Each subagent has its own isolated context, model, and MCP servers — keeping the orchestrator's context clean and focused.

Configure them in `~/.ollama-agent/settings.yaml`:

```yaml
subagents:
  - name: "research-agent"
    description: "Delegate here for complex research or web searches."
    system_prompt: "You are a research specialist. Search thoroughly and return concise summaries."
    model: "gemma4:26b"          # Optional, inherits from main agent
    context_window: 65536        # Optional, inherits from main agent
    mcp_servers:
      - name: "brave-search"
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-brave-search"]
        env:
          BRAVE_API_KEY: "${BRAVE_API_KEY}"

  - name: "database-agent"
    description: "Delegate here when the user asks about customer or sales data."
    system_prompt: "You are a database analyst. Query the database and summarize results."
    mcp_servers:
      - name: "sqlite-server"
        command: "uvx"
        args: ["mcp-server-sqlite", "--db-path", "./data/ventas.db"]
```

- **Context isolation**: Subagent tool calls don't bloat the main agent's context — only the final result is returned.
- **Environment injection**: Use `${VAR_NAME}` in MCP `env` fields to inject secrets from the host environment.
- **MCP per subagent**: Each subagent can have its own dedicated stdio MCP servers, completely independent from the main agent.
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
