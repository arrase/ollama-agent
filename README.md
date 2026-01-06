# Ollama Agent

Ollama Agent is a powerful command-line tool (CLI and REPL) that allows you to interact with local AI models through an Ollama-compatible API. It provides a persistent chat experience, session management, and the ability to execute local shell commands, turning your local models into helpful assistants for your daily tasks.

## Features

- **Interactive REPL**: A modern, terminal-based chat interface with Markdown rendering and slash commands.
- **Non-Interactive CLI**: Execute single prompts directly from your command line for quick queries.
- **Ollama Integration**: Connects to any Ollama-compatible API endpoint.
- **Per-session Model Switching**: Change the model mid-conversation and continue from that point with the new model (context preserved). The change is not permanent and only affects the current session.
- **Screen Vision (Screenshots)**: Attach monitor screenshots in prompts using `@dpN` for visual context.
- **Tool-Powered**: The agent can execute shell commands, allowing it to interact with your local environment to perform tasks.
- **Delegated MCP Agents**: Each configured MCP server can run through its own lightweight agent with custom model and instructions.
- **Session Management**: Conversations are automatically saved and can be reloaded, deleted, or switched between.
- **Task Management**: Save frequently used prompts as "tasks" and execute them with a simple command.
- **Configurable**: Easily configure the model, API endpoint, and agent reasoning effort.
- **Mem0 Memory Layer**: Persistent memory backed by Mem0 + Qdrant, exposed through function-calling tools.
- **RAG (Retrieval Augmented Generation)**: Create and manage document databases for context-aware responses using local embeddings and Qdrant.

## Prerequisites (Important)

Before installing/running the app, make sure you have:

- **Ollama (or compatible API) running**.
- **A model that supports tool calling** (required). If the selected model does not support tools/function-calling, the app will exit.
- **The embeddings model downloaded in Ollama**. By default, Mem0 and RAG use `nomic-embed-text:latest`.
- **Vision-capable model (optional)**: only required if you want to use Screen Vision (`@dpN`). If your model does not support vision, the app will still work but it won't be able to "see" screenshots.

```bash
# Required embeddings model (default for Mem0 and RAG)
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
- `/rag-search <query>`: Search the loaded RAG database.
- `/rag-delete <name>`: Delete a RAG database.
- `/exit`: Quit the application.

### Non-Interactive Mode

You can run a single prompt directly from the command line:

```bash
ollama-agent --prompt "List all files in the current directory as JSON."
# Or using the short form:
ollama-agent -p "List all files in the current directory as JSON."
```

### Screen Vision (Screenshots)

Screen vision is not limited to a specific mode: it works anywhere you can type a prompt (both REPL and CLI).

Attach a screenshot of a monitor as context by including `@dpN` in your prompt (`N` is a 0-based monitor index):

```bash
ollama-agent -p "Describe what you see in @dp0"
```

If you include multiple tokens (e.g. `@dp0 @dp1`), the agent will capture and attach each requested monitor.

### Common Options

You can override the configured model, reasoning effort, or built-in tool execution timeout:

```bash
ollama-agent --model "gpt-oss:20b" --effort "high" --prompt "What is the current date?"
# Or using short forms:
ollama-agent -m "gpt-oss:20b" -e "high" -p "What is the current date?"
```

**Note**: reasoning effort (`--effort`) currently **only has an effect with `gpt-oss` models**. For other models, set `--effort disabled` (or `reasoning_effort=disabled` in config/tasks) to avoid unexpected behavior.

```bash
ollama-agent --builtin-tool-timeout 60 --prompt "Run a long-running task"
# Or using short forms:
ollama-agent -t 60 -p "Run a long-running task"
```

**Available Parameters:**

- `-m`, `--model`: Specify the AI model to use
- `-p`, `--prompt`: Provide a prompt for non-interactive mode
- `-e`, `--effort`: Set reasoning effort level (low, medium, high, disabled)
- `-t`, `--builtin-tool-timeout`: Set built-in tool execution timeout in seconds
- `--rag <database>`: Load a RAG database for the session

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
reasoning_effort: "medium"  # low|medium|high|disabled
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

On the first run, the application will create a default configuration file at `~/.ollama-agent/config.ini`. You can edit this file to permanently change the default model, API URL, and other settings.

## Persistent Memory with Mem0

The agent can remember long-term facts by delegating storage and retrieval to [Mem0](https://github.com/mem0ai/mem0) running locally, backed by embedded/local Qdrant storage.

### Configure Mem0 storage path

In `~/.ollama-agent/config.ini` under `[mem0]`:

```ini
[mem0]
qdrant_path= ~/.ollama-agent/memory
```

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

**Search (CLI):**

```bash
ollama-agent rag-search my-docs "how to configure the agent"
# Limit results:
ollama-agent rag-search my-docs "configuration" -k 3
```

**Search (REPL):**

```text
/rag-load my-docs
/rag-search how to configure the agent
/rag-search configuration -k 3
```

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

In `~/.ollama-agent/config.ini` under `[rag]`:

```ini
[rag]
rag_dir = ~/.ollama-agent/rag
embedder_model = nomic-embed-text:latest
embedder_base_url = http://localhost:11434
embedding_dims = 768
default_top_k = 5
chunk_size = 500
chunk_overlap = 50
```

- `rag_dir`: Directory where RAG databases are stored
- `embedder_model`: Ollama model used for generating embeddings
- `embedding_dims`: Dimension of the embedding vectors (must match the model)
- `default_top_k`: Default number of results to return in searches
- `chunk_size`: Maximum size of text chunks (in characters)
- `chunk_overlap`: Overlap between consecutive chunks

## Agent Instructions

You can customize the agent's behavior by editing the instructions file at `~/.ollama-agent/instructions.md`. This file is automatically created on first use with default instructions.

## MCP Servers (Optional)

Ollama Agent supports the Model Context Protocol (MCP) to extend the agent's capabilities with additional tools and context. MCP servers are **optional** and can provide features like filesystem access, Git operations, and custom APIs.

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
- `ollama_agent/agent/`: Core agent logic, session management, and tools.
- `ollama_agent/tasks/`: Task management system.
- `ollama_agent/rag/`: RAG implementation for context retrieval.
- `ollama_agent/memory/`: Mem0 integration for long-term memory.
- `ollama_agent/vision/`: Screen vision and screenshot analysis.
- `ollama_agent/execution/`: execution helpers.
- `ollama_agent/streaming/`: Console output streaming and rendering.
- `ollama_agent/settings/`: Application configuration handling.
