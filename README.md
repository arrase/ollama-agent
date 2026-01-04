# Ollama Agent

Ollama Agent is a powerful command-line tool (CLI and REPL) that allows you to interact with local AI models through an Ollama-compatible API. It provides a persistent chat experience, session management, and the ability to execute local shell commands, turning your local models into helpful assistants for your daily tasks.

## Features

- **Interactive REPL**: A modern, terminal-based chat interface with Markdown rendering and slash commands (inspired by Claude Code).
- **Non-Interactive CLI**: Execute single prompts directly from your command line for quick queries.
- **Ollama Integration**: Connects to any Ollama-compatible API endpoint.
- **Screen Vision (Screenshots)**: Attach monitor screenshots in prompts using `@dpN` for visual context.
- **Tool-Powered**: The agent can execute shell commands, allowing it to interact with your local environment to perform tasks.
- **Delegated MCP Agents**: Each configured MCP server can run through its own lightweight agent with custom model and instructions.
- **Session Management**: Conversations are automatically saved and can be reloaded, deleted, or switched between.
- **Task Management**: Save frequently used prompts as "tasks" and execute them with a simple command.
- **Configurable**: Easily configure the model, API endpoint, and agent reasoning effort.
- **Mem0 Memory Layer**: Persistent memory backed by Mem0 + Qdrant, exposed through function-calling tools.

## Installation

Before you begin, ensure you have a running instance of [Ollama](https://ollama.com/) or another compatible API server.

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
- `/tasks`: List saved tasks.
- `/task-run <id>`: Run a specific task.
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

## Tasks

Tasks are saved prompts that can be executed repeatedly.

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

The agent can remember long-term facts by delegating storage and retrieval to [Mem0](https://github.com/mem0ai/mem0) running locally, backed by a Qdrant vector store that the agent automatically manages via Docker.

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

- `ollama_agent/main.py`: Main entry point, handles CLI arguments and starts the REPL or non-interactive mode.
- `ollama_agent/repl.py`: Interactive Read-Eval-Print Loop implementation.
- `ollama_agent/cli.py`: CLI plumbing and subcommands.
- `ollama_agent/runner.py`: Orchestrates runs (REPL/CLI) and agent execution.
- `ollama_agent/agent/agent.py`: Core agent implementation (OpenAI Agents SDK).
- `ollama_agent/streaming/`: Streaming events and renderers (console).
