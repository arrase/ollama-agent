# Ollama Agent

<p align="center">
  <strong>The autonomous, local-first AI assistant built natively for Ollama.</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-Native%20API-black" alt="Ollama Native API"></a>
  <a href="https://docs.langchain.com/oss/python/deepagents/overview"><img src="https://img.shields.io/badge/Built%20with-DeepAgents%20%26%20LangGraph-purple" alt="DeepAgents & LangGraph"></a>
  <a href="https://arrase.github.io/ollama-agent/"><img src="https://img.shields.io/badge/Docs-GitHub%20Pages-emerald" alt="Documentation"></a>
</p>

<p align="center">
  <a href="#-the-ollama-advantage">Why Ollama Agent?</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-cheat-sheet">Cheat Sheet</a> •
  <a href="https://arrase.github.io/ollama-agent/">Full Documentation ↗</a>
</p>

---

**Ollama Agent** is an autonomous terminal AI assistant (interactive REPL and scriptable CLI) designed from the ground up for local LLMs. Powered by [DeepAgents](https://docs.langchain.com/oss/python/deepagents/overview) and [LangGraph](https://github.com/langchain-ai/langgraph), it brings Claude Code-like agentic capabilities to your local machine: stateful multi-turn workflows, autonomous tool execution with human-in-the-loop safety, dynamic context management, project rules (`AGENTS.md`), local RAG, Agent Skills, and Model Context Protocol (MCP) integrations.

> 📖 **Comprehensive Guides & Technical Reference**: Visit the official documentation site at **[arrase.github.io/ollama-agent](https://arrase.github.io/ollama-agent/)**.

---

## 🦙 The Ollama Advantage

Most generic AI agents treat Ollama as just an OpenAI-compatible endpoint, leading to poor output quality, truncated context, and frustrating crashes. **Ollama Agent communicates directly with Ollama's native API** and is uniquely engineered to extract the full potential of local open-weights models:

| Feature | Generic OpenAI-Proxy Agents | Ollama Agent |
| :--- | :--- | :--- |
| **Context Window (`num_ctx`)** | ❌ Defaults to Ollama's 2K–4K limit; truncates large prompts and forgets context quickly. | ✅ **Auto-detects model capacity** from GGUF metadata (`context_length`) or Modelfile parameters. Sets `num_ctx` dynamically (or `/context max`). |
| **Model Hyperparameters** | ❌ Forces fixed defaults (`temp=0.7`, `top_p=1.0`), ignoring model-specific tuning. | ✅ **Auto-discovers optimal sampling** from the Modelfile (`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`). |
| **Token Telemetry** | ❌ Approximates tokens with `tiktoken` (inaccurate for Llama, Qwen, Gemma, DeepSeek). | ✅ **Reads native server metrics** (`prompt_eval_count` + `eval_count`) directly from Ollama for exact real-time tracking. |
| **Context Management** | ❌ Crashes with context overflow errors when conversation exceeds limit. | ✅ **Auto-compaction at 85% capacity**: summarizes older turns, prunes bulky tool arguments, and offloads history to disk. |
| **Reasoning Traces** | ❌ May leak raw `<think>` tokens into output or fail to configure thinking effort. | ✅ **Architecture-aware thinking**: translates effort levels per model family (Qwen, DeepSeek, GPT-OSS) into collapsible UI blocks. |
| **Model Verification** | ❌ Blindly attempts tool calls, failing with cryptic errors if unsupported. | ✅ **Pre-flight capability check**: verifies `tools` support, presents an interactive selector if unconfigured, and hot-swaps models mid-session (`/model set`). |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **[Ollama](https://ollama.com/)** running locally (or reachable via network)
- A local tool-calling model (e.g. `ollama pull qwen3.8:27b`, `ollama pull qwen2.5:14b`, or `ollama pull llama3.1:8b`)

### 1. Installation

Install in an isolated environment using **`pipx`** (recommended):

```bash
pipx install git+https://github.com/arrase/ollama-agent.git
```

*Or via pip into an active virtual environment:*
```bash
pip install git+https://github.com/arrase/ollama-agent.git
```

### 2. Launch the Interactive REPL

```bash
ollama-agent
```
*If no model is configured, Ollama Agent will automatically detect your downloaded models and let you choose one interactively.*

### 3. Run One-Off Commands from the Shell

```bash
# Quick query
ollama-agent -p "Explain the role of middleware in this project."

# Non-interactive refactoring with YOLO mode (auto-approves tool actions)
ollama-agent -m "qwen3.8:27b" -e "high" -y -p "Refactor @src/utils.py to comply with PEP 8."
```

---

## 🖥️ Interactive REPL Experience

The interactive terminal interface is built with **Textual** and **Rich** to provide a fast, keyboard-first development environment:

```text
● ollama-agent │ Model: qwen3.8:27b │ Context: 3.4k/32.0k (11%) │ Effort: high │ YOLO: OFF │ STEALTH: OFF
```

- **Non-Blocking Prompt Queue**: Never wait for generation to finish. Type follow-up prompts or execute read-only slash commands while the model is actively streaming or waiting for tool confirmation.
- **Multiline Editing**: Type `\` followed by `Enter` (`\ + Enter`) to create clean newlines. The input area expands dynamically up to 8 lines.
- **3-Level Tab Autocompletion**: Autocompletes slash commands, subcommands, entities (models, sessions, tasks, skills, RAG databases), and `@-mention` file paths.
- **Real-Time Token Gauge**: Color-coded header indicator (Cyan $\le 75\%$, Amber $76\%-90\%$, Red $>90\%$) based on real Ollama server token counts.
- **Mid-Session Switching**: Switch models (`/model set <name>`), change reasoning effort (`/effort high`), or update context window (`/context max`) mid-conversation without losing thread state.

---

## ✨ Key Features

### 🛡️ Human-in-the-Loop (HITL), YOLO & Stealth Modes
- **Action Approval Dialog**: Sensitive tool operations (running shell commands, writing/editing files) require keyboard confirmation (`y` approve, `n` reject, `a` allow for session, `c` cancel).
- **YOLO Mode (`-y` / `/yolo`)**: Bypass confirmation pauses for fully autonomous agent runs.
- **Stealth Mode (`-s` / `/stealth`)**: Run conversations in-memory without saving conversation turns or checkpoints to SQLite history.

### 📁 Interactive Context Injection (`@-mentions`)
Reference local files or folders directly in your prompts with autocompletion:
- **Single & Quoted Files**: `@src/main.py`, `@"data/financial report.csv"`
- **Directory Traversal**: `@src` or `@.` (recursively attaches all supported source files).
- **Multimodal Assets**: Automatically base64-encodes images (`.png`, `.jpg`, `.webp`), audio, video, and documents (`.pdf`, `.pptx`) for vision-enabled models.

### 🧠 Project Guidelines & Persistent Memory
- **Repository Guidelines (`AGENTS.md`)**: Automatically discovered in the working directory up to the git root and mounted into agent memory.
- **Cross-Session Memory (`MEMORY.md`)**: Preserves user preferences and architectural decisions across sessions.
- **Episodic Memory**: Autonomous past conversation search via the `search_past_conversations` tool and user search via `/session search <query>`.

### 🧩 Tasks, Skills & Local RAG
- **Saved Tasks**: Reusable YAML prompt templates with Jinja2 expressions (`~/.ollama-agent/tasks/`), input type validation, and CLI execution (`ollama-agent task run <id>`).
- **Agent Skills**: Modular procedural workflows adhering to the open [Agent Skills specification](https://agentskills.io/specification).
- **Local RAG Engine**: Embed and index documents into local Qdrant collections using Ollama embeddings (`ollama pull nomic-embed-text`), retrieved automatically via the `rag_search` tool.

### 🔌 Model Context Protocol (MCP) & Subagents
- **MCP Extensibility**: Connect external tools over `stdio`, `http`, and `sse` transports declared in `~/.ollama-agent/mcp.json`.
- **Specialized Subagents**: Configure isolated subagents in `settings.yaml` with their own model, system prompt, context window, and dedicated MCP tool servers.

---

## ⚡ Cheat Sheet

### Common Slash Commands (REPL)

| Command | Usage | Description |
| :--- | :--- | :--- |
| `/model` | `/model [list \| set <name>]` | List local models with tool support or switch active model. |
| `/context` | `/context [<size \| max>]` | Inspect or change context window (`num_ctx`) on the fly. |
| `/params` | `/params [list \| set <param> <val>]` | View effective parameters and resolution sources, or update sampling values. |
| `/effort` | `/effort [<level>]` | Set reasoning effort (`low`, `medium`, `high`, `xhigh`, `hide`, `disabled`). |
| `/queue` | `/queue [list \| clear \| rm <pos>]` | Inspect and manage pending prompts in the FIFO execution queue. |
| `/session` | `/session [list \| resume <id> \| new]` | Manage chat threads, resume past conversations, or start fresh. |
| `/task` | `/task [list \| run <id> \| create]` | List, run with variables (`key=val`), or create saved YAML tasks. |
| `/skill` | `/skill [list \| show <id> \| create]` | Inspect, create, or manage modular Agent Skills. |
| `/rag` | `/rag [status \| list \| load <db>]` | Inspect status, list vector databases, or attach knowledge bases. |
| `/agents` | `/agents [list]` | List specialized subagents and their assigned models/tools. |
| `/mcp` | `/mcp [list \| reload]` | Check MCP server connection health or reload tool definitions live. |
| `/yolo` | `/yolo [on \| off]` | Toggle confirmation bypass for autonomous execution. |
| `/stealth` | `/stealth [on \| off]` | Toggle ephemeral mode without saving to SQLite history. |

### Essential CLI Flags

```bash
ollama-agent -m <model>      # Specify Ollama model
ollama-agent -p "<prompt>"   # Run in non-interactive single-shot mode
ollama-agent -y              # Run in YOLO mode (bypass tool approvals)
ollama-agent -s              # Run in Stealth mode (in-memory only)
ollama-agent -c <num|max>    # Set context window size (tokens or 'max')
ollama-agent -e <effort>     # Set reasoning effort level
ollama-agent --rag <db>      # Preload a RAG vector collection
ollama-agent -l <lang>       # Override interface language (e.g. en, es, fr, de, ja, zh)
```

---

## 📚 Documentation

For complete architecture diagrams, configuration manuals, and development guides, visit our **[Documentation Site](https://arrase.github.io/ollama-agent/)**:

- 📖 **[CLI & REPL User Guide](https://arrase.github.io/ollama-agent/cli_repl/)** — Full terminal navigation, slash commands, multiline inputs, and scriptable subcommands.
- 🧩 **[Agent Skills](https://arrase.github.io/ollama-agent/skills/)** — Modular procedural workflows adhering to the open Agent Skills standard.
- 📋 **[Saved Tasks](https://arrase.github.io/ollama-agent/tasks/)** — Reusable Jinja2 prompt automation routines with typed inputs.
- 🔌 **[Model Context Protocol (MCP)](https://arrase.github.io/ollama-agent/mcp/)** — External tool servers, stdio/SSE transports, and live reloading.
- 🤖 **[Specialized Subagents](https://arrase.github.io/ollama-agent/subagents/)** — Isolated subagent graphs, dedicated models, and exclusive MCP tools.
- 🧠 **[Memory & Guidelines](https://arrase.github.io/ollama-agent/memory/)** — `AGENTS.md` project rules, `MEMORY.md` user preferences, SQLite sessions, and episodic memory.
- 📚 **[Local RAG Guide](https://arrase.github.io/ollama-agent/rag/)** — Qdrant vector store management, Ollama embeddings, chunking, and semantic search.
- ⚙️ **[Configuration Reference](https://arrase.github.io/ollama-agent/configuration/)** — Complete `settings.yaml` schema, parameter precedence, context resolution, and LangSmith.
- 🏗️ **[System Architecture](https://arrase.github.io/ollama-agent/architecture/)** — DeepAgents orchestration, SQLite checkpoints, streaming parsers, and compaction engine.
- 🛠️ **[Developer Guide](https://arrase.github.io/ollama-agent/developer_guide/)** — Contributing guidelines, local environment setup, and test suite execution.

---

## 💻 Developer Setup

```bash
# Clone the repository
git clone https://github.com/arrase/ollama-agent.git
cd ollama-agent

# Create virtual environment and install in editable mode
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run test suite
.venv/bin/python -m unittest discover -s tests
```

For detailed contributing standards, linting with Ruff, and architectural breakdowns, see the [Developer Guide](docs/developer_guide.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
