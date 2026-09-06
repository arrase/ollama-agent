# Ollama Agent

<p align="center">
  <strong>Your AI assistant. Your rules. Running on your machine.</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Ollama-Native%20API-black" alt="Ollama Native API"></a>
  <a href="https://docs.langchain.com/oss/python/deepagents/overview"><img src="https://img.shields.io/badge/Built%20with-DeepAgents%20%26%20LangGraph-purple" alt="DeepAgents &amp; LangGraph"></a>
  <a href="https://arrase.github.io/ollama-agent/"><img src="https://img.shields.io/badge/Docs-GitHub%20Pages-emerald" alt="Documentation"></a>
</p>

<p align="center">
  <a href="#-not-another-coding-agent">Why Ollama Agent?</a> •
  <a href="#-built-for-ollama-not-bolted-on">The Ollama Advantage</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-cheat-sheet">Cheat Sheet</a> •
  <a href="https://arrase.github.io/ollama-agent/">Full Documentation ↗</a>
</p>

---

**Ollama Agent** is an autonomous, local-first AI assistant that runs entirely on your hardware through [Ollama](https://ollama.com/). It's a **general-purpose agent** — not a coding assistant, not a chatbot, not a narrow tool — but whatever *you* need it to be. Write research reports, manage files, analyze documents, automate workflows, query knowledge bases, or yes, write code too. Every prompt that shapes its behavior is an editable file on your disk, so you stay in control of *what* the agent is and *how* it thinks.

> 📖 **Comprehensive Guides & Technical Reference**: Visit the official documentation site at **[arrase.github.io/ollama-agent](https://arrase.github.io/ollama-agent/)**.

---

## 🎯 Not Another Coding Agent

Most open-source AI agents are coding assistants in disguise. They ship with hardcoded system prompts about writing code, generating tests, and refactoring functions. If you want to use them for anything else, you're fighting their DNA.

**Ollama Agent is different.** It ships as a blank canvas with general-purpose defaults, and gives you the tools to make it *yours*:

### 🧬 Fully Editable Prompts — You Define the Agent

The system prompt that drives Ollama Agent is a plain Jinja2 template sitting at `~/.ollama-agent/prompts/instructions.md`. Open it, rewrite it, and the agent becomes whatever you need:

```
~/.ollama-agent/
├── prompts/
│   └── instructions.md      ← The agent's brain. Edit freely.
├── MEMORY.md                 ← Persistent cross-session memory
├── settings.yaml             ← Model, context, sampling, subagents
├── tasks/                    ← Reusable YAML prompt templates
├── skills/                   ← Modular procedural workflows
└── mcp.json                  ← External tool server connections
```

- **Research analyst?** Rewrite the prompt to focus on source evaluation, citation formatting, and document synthesis.
- **System administrator?** Shape it around infrastructure monitoring, log analysis, and runbook execution.
- **Creative writer?** Tune it for narrative structure, world-building, and stylistic consistency.
- **Personal assistant?** Optimize it for calendar planning, email drafting, and task management.
- **Software engineer?** Sure, that works too — but it's *your choice*, not the default assumption.

The template has access to the full configuration context via Jinja2 variables (`runtime`, `model`, `rag`, `settings`), so your instructions can adapt dynamically to runtime conditions. Made a mess? `ollama-agent --config-reset system-prompt` restores the defaults instantly.

### 🌍 Multilingual Interface

The entire UI speaks your language. Ollama Agent ships with **15 built-in locales** (English, Spanish, French, German, Japanese, Chinese, Korean, Arabic, Hindi, Italian, Dutch, Polish, Portuguese, Russian, Turkish, Ukrainian) and auto-detects your system locale. Override anytime with `-l <lang>` or `runtime.language` in settings.

---

## 🦙 Built for Ollama, Not Bolted On

Most agents treat Ollama as a dumb OpenAI-compatible proxy. They send requests to `/v1/chat/completions`, cross their fingers, and wonder why the output is truncated, the context is wrong, and the model ignores tool calls. **Ollama Agent talks directly to Ollama's native API** and is engineered to squeeze every capability out of your local models:

| What goes wrong | Generic OpenAI-proxy agents | Ollama Agent |
| :--- | :--- | :--- |
| **Context window** | Default to Ollama's 2K–4K `num_ctx`; large prompts get silently truncated. | Auto-detects model capacity from GGUF metadata and sets `num_ctx` dynamically. Use `/context max` for full range. |
| **Sampling parameters** | Force fixed defaults (`temp=0.7`, `top_p=1.0`), ignoring the Modelfile. | Auto-discovers optimal sampling (`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`) from the Modelfile. |
| **Token counting** | Approximate with `tiktoken` — wrong tokenizer for Llama, Qwen, Gemma, DeepSeek. | Reads native server metrics (`prompt_eval_count` + `eval_count`) for exact real-time tracking. |
| **Context overflow** | Crash with obscure errors when the conversation exceeds the limit. | Auto-compaction at 85% capacity: summarizes older turns, prunes tool output, offloads history to disk. |
| **Reasoning traces** | Leak raw `<think>` tokens into output or fail to configure thinking effort. | Architecture-aware thinking: translates effort levels per model family (Qwen, DeepSeek, GPT-OSS) into collapsible UI blocks. |
| **Model compatibility** | Blindly attempt tool calls on models that don't support them; fail with cryptic errors. | Pre-flight capability check: verifies `tools` support, offers interactive model selector, hot-swaps models mid-session (`/model set`). |

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
ollama-agent -p "Summarize the key findings in @report.pdf"

# Autonomous research with YOLO mode (auto-approves tool actions)
ollama-agent -m "qwen3.8:27b" -e "high" -y -p "Analyze the logs in @/var/log/syslog and report anomalies."
```

---

## 🖥️ Interactive REPL Experience

The interactive terminal interface is built with **Textual** and **Rich** to provide a fast, keyboard-first environment:

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
