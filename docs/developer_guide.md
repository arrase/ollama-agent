# Developer Guide & Contribution Standards

This guide explains how to set up a local development environment, run test suites, navigate the codebase, and adhere to the architectural and engineering principles of `ollama-agent`.

---

## Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/arrase/ollama-agent.git
cd ollama-agent
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install in Editable Mode
```bash
# Basic installation
pip install -e .

# Development installation (includes ruff linter)
pip install -e ".[dev]"
```

### 4. Optional: Install Documentation Dependencies
```bash
pip install mkdocs-material
```

### 5. Code Quality & Linting
Run Ruff to verify code formatting and compliance with project standards:
```bash
.venv/bin/ruff check .
```

---

## Testing & Quality Assurance

`ollama-agent` maintains an automated test suite covering runtime mechanics, prompt processing, session management, RAG vector stores, skills, and TUI components.

### Run Unit Tests
Always execute tests using the virtual environment interpreter:

```bash
.venv/bin/python -m unittest discover -s tests
```

### Test Suite Structure

```text
tests/
├── test_agent_runtime.py          # DeepAgents graph initialization, reloading & tool timeout
├── test_agents_md.py              # Hierarchical AGENTS.md discovery up to .git root
├── test_clipboard.py              # Cross-platform clipboard backend integration
├── test_common.py                 # Payload text extraction & identifier validation
├── test_compaction.py              # SummarizationMiddleware delegation, history offloading & cutoffs
├── test_compaction_interop.py     # Interop contract with deepagents SummarizationMiddleware internals
├── test_config.py                 # Settings loading, dataclass conversions & env injection
├── test_dispatch_cli.py           # CLI command handlers and argument parsing
├── test_episodic_memory.py        # Episodic memory search over stored conversations
├── test_i18n.py                   # Locale catalog validation & translation loading
├── test_interfaces_commands.py    # Session, model, task, skill, and RAG dispatching
├── test_mcp_loader.py             # MCP server configs, env expansions & connection handling
├── test_models.py                 # Capability checks, context window resolution & reasoning
├── test_prompt_processor.py       # @-mentions parsing, multimodal attachments & safety
├── test_prompt_queue.py           # Asynchronous prompt queue, non-blocking execution & FIFO draining
├── test_rag.py                    # RAG operations, Qdrant client & embeddings
├── test_rag_manager.py            # Document chunking, batch embeddings & stale point cleanup
├── test_resource_manager.py       # Abstract BaseFileStoreManager tests
├── test_sessions.py               # SQLite session resumption, listing, export & deletion
├── test_skills.py                 # SKILL.md parsing, frontmatter extraction & validation
├── test_skills_commands.py        # Skill CRUD operations & error handling
├── test_streaming.py              # Streaming event generators & async iteration
├── test_streaming_parsers.py      # Text & reasoning delta extraction logic
├── test_tasks.py                  # YAML task persistence, data model & loading
├── test_tasks_commands.py         # Task execution & CLI dispatch tests
└── test_tui.py                    # Textual REPL widgets, header status & autocomplete
```

---

## Coding Standards & Engineering Rules

All contributions must strictly follow the engineering guidelines:

### 1. KISS (Keep It Simple, Stupid)
- **Radical Simplicity**: Write the least amount of straightforward, readable code that directly solves the problem. Never overengineer.
- **Linear & Obvious Flow**: Code must read top-to-bottom with obvious control flow. Avoid convoluted branching or unnecessary wrapper layers.
- **No Premature Abstraction (YAGNI)**: Do not create interfaces, abstract base classes, or factories unless there is an immediate, concrete need.

### 2. Zero Defensive Bloat
- **No Unsolicited Fallbacks**: Never mask errors or missing values with artificial defaults (e.g. returning `""`, `[]`, `{}`, `None`, `0`, or a fallback object) unless explicitly requested.
- **No Defensive Catch-and-Swallow**: Never wrap code in `try/except` just to catch generic exceptions, log a warning, and return a fallback value. Let exceptions propagate naturally.
- **Fail Fast, Fail Loud**: If an invariant is violated or required input is missing, let the application fail immediately.

### 3. Top-Level Imports Only
- All `import` and `from ... import` statements must reside at the very top of each Python file (PEP 8 standard). Never use function-level or inline imports.

### 4. Dependency Management
- Dependencies and packaging metadata are managed strictly in `pyproject.toml`.

---

## Codebase Architecture

```text
ollama-agent/
├── ollama_agent/
│   ├── main.py              # CLI/REPL entry point, signal routing, config resets
│   ├── agent/               # DeepAgents graph orchestration, middleware, tools & subagents
│   │   ├── agent.py         # AgentRuntime lifecycle, backend mounting, graph construction
│   │   ├── builtin_tools.py # Built-in tools (rag_search) and runtime context variables
│   │   ├── compaction.py    # Manual compaction delegating to deepagents SummarizationMiddleware
│   │   ├── environment.py   # Shared prompt-environment helpers (OS, CWD, datetime)
│   │   ├── episodic_memory.py # Episodic memory search engine over past conversations
│   │   ├── middleware.py    # Tool call event streaming & execution timeout protection
│   │   └── subagents.py     # SubAgentSettings to DeepAgents subagent specification builder
│   ├── core/                # Model capability checks, context resolution, prompt processing
│   │   ├── common.py        # Shared types, identifier validation, text extraction
│   │   ├── models.py        # ChatOllama initialization, tool checks, reasoning mapping
│   │   ├── prompt_processor.py # @-mention parsing, path resolution, multimodal encoding
│   │   └── resource_manager.py # Generic BaseFileStoreManager for tasks and skills
│   ├── i18n/                # Internationalization engine and translation catalogs
│   │   ├── __init__.py      # Translation loader, _() helper, locale negotiation
│   │   └── locales/         # JSON translation catalogs (15 languages: ar, de, es, fr, hi, it, ja, ko, nl, pl, pt, ru, tr, uk, zh)
│   ├── interfaces/          # User interface implementations (CLI & Textual REPL)
│   │   ├── cli.py           # Argparse setup, command-line dispatch, non-interactive mode
│   │   ├── clipboard.py     # OS clipboard integration (macOS, Wayland, X11, Windows)
│   │   ├── dispatch.py      # Unified CLI/REPL command handler registry
│   │   ├── model_commands.py# Model listing with tool capabilities, model switching
│   │   ├── session_commands.py # SQLite session management, markdown export, compaction
│   │   ├── tui_components.py# Textual TUI widgets (header, footer, input, messages, approvals, prompt queue, system notices)
│   │   ├── repl.py          # Interactive Textual REPL application & autocomplete
│   │   └── repl.css         # Styling for Textual REPL interface
│   ├── mcp/                 # Model Context Protocol integration
│   │   ├── commands.py      # MCP status inspection (/mcp and mcp list)
│   │   └── loader.py        # MultiServerMCPClient loader with env expansion
│   ├── rag/                 # Local RAG engine
│   │   ├── commands.py      # CLI/REPL RAG command handlers
│   │   └── manager.py       # Qdrant client, chunking, Ollama embeddings pipeline
│   ├── settings/            # Configuration management
│   │   ├── config.py        # YAML configuration loader, dataclasses (ModelSettings, RAGSettings), reset logic
│   │   ├── paths.py         # Centralized filesystem constants (~/.ollama-agent/)
│   │   └── prompts/         # Bundled markdown prompt templates
│   │       ├── default_instructions.md
│   │       ├── fs_policy_sandboxed.md
│   │       ├── fs_policy_traversal.md
│   │       └── rag_policy.md
│   ├── skills/              # Agent Skills implementation
│   │   ├── builtin/         # Internal application skills (skill-creator, task-creator)
│   │   ├── commands.py      # Skill CLI and REPL handlers
│   │   └── manager.py       # SkillManager and SKILL.md YAML frontmatter parser
│   ├── streaming/           # Streaming event handling and rendering
│   │   ├── base.py          # Abstract StreamingRenderer
│   │   ├── console_renderer.py # Rich live console renderer for CLI output
│   │   ├── events.py        # stream_agent_events and non-interactive runner
│   │   ├── interrupts.py    # Tool interrupt and human approval streaming handlers
│   │   └── parsers.py       # ThinkTagParser, streaming_text, and streaming_reasoning chunk parsers
│   └── tasks/               # Saved task management
│       ├── commands.py      # Task CLI and REPL handlers
│       └── manager.py       # TaskManager and Task YAML serializer
├── tests/                   # Automated unit test suite (26 test modules)
├── docs/                    # MkDocs documentation source files
├── mkdocs.yml               # MkDocs configuration
├── AGENTS.md                # Development guidelines and coding conventions
├── pyproject.toml           # Project dependencies and packaging metadata
├── LICENSE                  # MIT license file
└── README.md                # Project documentation overview
```

---

## Building Documentation

To build and preview the documentation site locally:

```bash
# Build the static site into site/
.venv/bin/python -m mkdocs build

# Start the local development server with live reload
.venv/bin/python -m mkdocs serve
```
