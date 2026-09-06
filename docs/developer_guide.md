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
`ollama-agent` requires Python 3.11 or higher:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install in Editable Mode
Always manage dependencies and installations through `pyproject.toml`:

```bash
# Basic installation
.venv/bin/pip install -e .

# Development installation (includes ruff linter)
.venv/bin/pip install -e ".[dev]"

# Full installation (includes dev tools and mkdocs-material)
.venv/bin/pip install -e ".[dev,docs]"
```

### 4. Code Quality & Linting
Run Ruff to verify code formatting and compliance with project standards:
```bash
.venv/bin/ruff check .
```

---

## Testing & Quality Assurance

`ollama-agent` maintains an automated test suite covering runtime mechanics, prompt processing, session management, RAG vector stores, skills, and TUI components (27 test modules, 556 tests).

### Run Unit Tests
Always execute tests using the virtual environment interpreter:

```bash
# Run all unit tests
.venv/bin/python -m unittest discover -s tests

# Run a specific test module
.venv/bin/python -m unittest tests/test_agent_runtime.py
```

### Test Suite Structure

```text
tests/
├── test_agent_runtime.py          # DeepAgents graph initialization, reloading & tool timeout
├── test_agents_md.py              # Hierarchical AGENTS.md discovery up to .git root
├── test_clipboard.py              # Cross-platform clipboard backend integration
├── test_common.py                 # Payload text extraction & identifier validation
├── test_config.py                 # Settings loading, dataclass conversions & env injection
├── test_dispatch_cli.py           # CLI command handlers and argument parsing
├── test_episodic_memory.py        # Episodic memory search over stored conversations
├── test_i18n.py                   # Locale catalog validation & translation loading
├── test_interfaces_commands.py    # Session, model, task, skill, and RAG dispatching
├── test_mcp_loader.py             # MCP server configs, env expansions & connection handling
├── test_models.py                 # Capability checks, context window resolution & reasoning
├── test_prompt_processor.py       # @-mentions parsing, multimodal attachments & safety
├── test_prompt_queue.py           # Asynchronous prompt queue, non-blocking execution & FIFO draining
├── test_rag_commands.py           # CLI/REPL RAG operations, database lifecycle & resolution
├── test_rag_manager.py            # Document chunking, batch embeddings & stale point cleanup
├── test_repl.py                   # Textual REPL application, prompt queue execution & immediate commands
├── test_resource_manager.py       # Abstract BaseFileStoreManager tests
├── test_sessions.py               # SQLite session resumption, listing, export & deletion
├── test_skills.py                 # SKILL.md parsing, frontmatter extraction & validation
├── test_skills_commands.py        # Skill CRUD operations & error handling
├── test_stealth.py                # In-memory session execution without SQLite persistence
├── test_streaming.py              # Streaming event generators & async iteration
├── test_streaming_parsers.py      # Text & reasoning delta extraction logic
├── test_subagents.py              # Subagent graph compilation & isolated MCP server configs
├── test_tasks.py                  # YAML task persistence, data model & loading
├── test_tasks_commands.py         # Task execution & CLI dispatch tests
└── test_tui.py                    # Textual REPL widgets, header status & autocomplete
```

---

## Coding Standards & Engineering Rules

All contributions must strictly follow the engineering guidelines:

### 1. KISS (Keep It Simple, Stupid)
- **Radical Simplicity**: Write the least amount of straightforward, readable code that directly solves the problem. Never overengineer.
- **Do What Was Asked**: Implement the exact requirements. Do not anticipate hypothetical scenarios or speculative edge cases.
- **Linear & Obvious Flow**: Code must read top-to-bottom with obvious control flow. Avoid convoluted branching, unnecessary indirection layers, or wrapper functions.
- **No Premature Abstraction (YAGNI)**: Do not create interfaces, abstract base classes, or factories unless there is an immediate, concrete need.
- **Single Responsibility (SRP)**: Functions and modules should do one cohesive task and do it well. Keep them focused and concise.
- **Self-Documenting Code**: Write code so clear that comments explaining "what" it does are redundant. Only use comments to explain non-obvious business rules or external quirks ("why").

### 2. Zero Defensive Bloat
- **No Unsolicited Fallbacks & Safe Defaults**: Never mask errors or missing values with artificial defaults (e.g., returning `""`, `[]`, `{}`, `None`, `0`, or fallback objects) unless explicitly requested. Access properties and dictionary keys directly.
- **No Defensive Catch-and-Swallow**: Never wrap code in `try/except` just to catch generic exceptions, log a warning, and return a fallback value. Let exceptions propagate naturally unless performing an explicit retry or converting low-level errors at a system boundary.
- **No Internal Paranoid Null/Type Checking**: Do not check for `None` or validate types inside internal functions when data flow is guaranteed. Strict input validation belongs exclusively at public system boundaries (CLI inputs, raw user input, external APIs).
- **No Unnecessary `Optional` Types**: Do not use `Optional` (`| None`) types or fallback checks for parameters/variables when their presence and values are fully controlled and guaranteed by internal flow.
- **Fail Fast, Fail Loud**: If an invariant is violated or required input is missing, let the application fail immediately.

### 3. Top-Level Imports & Clean Architecture
- **Top-Level Imports Only**: All `import` and `from ... import` statements must reside at the very top of each Python file (PEP 8 standard). Never use function-level or inline imports.
- **No Structural Shortcuts**: Do not use inline imports or hacky workarounds to bypass circular dependencies. Solve the underlying structural problem properly through refactoring.

### 4. Virtual Environment & Python Tooling
- Always execute Python scripts, tools, and test suites using the project's virtual environment (`.venv/bin/python`).

### 5. Dependency Management
- Dependencies and packaging metadata are managed strictly in `pyproject.toml`.

### 6. Internationalization (i18n) Workflow
`ollama-agent` natively supports 16 languages (English baseline in Python source code + 15 translated JSON catalogs in `ollama_agent/i18n/locales/`):
- **Wrapping Strings**: All user-facing strings must be wrapped with `_("Message {param}", param=value)` imported from `ollama_agent.i18n`.
- **Locale Catalogs**: Translation dictionaries reside in `ollama_agent/i18n/locales/<locale>.json`. The keys must match the exact English format string.
- **Completeness Enforcement**: `tests/test_i18n.py` uses AST static analysis to parse every Python source file across the repository. It verifies that every `_()` call in the codebase exists in all 15 JSON catalogs and that interpolation keys match. Introducing a new localized string requires adding translations to all 15 JSON catalogs to ensure test suite passes.

---

## Codebase Architecture

```text
ollama-agent/
├── ollama_agent/
│   ├── __init__.py          # Package exports and version metadata
│   ├── main.py              # CLI/REPL entry point, signal routing, config resets
│   ├── agent/               # DeepAgents graph orchestration, middleware, tools & subagents
│   │   ├── agent.py         # AgentRuntime lifecycle, backend mounting, graph construction
│   │   ├── builtin_tools.py # Built-in tools (rag_search, search_past_conversations) and runtime context variables
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
│   │   ├── session_commands.py # SQLite session management, markdown export
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
│   │       └── default_instructions.md
│   ├── skills/              # Agent Skills implementation
│   │   ├── builtin/         # Internal application skills (mcp-configurator, skill-creator, task-creator)
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
├── tests/                   # Automated unit test suite (27 test modules)
├── docs/                    # MkDocs documentation source files
├── mkdocs.yml               # MkDocs configuration
├── AGENTS.md                # Development guidelines and coding conventions
├── pyproject.toml           # Project dependencies and packaging metadata
├── LICENSE                  # MIT license file
└── README.md                # Project documentation overview
```

---

## Contributing Guidelines

When contributing changes to `ollama-agent`:

1. **Branch**: Create a focused feature or bugfix branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. **Implement**: Keep changes minimal, direct, and aligned with KISS and Zero Defensive Bloat principles.
3. **Verify Locally**: Run linting and the test suite before submitting:
   ```bash
   .venv/bin/ruff check .
   .venv/bin/python -m unittest discover -s tests
   ```
4. **Pre-Submission Checklist**:
   - [ ] All unit tests pass (`.venv/bin/python -m unittest discover -s tests`).
   - [ ] Ruff check reports no errors (`.venv/bin/ruff check .`).
   - [ ] All imports are strictly placed at the top of files (PEP 8).
   - [ ] No unsolicited fallback defaults, defensive try/except blocks, or unnecessary `Optional` types.
   - [ ] Any new dependencies are declared in `pyproject.toml`.

---

## Building Documentation

To build and preview the documentation site locally:

```bash
# Build the static site into site/
.venv/bin/python -m mkdocs build

# Start the local development server with live reload
.venv/bin/python -m mkdocs serve
```
