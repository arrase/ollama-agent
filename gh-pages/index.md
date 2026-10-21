---
hide:
  - toc
---

<div class="hero-section">
  <div class="hero-eyebrow">
    <i class="fa-solid fa-robot" aria-hidden="true"></i>
    <span>Autonomous Local AI Assistant</span>
  </div>
  <h1>Ollama Agent</h1>
  <p class="hero-subtitle">
    An autonomous, local-first AI assistant running entirely on your machine via Ollama. Features an interactive terminal REPL, one-shot CLI, Model Context Protocol (MCP) tools, specialized subagents, local RAG, and persistent memory.
  </p>
  <div class="hero-badges">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
    <a href="https://docs.langchain.com/oss/python/deepagents/overview"><img src="https://img.shields.io/badge/Framework-DeepAgents-purple.svg" alt="DeepAgents"></a>
    <a href="https://github.com/langchain-ai/langchain"><img src="https://img.shields.io/badge/LLM-LangChain-green.svg" alt="LangChain"></a>
    <a href="https://ollama.com/"><img src="https://img.shields.io/badge/Backend-Ollama-black.svg" alt="Ollama"></a>
  </div>
  <div class="hero-badges">
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=alert_status" alt="Quality Gate Status"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=coverage" alt="Coverage"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=bugs" alt="Bugs"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=vulnerabilities" alt="Vulnerabilities"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=code_smells" alt="Code Smells"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=sqale_index" alt="Technical Debt"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=duplicated_lines_density" alt="Duplicated Lines (%)"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=security_rating" alt="Security Rating"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=reliability_rating" alt="Reliability Rating"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=sqale_rating" alt="Maintainability Rating"></a>
    <a href="https://sonarcloud.io/summary/overall?id=arrase_ollama-agent"><img src="https://sonarcloud.io/api/project_badges/measure?project=arrase_ollama-agent&amp;metric=ncloc" alt="Lines of Code"></a>
  </div>
</div>

<div class="section-header">
  <h2>Key Features</h2>
  <span>Local-first intelligence &amp; extensible tooling</span>
</div>

<div class="projects-grid">
  <div class="feature-card">
    <div>
      <i class="fa-solid fa-terminal feature-icon" aria-hidden="true"></i>
      <h3>Interactive CLI &amp; REPL</h3>
      <p>Rich terminal UI with live syntax-highlighted streaming, multiline editing, 3-level tab autocompletion, and one-shot CLI mode (<code>-p</code>) for shell automation.</p>
    </div>
  </div>
  <div class="feature-card">
    <div>
      <i class="fa-solid fa-plug feature-icon" aria-hidden="true"></i>
      <h3>Model Context Protocol (MCP)</h3>
      <p>Connect seamlessly to external MCP tool servers over <code>stdio</code>, <code>http</code>, or <code>sse</code> with live tool discovery and runtime management.</p>
    </div>
  </div>
  <div class="feature-card">
    <div>
      <i class="fa-solid fa-sitemap feature-icon" aria-hidden="true"></i>
      <h3>Specialized Subagents</h3>
      <p>Delegate complex workflows to custom subagents with isolated context windows, dedicated models, tailored system prompts, and exclusive MCP toolsets.</p>
    </div>
  </div>
  <div class="feature-card">
    <div>
      <i class="fa-solid fa-database feature-icon" aria-hidden="true"></i>
      <h3>Local RAG Engine</h3>
      <p>Index local documentation and codebases into an embedded vector store powered by Ollama embeddings for fast, private semantic search.</p>
    </div>
  </div>
  <div class="feature-card">
    <div>
      <i class="fa-solid fa-wand-magic-sparkles feature-icon" aria-hidden="true"></i>
      <h3>Persistent Memory &amp; Skills</h3>
      <p>Retain user preferences across sessions (<code>MEMORY.md</code>), auto-discover project guidelines (<code>AGENTS.md</code>), and equip reusable Agent Skills and Saved Tasks.</p>
    </div>
  </div>
  <div class="feature-card">
    <div>
      <i class="fa-solid fa-shield-halved feature-icon" aria-hidden="true"></i>
      <h3>100% Local &amp; Private</h3>
      <p>Built natively for Ollama with automatic context detection, creator hyperparameter discovery, human-in-the-loop safety gates, and zero cloud dependency.</p>
    </div>
  </div>
</div>

---

## Quick Installation

Install Ollama Agent in an isolated Python environment using **pipx** (recommended) or standard **pip**:

=== "pipx (Recommended)"

    ```bash
    # Install globally in an isolated environment
    pipx install git+https://github.com/arrase/ollama-agent.git

    # Upgrade to the latest release
    pipx upgrade ollama-agent
    ```

=== "pip (Virtual Environment)"

    ```bash
    # Install into your active Python 3.11+ virtual environment
    pip install git+https://github.com/arrase/ollama-agent.git

    # Upgrade existing installation
    pip install --upgrade git+https://github.com/arrase/ollama-agent.git
    ```

---

## Prerequisites

Before launching Ollama Agent, ensure the following are available on your system:

1. **Python 3.11+**: Verify with `python3 --version`.
2. **Ollama**: Downloaded, installed, and running (`ollama serve`).
3. **Tool-Calling Model**: A model with tool/function-calling capabilities:
   ```bash
   ollama pull qwen2.5-coder:14b
   # or: ollama pull llama3.1:8b
   ```
   *(If unconfigured, Ollama Agent automatically scans your installed Ollama models and presents an interactive selector).*
4. **Embeddings Model (Optional, for Local RAG)**:
   ```bash
   ollama pull embeddinggemma-2:740m
   ```

---

## 60-Second Quick Start

### 1. Launch the Interactive REPL
Start the full-featured terminal workspace with live markdown streaming, context tracking, and command autocompletion:
```bash
ollama-agent
```

### 2. Run a One-Off CLI Prompt
Execute a task directly from your shell and stream the answer to standard output:
```bash
ollama-agent -p "Summarize git commits made in the last 7 days."
```

### 3. Combine Advanced Flags
Specify a dedicated model, reasoning effort, and autonomous execution (YOLO mode):
```bash
ollama-agent -m "qwen2.5-coder:14b" -e high -y -p "Audit src/auth.py for security vulnerabilities."
```

!!! tip "Common CLI Flags at a Glance"
    * `-p, --prompt`: Run in non-interactive single-shot mode and exit when done.
    * `-m, --model`: Select any installed Ollama model for the session.
    * `-e, --effort`: Set reasoning effort (dynamically matched to model-supported values, e.g. `low`, `high`, `max`, `default`).
    * `-y, --yolo`: Enable YOLO mode (runs tools autonomously without approval prompts).
    * `-s, --stealth`: Run in-memory without saving conversation history to disk.

---

## Why Ollama Agent? (The Local Advantage)

Most agentic frameworks treat Ollama as a generic OpenAI-compatible proxy. This often causes truncated outputs, missed tool calls, and lost context. **Ollama Agent is built natively for Ollama**, harnessing the full potential of local LLMs:

| Capability | Generic OpenAI Proxy Agents | Ollama Agent |
| :--- | :--- | :--- |
| **Data Privacy** | Frequently routes data to third-party endpoints or telemetry servers. | **100% Local & Private.** Prompts, code, and documents never leave your machine. |
| **Context Window (`num_ctx`)** | Defaults to Ollama's 2,048-token limit, causing premature amnesia. | **Auto-detects maximum context** directly from model metadata (or up to 128k+). |
| **Model Hyperparameters** | Forces hardcoded defaults (`temp=0.7`), ignoring creator recommendations. | **Auto-discovers creator settings** from Modelfiles (`top_k`, `min_p`, `repeat_penalty`). |
| **Token Accuracy** | Relies on inaccurate `tiktoken` approximations designed for GPT models. | **Server-native token counting** reads exact evaluation metrics from Ollama. |
| **Reasoning Traces** | Leaks raw `<think>` tokens into conversation text or fails to parse them. | **API-driven thinking controls**: queries `/api/show` for supported effort levels and defaults, rendering traces into clean collapsible UI blocks. |
| **Safety Controls** | All-or-nothing execution without fine-grained user confirmation. | **Human-in-the-Loop approval** before file edits or shell runs, with one-flag YOLO toggle. |

---

## Screenshot Gallery

| Interactive Terminal REPL | Non-Interactive Single-Shot CLI |
| :---: | :---: |
| ![Interactive REPL UI](assets/img/agent_repl_main.png) | ![Non-Interactive CLI](assets/img/agent_noninteractive_main.png) |
| *Full-featured TUI with live streaming markdown, status header, and tool approvals* | *Streamlined one-shot execution directly in your shell for scripts and CI* |

---

## How It Works

Ollama Agent operates with a transparent, user-centered execution loop designed to keep you informed and in control:

```mermaid
flowchart LR
    User([User Prompt]) --> Agent{Ollama Agent}
    Agent --> Model[Local Model via Ollama]
    Model --> Plan[Analyze & Select Tools]
    Plan --> Safety{Safety Gate}
    Safety -- "HITL Approval" --> Confirm[User Confirms Action]
    Safety -- "YOLO Flag (-y)" --> RunTool[Run Tool Directly]
    Confirm --> RunTool
    RunTool --> Execute[Files / Shell / RAG / MCP]
    Execute --> Result[Streamed Markdown Response]
    Result --> Memory[(SQLite History & Memory)]
```

1. **Prompt Ingestion**: Type in the interactive REPL, pass a single query with `-p`, or run a saved task.
2. **Context Resolution**: The agent loads project standards (`AGENTS.md`), user preferences (`MEMORY.md`), and relevant history.
3. **Local Reasoning**: Your local model evaluates instructions and determines necessary tool actions.
4. **Safety Verification**: Interactive confirmation prompts appear before running shell commands or modifying files, unless you explicitly enable YOLO mode (`-y`).
5. **Real-Time Streaming**: Tool outputs, reasoning traces, and markdown responses stream live to your screen.

---

## Practical Everyday Recipes

### 1. Codebase Audit & Refactoring
Inspect project files and refactor legacy patterns while keeping changes local:
```bash
ollama-agent -m "qwen2.5-coder:14b" -e high -p "Audit src/auth.py for insecure token storage and propose fixes."
```

### 2. Interactive `@-Mention` Attachments
In the interactive REPL, attach local files, directories, or media directly using `@`:
```text
>>> Explain how authentication works in @src/auth/jwt.py and compare it to @docs/spec.md
```

### 3. Local Knowledge Base (RAG) Querying
Create a local semantic search database for your documentation and query it on demand:
```bash
# Create knowledge base collection and index local docs
ollama-agent rag create project-docs
ollama-agent rag add project-docs ./docs --dir

# Query the collection via CLI
ollama-agent --rag project-docs -p "How do I configure subagents in settings.yaml?"
```

### 4. Running Reusable Tasks with Variables
Run structured, parameterized tasks with dynamic CLI overrides:
```bash
ollama-agent task run code-review target_file=src/api.py strict=true -y
```

### 5. Instant Clipboard Transformation
Grab clipboard contents, transform them with a local LLM, and output ready-to-use code:
```bash
ollama-agent -p "Read the JSON on my clipboard and convert it into a typed Pydantic model."
```

---

## CLI Reference & Options

| Option | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prompt <text>` | `-p` | *None* | Runs in non-interactive mode with the given prompt and exits. |
| `--model <name>` | `-m` | `settings.yaml` | Name of the local Ollama model to use for the session. |
| `--effort <level>` | `-e` | `default` | Reasoning effort (dynamically matched to model-supported values, e.g. `low`, `high`, `max`, `default`). |
| `--num-ctx <int\|max>`| `-c` | `10000` | Context window size in tokens, or `max` for model capacity. |
| `--yolo` | `-y` | `False` | Autonomous mode: bypasses all tool approval prompts. |
| `--stealth` | `-s` | `False` | Ephemeral mode: does not persist chat history to SQLite. |
| `--rag <collection>` | — | *None* | Preloads a RAG database collection into the agent's context. |
| `--language <code>` | `-l` | *Auto* | UI language code (e.g. `en`, `es`, `fr`, `de`, `zh`, `ja`). |
| `--builtin-tool-timeout`| `-t` | `30` | Timeout in seconds for individual tool and command executions. |
| `--allow-traversal` | — | `False` | Permits filesystem tools to access files outside current directory. |
| `--config-reset <type>`| — | *None* | Resets configuration files: `all`, `system-prompt`, or `config-file`. |

---

## Pro Tips & Best Practices

!!! tip "1. Match Model Size to Your Task"
    For programming, code review, and script automation, models like `qwen2.5-coder:14b` or `qwen2.5-coder:32b` deliver top-tier tool calling and syntax generation. For general writing, research, and analysis, `llama3.1:8b` or `llama3.1:70b` provide broad conceptual reasoning.

!!! tip "2. Drop an `AGENTS.md` into Your Projects"
    Create an `AGENTS.md` file in your project root. Ollama Agent automatically discovers it and adopts your project's coding standards, build commands, and architectural preferences without cluttering every prompt.

!!! tip "3. Use Tab Completion Everywhere"
    In REPL mode, press <kbd>Tab</kbd> to autocomplete slash commands (e.g. `/model`, `/session`, `/context`, `/rag`), entity arguments, and local files prefixed with `@`.

!!! tip "4. Automatic Context Compaction"
    During long troubleshooting or exploratory sessions, Ollama Agent tracks your token usage against your model's context window. When usage exceeds 85%, background compaction summarizes past turns while preserving essential working memory. You can also trigger this manually in the REPL with `/context compact`.

<div class="section-header">
  <h2>Documentation</h2>
  <span>Guides, extensibility &amp; system architecture</span>
</div>

<div class="projects-grid">
  <a href="cli_repl/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-terminal feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>CLI &amp; REPL Interface</h3>
      <p>Master the interactive terminal, slash commands, multiline editor, <code>@</code>-file attachments, and CLI automation.</p>
    </div>
  </a>
  <a href="skills/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-wand-magic-sparkles feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>Agent Skills</h3>
      <p>Extend capabilities with custom procedural workflows and progressive skill discovery following the Agent Skills standard.</p>
    </div>
  </a>
  <a href="tasks/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-list-check feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>Saved Tasks</h3>
      <p>Build and automate repeatable prompt templates with dynamic Jinja2 variables and model overrides.</p>
    </div>
  </a>
  <a href="mcp/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-plug feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>Model Context Protocol (MCP)</h3>
      <p>Connect external tool servers over <code>stdio</code>, <code>http</code>, or <code>sse</code> with hot-reloading and runtime inspection.</p>
    </div>
  </a>
  <a href="subagents/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-sitemap feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>Specialized Subagents</h3>
      <p>Configure dedicated subagents with tailored prompts, models, MCP tools, and isolated context windows.</p>
    </div>
  </a>
  <a href="memory/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-book-bookmark feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>Memory &amp; Guidelines</h3>
      <p>Leverage <code>AGENTS.md</code> project guidelines, persistent user memory (<code>MEMORY.md</code>), and SQLite session history.</p>
    </div>
  </a>
  <a href="rag/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-database feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>Local RAG Engine</h3>
      <p>Build, embed, and query private vector databases from your local documentation and source files.</p>
    </div>
  </a>
  <a href="configuration/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-sliders feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>Configuration Reference</h3>
      <p>Complete guide to <code>settings.yaml</code>, environment variables, context sizing, and model parameter tuning.</p>
    </div>
  </a>
  <a href="architecture/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-diagram-project feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>System Architecture</h3>
      <p>In-depth breakdown of state graphs, persistence, tool middleware, and real-time streaming pipelines.</p>
    </div>
  </a>
  <a href="developer_guide/" class="feature-card">
    <div>
      <div class="card-top">
        <i class="fa-solid fa-code feature-icon" aria-hidden="true"></i>
        <span class="card-badge">Read →</span>
      </div>
      <h3>Developer Guide</h3>
      <p>Development environment setup, codebase structure, testing procedures, and contribution guidelines.</p>
    </div>
  </a>
</div>
