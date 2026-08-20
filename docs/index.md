# Ollama Agent

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![DeepAgents](https://img.shields.io/badge/Framework-DeepAgents-purple.svg)](https://docs.langchain.com/oss/python/deepagents/overview)
[![LangChain](https://img.shields.io/badge/LLM-LangChain-green.svg)](https://github.com/langchain-ai/langchain)
[![Ollama](https://img.shields.io/badge/Backend-Ollama-black.svg)](https://ollama.com/)

**Ollama Agent** is an autonomous local AI assistant that combines an interactive terminal REPL and a non-interactive CLI with deep system tools, subagent delegation, vector-based RAG, long-term memory, and full Model Context Protocol (MCP) support. Built on top of **DeepAgents** and **LangChain**, it runs entirely on your local machine using **Ollama** models while guaranteeing privacy, local execution control, and real-time streaming feedback.

!!! tip "Quick Installation"
    Install Ollama Agent instantly in an isolated Python environment using `pipx`:
    ```bash
    pipx install git+https://github.com/arrase/ollama-agent.git
    ```

---

## Key Features

<div class="projects-grid">
  <div class="feature-card">
    <i class="fa-solid fa-terminal feature-icon"></i>
    <h3>Interactive REPL</h3>
    <p>Modern terminal UI with live Markdown rendering, history navigation, interactive slash commands, mid-session model switching, and prompt autocompletion.</p>
  </div>
  <div class="feature-card">
    <i class="fa-solid fa-square-terminal feature-icon"></i>
    <h3>Shell Execution</h3>
    <p>Integrated local shell backend with configurable directory traversal, sandboxing, environment inheritance, and customizable execution timeouts.</p>
  </div>
  <div class="feature-card">
    <i class="fa-solid fa-database feature-icon"></i>
    <h3>Deep RAG Engine</h3>
    <p>Local vector search database powered by Ollama embeddings and Qdrant backend for semantic document retrieval and context-aware answers.</p>
  </div>
  <div class="feature-card">
    <i class="fa-solid fa-brain feature-icon"></i>
    <h3>Memory & AGENTS.md</h3>
    <p>Native memory layer backed by <code>MEMORY.md</code> and project-level <code>AGENTS.md</code> standard for repository guidelines and persistent context.</p>
  </div>

  <div class="feature-card">
    <i class="fa-solid fa-sitemap feature-icon"></i>
    <h3>Custom Subagents</h3>
    <p>Define specialized subagents in <code>settings.yaml</code> with isolated context windows, specialized system prompts, dedicated models, and MCP tool sets.</p>
  </div>
  <div class="feature-card">
    <i class="fa-solid fa-plug feature-icon"></i>
    <h3>MCP Integration</h3>
    <p>Native support for Model Context Protocol servers to dynamically extend agent capabilities with external tools, APIs, and databases.</p>
  </div>
  <div class="feature-card">
    <i class="fa-solid fa-wand-magic-sparkles feature-icon"></i>
    <h3>Agent Skills</h3>
    <p>Reusable, modular skill definitions adhering to the Agent Skills spec with progressive disclosure and dynamic context loading.</p>
  </div>
  <div class="feature-card">
    <i class="fa-solid fa-list-check feature-icon"></i>
    <h3>Task Management</h3>
    <p>Save, configure, and re-execute multi-step prompts with custom model choices and reasoning effort parameters on demand.</p>
  </div>
  <div class="feature-card">
    <i class="fa-solid fa-shield-halved feature-icon"></i>
    <h3>HITL & YOLO Mode</h3>
    <p>Human-in-the-Loop approval workflow for destructive shell & file operations with optional zero-friction YOLO mode toggle.</p>
  </div>
</div>

---

## Screenshot Gallery

| Interactive REPL | Non-Interactive Prompt |
| :---: | :---: |
| ![Interactive REPL UI](assets/img/agent_repl_main.png) | ![Non-Interactive CLI](assets/img/agent_noninteractive_main.png) |
| *Full REPL interface with markdown rendering, status bar & tool calls* | *Single prompt execution from CLI with structured output* |

---

## Documentation Index

Explore the complete technical guides for Ollama Agent:

- **[System Architecture](architecture.md)**: Graph orchestration, SQLite state persistence, streaming parsers, shell middleware, and reasoning trace capture.
- **[CLI & REPL User Guide](cli_repl.md)**: Terminal interface commands, keyboard shortcuts, `@`-file mentions, multiline inputs, and non-interactive usage.
- **[MCP & Subagents Architecture](mcp_subagents.md)**: Model Context Protocol setup, subagent configuration, and isolated task delegation.
- **[RAG Engine Guide](rag.md)**: Local vector database creation, document chunking, embeddings setup with Ollama & Qdrant.
- **[Skills, Tasks & Memory](skills_tasks_memory.md)**: Authoring reusable skills, managing saved task templates, `AGENTS.md` project rules, and configuring persistent `MEMORY.md`.
- **[Configuration & Tracing](configuration.md)**: Comprehensive `settings.yaml` reference, context window auto-resolution, and model reasoning effort levels.
