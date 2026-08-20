# Ollama Agent Architectural Overview

Ollama Agent is designed around a modular, event-driven architecture that bridges local LLM inference engines (via Ollama and LangChain) with stateful graph orchestration (via DeepAgents and LangGraph). This document outlines the core system design, execution pipeline, persistence layer, tool middleware, context compaction engine, and streaming parsers.

---

## High-Level Architecture

The system uses a layered architecture where user interactions (CLI or REPL UI) trigger asynchronous event streams through a stateful graph. The graph coordinates tool invocation, memory read/writes, RAG queries, and subagent delegation while maintaining human-in-the-loop (HITL) checkpoints.

```mermaid
flowchart TD
    subgraph UI ["User Interface Layer"]
        REPL["Interactive REPL UI (Textual / Rich)"]
        CLI["Non-Interactive CLI (argparse)"]
    end

    subgraph Core ["Agent Runtime & Graph Engine"]
        Runtime["AgentRuntime (State Manager & AsyncExitStack)"]
        Graph["DeepAgents Graph (create_deep_agent)"]
        Checkpointer["AsyncSqliteSaver (~/.ollama-agent/history.db)"]
    end

    subgraph Middleware ["Execution & Control Layer"]
        ToolMW["ShellToolMiddleware (stream_tool_events_mw)"]
        SummarizerMW["Summarization Middleware (create_summarization_tool_middleware)"]
        HITL["Human-in-the-Loop Interrupt Controller"]
    end

    subgraph Adapters ["Integration & Backend Adapters"]
        OllamaLLM["LangChain ChatOllama"]
        ShellBackend["LocalShellBackend / CompositeBackend"]
        RAGEngine["Qdrant Vector Store & Ollama Embeddings"]
        MCPAdapter["MCP Server Adapters (mcp_servers.json)"]
        MemoryStore["FilesystemBackend (/agent/MEMORY.md, AGENTS.md, /skills/)"]
    end

    REPL --> Runtime
    CLI --> Runtime
    Runtime --> Graph
    Graph <--> Checkpointer
    Graph --> ToolMW
    Graph --> SummarizerMW
    Graph --> HITL
    ToolMW --> ShellBackend
    ToolMW --> MCPAdapter
    ToolMW --> RAGEngine
    Graph --> OllamaLLM
    Graph --> MemoryStore
```

---

## Component Breakdowns

### 1. DeepAgents Graph Integration & Backend Routing

The core agent state machine is built using **DeepAgents** (`deepagents.create_deep_agent`), which compiles a LangGraph state graph configured with specialized backends, system prompts, memory layers, and tool subnets.

```mermaid
sequenceDiagram
    autonumber
    participant UI as Terminal REPL / CLI
    participant Runtime as AgentRuntime
    participant Backend as CompositeBackend
    participant Graph as DeepAgents Graph
    participant LLM as Ollama LLM

    UI->>Runtime: reload() / run_streamed(prompt)
    Runtime->>Backend: Initialize CompositeBackend (Shell + Virtual /agent/ + /skills/ + /project/)
    Runtime->>Graph: create_deep_agent(model, tools, backend, checkpointer, interrupt_on)
    UI->>Graph: astream(inputs, config, stream_mode=['messages', 'custom'])
    Graph->>LLM: Generate response / tool calls
    LLM-->>Graph: Tool Call Request
    Graph-->>Runtime: Emit tool_call stream event
    Graph-->>UI: Yield text & reasoning deltas
```

#### Graph Construction Details
- **Lifecycle Management**: `AgentRuntime` owns an internal `AsyncExitStack` to manage resources (SQLite database connections, MCP process pipes, and HTTP sessions). Calling `reload()` gracefully tears down existing resources and re-instantiates the graph.
- **Backend Composition**: A `CompositeBackend` routes filesystem and tool requests:
  - `/agent/`: Routed to `FilesystemBackend` pointing to `~/.ollama-agent/` (e.g. `MEMORY.md`, global `AGENTS.md`).
  - `/skills/`: Routed to `FilesystemBackend` pointing to `~/.ollama-agent/skills/`.
  - `/project/`: Optional route to `FilesystemBackend` pointing to repository root when `AGENTS.md` is in an ancestor directory.
  - Default route: `LocalShellBackend` operating on the current working directory (`Path.cwd()`).
- **Dynamic System Instructions**: The system prompt is constructed dynamically by blending base instructions, filesystem policy directives (traversal mode vs sandboxed mode), dynamic RAG search policies, and local environment runtime metadata (`platform.system()`, `platform.release()`).

---

### 2. Context Compression & Compaction Engine

To prevent conversation degradation and context overflow errors, Ollama Agent integrates both automatic background summarization and on-demand context compaction.

```mermaid
flowchart LR
    A[Conversation Turns] -->|Auto at 85% num_ctx OR /compact| B[Summarization Engine]
    B --> C[Structured Summary\n• Session Intent\n• Key Decisions\n• Artifacts\n• Next Steps]
    B --> D[Durable History Saved to\n<cwd>/conversation_history/{thread_id}.md]
    C --> E[Reclaimed Context Window]
```

1. **Summarization Middleware**: Configured via `create_summarization_tool_middleware(model, backend)` from `deepagents.middleware.summarization`.
2. **Automatic Background Summarization**:
   - **Trigger Threshold**: Automatically runs when conversation tokens reach **85%** of the model's configured context window (`num_ctx`).
   - **Token Retention**: Older messages are compressed into a structured summary, preserving the most recent **10%** of tokens intact.
   - **Tool Argument Pruning**: Large arguments in past tool calls (e.g. file contents in `write_file` / `edit_file`) are truncated to 2,000 characters.
   - **Durable History Offloading**: Evicted messages are written to `<cwd>/conversation_history/{thread_id}.md` for long-term recovery.
3. **On-Demand Compaction (`/compact` or `/compress`)**:
   - Manually triggered anytime via `AgentRuntime.compact_context()`.
   - Recomputes approximate token counts using `count_tokens_approximately()` and refreshes the TUI context gauge immediately.

---

### 3. State Persistence via `langgraph-checkpoint-sqlite`

Session persistence is handled by `AsyncSqliteSaver` from `langgraph-checkpoint-sqlite`.

```mermaid
flowchart LR
    subgraph Storage ["Persistent Storage"]
        DB[("~/.ollama-agent/history.db")]
    end

    subgraph Sessions ["Session Threads"]
        T1["Thread ID: session-abc"]
        T2["Thread ID: session-xyz"]
    end

    subgraph Runtime ["Agent Graph Execution"]
        GraphState["Graph State & Message History"]
        InterruptState["Interrupt & Decision Checkpoint"]
    end

    T1 --> DB
    T2 --> DB
    DB <--> GraphState
    DB <--> InterruptState
```

- **Thread Tracking**: Each chat session is assigned a unique `thread_id`. State snapshots are written to SQLite after every node execution step in the graph.
- **Mid-Session Continuation**: When changing models mid-conversation via `/model set <model>`, the thread configuration (`{"configurable": {"thread_id": thread}}`) is passed to `astream()`, preserving conversation state without losing context.
- **Session Export & Resumption**: Past sessions can be inspected, resumed with full state restoration (`/session resume <id>`), exported to structured Markdown documents (`/session export`), or deleted (`/session delete <id>`).
- **HITL Checkpoints**: When execution is paused for user confirmation (`interrupt_on`), the graph state is snapshotted in SQLite. Resuming execution sends a `Command(resume={"decisions": ...})` payload back to the same thread ID.

---

### 4. Streaming Responses & Event Processing

Ollama Agent processes inference and execution in real time by listening to LangGraph event streams.

```mermaid
flowchart TD
    A["graph.astream(inputs, stream_mode=['messages', 'custom'])"] --> B{"Event Mode?"}
    
    B -- "custom" --> C["Emit Tool Events (tool_call / tool_output)"]
    B -- "messages" --> D["Extract Message Chunk & Metadata"]
    
    D --> E["Track Token Count (prompt_eval_count + eval_count)"]
    D --> F["streaming_reasoning(content, additional_kwargs)"]
    D --> G["streaming_text(content)"]
    
    F -- "Reasoning Delta" --> H["Render Thinking Trace in UI"]
    G -- "Text Delta" --> I["Render Markdown Response in UI"]
    
    C --> J["Update Terminal Tool Status Widget"]
```

- **Dual-Stream Listening**: The agent streams both `messages` (raw LLM token outputs) and `custom` events (tool middleware status updates).
- **Token Consumption Tracking**: Inspects `response_metadata` for `prompt_eval_count` and `eval_count` to maintain accurate `last_context_tokens` metrics for the live gauge.
- **Text Delta Extraction**: `streaming_text()` extracts text content regardless of payload shape (handles raw strings, single dicts, or lists of text blocks).
- **Reasoning Delta Extraction**: `streaming_reasoning()` extracts thinking content from `additional_kwargs['reasoning_content']` or OpenAI-style reasoning blocks.

---

### 5. ShellToolMiddleware & Command Execution

Command execution is managed by custom middleware (`stream_tool_events_mw`) built using `langchain.agents.middleware.wrap_tool_call`.

```mermaid
sequenceDiagram
    autonumber
    participant Graph as DeepAgents Graph
    participant MW as stream_tool_events_mw
    participant Writer as runtime.stream_writer
    participant Handler as Tool Handler / Shell
    
    Graph->>MW: Invoke Tool Request
    MW->>Writer: Emit event {"type": "tool_call", "name": tool_name, "agent_name": agent_name}
    alt Execution within Timeout
        MW->>Handler: asyncio.wait_for(handler(request), timeout)
        Handler-->>MW: Tool Execution Result
        MW->>Writer: Emit event {"type": "tool_output", "output_len": len, "agent_name": agent_name}
        MW-->>Graph: Return Tool Output
    else Execution Timeout
        MW->>MW: TimeoutError Raised
        MW-->>Graph: Return Timeout Error Message
    end
```

- **Tool Call Emitting**: Emits structured UI events before tool execution starts (`tool_call`) and after completion (`tool_output`), allowing the terminal renderer to update spinners, status lines, and subagent attribution tags.
- **Timeout Protection**: Wraps execution in `asyncio.wait_for(timeout=builtin_tool_timeout)` to prevent hanging sub-processes or stuck tool calls.
- **Security Policies**: Respects the `runtime.allow_traversal` setting:
  - `True`: `LocalShellBackend` allows execution across the host filesystem.
  - `False`: `LocalShellBackend` enforces virtual mode sandboxing restricted to the current working directory.

---

### 6. Ollama Thinking Trace Capture

The agent incorporates reasoning capabilities from models such as DeepSeek R1, Qwen 3, and GPT-OSS.

```mermaid
flowchart TD
    A["Model Selected"] --> B["get_model_capabilities(model, base_url)"]
    B --> C{"Supports 'thinking'?"}
    
    C -- Yes --> D["resolve_ollama_reasoning()"]
    C -- No --> E["Disable Reasoning Engine"]
    
    D --> F{"Model Family"}
    F -- "GPT-OSS" --> G["Map effort ('low' | 'medium' | 'high') to Ollama think parameter"]
    F -- "Standard Thinking Model" --> H["Map effort to boolean true/false"]
    
    G --> I["ChatOllama Request"]
    H --> I
    
    I --> J["Parse Response"]
    J --> K{"reasoning_effort Setting"}
    
    K -- "hide / disabled" --> L["Suppress Thinking Output from UI"]
    K -- "low / medium / high / enabled" --> M["Stream Thinking Trace to UI Collapsible Block"]
```

- **Capability Detection**: Queries `ollama.AsyncClient.show()` to inspect model capabilities for the `thinking` flag.
- **Reasoning Effort Translation**:
  - `GPT-OSS` models receive string values (`"low"`, `"medium"`, `"high"`).
  - General reasoning models receive boolean flags (`true` / `false`).
- **UI Filtering**: When `reasoning_effort` is set to `hide` or `disabled`, reasoning chunks extracted by `streaming_reasoning()` are filtered out before reaching the UI layer.

---

### 7. Custom Subagents Architecture

Subagents are auxiliary AI agent instances configured in `settings.yaml` to handle specialized subtasks with isolated context windows:

```mermaid
flowchart TD
    MainAgent["Main Agent (ollama-agent)"] -->|Delegates Task| SubagentGraph["Subagent Graph"]
    
    subgraph SubagentGraph ["Subagent Execution Environment"]
        SubModel["Custom Ollama Model Instance"]
        SubPrompt["Isolated System Prompt + OS Info"]
        SubSkills["Mounted Skills (/skills/)"]
        SubMCP["Dedicated MCP Tools (load_subagent_mcp_tools)"]
    end
```

- **Isolated Execution**: Subagents run on separate graph nodes with independent context windows and custom system prompts with OS environment details.
- **Dedicated Tools**: Subagent MCP servers are loaded independently via `load_subagent_mcp_tools()` and isolated from the main agent's tool set.
- **Attribution**: Tool execution middleware attaches `agent_name` metadata to `tool_call` and `tool_output` events for clear attribution in the UI.
