# CLI & REPL Interface Guide

`ollama-agent` offers two operational interfaces for interacting with local LLMs: an interactive terminal user interface (**REPL**) powered by Textual and Rich, and a non-interactive command-line interface (**CLI**) for automation and single-shot queries.

---

## Interactive REPL vs Non-Interactive CLI Mode

### Interactive REPL Mode

The REPL (Read-Eval-Print Loop) is the default mode when launching `ollama-agent` without prompt arguments. It provides a full terminal workspace featuring:

* **Stateful Sessions**: Multi-turn conversation history stored and checkpointed in SQLite (`~/.ollama-agent/history.db`).
* **Rich Markdown Formatting**: Real-time streaming output with syntax-highlighted code blocks, thinking containers, and status cards.
* **Live Context & Token Gauge**: Dynamic header showing consumed tokens vs. model context limit (`num_ctx`) with color-coded alert thresholds.
* **Context Compaction**: Reclaim tokens on demand via `/compact` or `/compress` with persistent history offloading to `/conversation_history/session_<uuid>.md`.
* **Human-in-the-Loop (HITL) Approvals**: Inline approval widgets before executing shell commands or editing files, with YOLO mode bypass.
* **3-Level Tab Autocompletion**: Autocompletion for slash commands, subcommands, entities (models, sessions, tasks, skills, RAG databases), and `@-mention` file paths.
* **System Clipboard Integration**: Native copy and paste across macOS, Linux (Wayland / X11), and Windows.

To start the REPL:
```bash
ollama-agent
```

### Non-Interactive CLI Mode

Non-interactive mode enables single-shot execution directly from your terminal or shell scripts. When provided with `-p` or `--prompt`, `ollama-agent` runs the input, processes tool calls, streams the output directly to standard output, and exits cleanly.

```bash
# Basic single-shot query
ollama-agent -p "Summarize the git commits made in the last 7 days."

# Advanced non-interactive query with model, effort, timeout, language, and YOLO mode
ollama-agent -m "gemma4:26b" -e "high" -t 60 -l "es" -y -p "Refactor src/utils.py to follow PEP 8."

# Run a query against a preloaded RAG database
ollama-agent --rag project-docs -p "How is authentication configured in this repository?"
```

> [!NOTE]
> The `-p` / `--prompt` option runs single-shot execution and cannot be combined with subcommands (such as `task`, `rag`, `skill`, or `session`).

---

## CLI Reference & Options

### Global Flags & Options

| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | `-m` | `str` | `settings.yaml` | Specify the Ollama model for this session (falls back to interactive selection if unconfigured or missing in Ollama). |
| `--prompt` | `-p` | `str` | `None` | Run in non-interactive mode with the provided prompt. |
| `--effort` | `-e` | `str` | `medium` | Set reasoning effort level (`low`, `medium`, `high`, `xhigh`, `disabled`, `hide`, `enabled`). |
| `--num-ctx` | `-c` | `int \| str` | `10000` | Set context window size in tokens (`num_ctx`) or `'max'`. |
| `--lang`, `--language` | `-l` | `str` | `auto` | Set UI language code (`en`, `es`, `fr`, `de`, `it`, `pt`, `zh`, `ja`, `ko`, `ru`, `hi`, `ar`, `tr`, `pl`, `nl`, `uk`). |
| `--builtin-tool-timeout` | `-t` | `int` | `30` | Timeout in seconds for tool executions (including shell commands). |
| `--yolo` | `-y` | `flag` | `False` | Enable YOLO mode (bypasses all tool approval prompts). |
| `--rag` | — | `str` | `None` | Preload a RAG database collection at startup. |
| `--allow-traversal` | — | `flag` | `False` | Allow filesystem traversal outside current working directory. |
| `--no-allow-traversal` | — | `flag` | `True` | Sandbox filesystem operations to current working directory (default). |
| `--config-reset` | — | `str` | `None` | Reset configuration files: `all`, `system-prompt`, or `config-file`. |

---

### CLI Subcommands

`ollama-agent` provides subcommands for managing tasks, RAG databases, skills, and chat sessions:

#### 1. Task Commands
```bash
# List all saved tasks
ollama-agent task list

# Create a new saved task
ollama-agent task create code-review \
    --title "Code Review Assistant" \
    --task-prompt "Review the git diff against main and highlight bugs, complexity, and styling issues." \
    --task-model "gemma4:26b" \
    --task-effort "high" \
    [--force]

# Execute a saved task (with optional YOLO mode)
ollama-agent task run code-review -y

# Delete a saved task
ollama-agent task delete code-review
```

#### 2. RAG Commands
```bash
# List all RAG vector databases
ollama-agent rag list

# Create a new RAG database
ollama-agent rag create project-docs

# Ingest a single file or an entire directory
ollama-agent rag add project-docs ./docs/architecture.md
ollama-agent rag add project-docs ./src --dir

# Delete a RAG database
ollama-agent rag delete project-docs
```

#### 3. Skill Commands
```bash
# List all available skills
ollama-agent skill list

# Show details and instructions for a skill
ollama-agent skill show api-design

# Create a new skill
ollama-agent skill create api-design \
    --name "API Design Guidelines" \
    --description "RESTful and OpenAPI standards" \
    --instructions "Ensure all endpoints use nouns and camelCase properties." \
    [--force]

# Delete a skill
ollama-agent skill delete api-design
```

#### 4. Session Commands
```bash
# List all saved chat sessions with step counts
ollama-agent session list

# Search past sessions by keyword
ollama-agent session search "dockerize fastapi"

# Export a session to a Markdown document
ollama-agent session export 4d7e2a1b -o ./exports/session_summary.md

# Delete a session from SQLite history
ollama-agent session delete 4d7e2a1b
```

#### 5. MCP (Model Context Protocol) Commands
```bash
# List all configured MCP servers and check their connection status
ollama-agent mcp list
```

---

## Interactive REPL Reference

The interactive REPL is powered by Textual and Rich, featuring rich Markdown rendering, live token gauges, modal forms, and multi-turn persistence.

```text
● ollama-agent │ Model: gemma4:26b │ Context: 2.1k/10.0k (21%) │ Effort: medium │ YOLO: OFF
```

### Slash Commands Reference

Slash commands provide full application control directly within the REPL:

| Command | Subcommands / Syntax | Description |
| :--- | :--- | :--- |
| `/model` | `/model [list \| set <model>]` | List available Ollama models (with tool support indicators) or switch active model for current session. |
| `/effort` | `/effort [<level>]` | Show current reasoning effort or change thinking/reasoning effort mid-session (`low`, `medium`, `high`, `xhigh`, `disabled`, `hide`, `enabled`). |
| `/context` | `/context [<size\|max>]` | Show current context window or switch context window token size (`num_ctx`) or `'max'` for the active session. |
| `/params` | `/params [list \| set <parameter> <value>]` | Inspect active sampling parameters and resolution sources, or dynamically update parameter values for the active session. |
| `/queue` | `/queue [list \| clear \| rm <position>]` | Inspect pending prompts in the queue, remove an item by index (`/queue rm <id>`, aliases: `remove`, `delete`), or clear all queued prompts. |
| `/session` | `/session [list \| search <query> \| resume <id> (alias: switch) \| new \| export [path] \| delete <id>]` | Manage persistent chat sessions. Search past conversations, resume threads, export to Markdown, or delete history. |
| `/compact` | `/compact` (alias: `/compress`) | Manually compact conversation history into a structured summary to reclaim context window tokens. |
| `/task` | `/task [list \| create [<id>] \| run <id> [-y] \| delete <id>]` | Manage saved prompt tasks. `/task create` launches an interactive conversational creation flow with the agent. |
| `/skill` | `/skill [list \| show <id> \| create [<id>] \| delete <id>]` | Manage agent skills. `/skill create` launches an interactive conversational creation flow with the agent. |
| `/rag` | `/rag [status \| list \| create <name> \| load <name> \| unload \| add <path> [--dir] \| delete <name>]` | Manage local RAG databases, index documents, and toggle active knowledge bases. |
| `/mcp` | `/mcp [list \| reload]` | List configured MCP servers, check connection status, or reload MCP servers and rebuild tool graph mid-session. |
| `/agents` | `/agents [list]` | List configured specialized subagents, inspecting their models, context windows, and dedicated MCP tool servers. |
| `/yolo` | `/yolo [on \| off]` | Toggle YOLO mode or set it explicitly to bypass tool execution confirmation prompts. |
| `/new` | `/new` (alias: `/clear`) | Start a clean session with fresh context and clear the screen (alias for `/session new`). |
| `/clear` | `/clear` | Clear the screen and start a clean session (alias for `/new`). |
| `/exit` | `/exit` (alias: `/quit`) | Exit the application cleanly. |

---

### Multiline Input & Keyboard Navigation

The prompt input box (`ReplInput`) provides intuitive editing, history navigation, and keybindings:

* **Insert Newline (`\ + Enter`)**: End any line with a backslash `\` and press `Enter`. The trailing backslash is automatically removed, inserting a clean newline. The input container dynamically expands up to 8 lines.
* **Submit Prompt (`Enter`)**: Press `Enter` without a trailing backslash to submit your message.
* **Cursor Navigation (`↑` / `↓`)**: Move freely between lines in multiline text.
* **Command History**: Pressing `↑` anywhere on row 0 recalls prior user prompts; pressing `↓` on the last line at the end of the text navigates forward. Slash commands (`/cmd`) are filtered out from stored history.
* **Tab Autocompletion (`Tab`)**: Activates 3-level autocompletion:
  1. *Level 0*: Root slash commands (`/mo` -> `/model`, `/co` -> `/compact`, `/qu` -> `/queue`).
  2. *Level 1*: Subcommands (`/task ` -> `list`, `create`, `run`, `delete`, `/queue ` -> `clear`, `rm`, `remove`, `delete`).
  3. *Level 2*: Dynamic entities:
     - `/model set ` -> Dynamic list of available Ollama models + disk size.
     - `/queue rm `, `/queue remove `, `/queue delete ` -> Dynamic list of active queued prompt IDs (`#1`, `#2`, ...) + preview text.
     - `/task run `, `/task delete ` -> Dynamic list of saved task IDs + titles.
     - `/skill show `, `/skill delete ` -> Dynamic list of discovered skill IDs + names.
     - `/session resume `, `/session switch `, `/session delete ` -> Dynamic list of session IDs + step counts.
     - `/rag load `, `/rag delete ` -> Dynamic list of RAG databases + chunk counts.
  4. *Filesystem*: Path autocompletion for `@-mentions` with directory traversal.
* **Interrupt / Cancel (`Esc` / `Ctrl+C`)**: `Esc` cancels active generation or tool approvals and purges the prompt queue (or dismisses autocompletion). `Ctrl+C` cancels generation and queue if active, or exits the REPL if idle.
* **Clipboard Shortcuts**:
  - Copy: `Super+C`, `Ctrl+Shift+C`, `Ctrl+Insert`, or mouse selection.
  - Paste: `Super+V`, `Ctrl+V`, `Shift+Insert`.

---

### Prompt Queue & Non-Blocking Execution

The REPL is designed with an asynchronous non-blocking event loop. The user input field remains accessible and interactive at all times, allowing you to submit commands and prompts while inference is actively streaming or while a tool approval prompt is waiting for user confirmation.

#### Immediate (Non-Blocking) Commands
Slash commands that perform read-only queries or instant state toggles execute immediately in the chat viewport without waiting for the active stream to complete:

* **Inspection & Queue Removal**: `/queue`, `/queue rm <position>`, `/model list`, `/effort`, `/context`, `/params list`, `/session list`, `/session search`, `/session export`, `/task list`, `/skill list`, `/skill show`, `/rag status`, `/rag list`, `/mcp list`, `/agents list`.
* **Toggles & Exit**: `/yolo`, `/exit`, `/quit`.

#### Enqueued Prompts & Stateful Commands
Normal chat prompts and commands that mutate graph state (e.g., `/model set`, `/compact`, `/session resume`, `/session new`, `/task run`, `/skill create`) are placed in a FIFO queue:

* **Queue Feedback**: Submitting an item while busy renders a subtle notification (`⏳ Prompt added to queue (position #N)`) and updates the footer counter (`⏳ N queued`).
* **Persistent TUI Queue Panel**: A dedicated `PromptQueueWidget` card renders above the input container whenever items are queued, showing prompt previews and position numbers in real time.
* **FIFO Draining**: As soon as the active stream or tool execution completes, the next queued item is automatically dispatched.
* **Unblocked Tool Approvals**: The prompt input box is not locked while a `ToolApprovalWidget` modal is displayed, allowing you to queue follow-up prompts while reviewing pending tool actions.
* **Managing the Queue**:
  * Run `/queue` to inspect all pending prompts and their indices.
  * Run `/queue rm <position>` (aliases: `remove`, `delete`) to remove a single prompt without interrupting active inference.
  * Run `/queue clear` to purge all queued items while letting active inference continue.
  * Press `Esc` or `Ctrl+C` to cancel current generation or tool approvals and purge the queue simultaneously.

---

### Live Context Usage & Token Gauge

The dynamic header bar monitors token consumption and model parameters in real time:

```text
● ollama-agent │ Model: gemma4:26b │ Context: 3.4k/10.0k (34%) │ Effort: medium │ RAG: my-docs │ YOLO: OFF
```

* **Metrics**: Displays consumed tokens vs. effective context window limit (`num_ctx`), formatted with `k` suffixes.
* **Visual Alert Thresholds**:
  - 🔵 **Cyan / Sky Blue (`#38bdf8`)**: Healthy context utilization (`≤ 75%`).
  - 🟡 **Yellow / Amber (`#fbbf24`)**: Elevated context warning (`76% – 90%`).
  - 🔴 **Red (`#f87171`)**: Critical limit proximity (`> 90%`).
* **Dynamic Indicators**: Displays active RAG database in purple (`#a78bfa`) when loaded, reasoning effort level, and highlighted YOLO status badge.

---

### Context Compression & Compaction (`/compact`)

To prevent conversation degradation and context overflow errors:

1. **Automatic Background Summarization**:
   - Triggers automatically when conversation tokens reach **85%** of `max_input_tokens`.
   - Compresses older turns into a structured summary while keeping the most recent **10%** of tokens (or 6 messages) intact.
   - Large tool arguments are truncated to 2,000 characters.
   - Evicted turns are appended to `/conversation_history/session_<uuid>.md`.
2. **On-Demand Compaction (`/compact` or `/compress`)**:
   - Type `/compact` anytime in the REPL to immediately compress prior messages, preserve the last 2 messages (`KEEP_RECENT_MESSAGES = 2`), offload history, and refresh the token gauge:

```text
❯ /compact
⚡ Compacting conversation context...
✓ Context compacted successfully:
  • Messages summarized: 14
  • Recent messages preserved: 2
  • History offloaded to: /conversation_history/session_9f86d081884c7d659a2feaa0c55ad015.md
```

---

### File & Directory Context (`@-mentions`)

Reference files or entire folder trees directly inside your prompts using `@` syntax. The agent resolves the paths and injects the contents into the model's context.

* **Single Files**: `@filename.txt`, `@src/main.py`
* **Quoted Paths (with spaces)**: `@"my notes/todo.txt"` or `@'my notes/todo.txt'`
* **Directory Traversal**: `@src` or `@.` (recursively reads all supported text files within the directory).
* **Interactive Autocompletion**: Type `@` and press `Tab` in the REPL to interactively search and insert file paths.

#### Supported Content Types
* **Text Files**: Read as UTF-8 and attached as structured `<context_file path="...">...</context_file>` blocks.
* **Multimodal Attachments**: Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.svg`, `.heic`, `.heif`), audio (`.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.aac`, `.aiff`), video (`.mp4`, `.mpeg`, `.mov`, `.avi`, `.flv`, `.mpg`, `.webm`, `.wmv`, `.3gpp`), and documents (`.pdf`, `.ppt`, `.pptx`) are base64-encoded and attached as native multimodal inputs.
* **Binary Safety**: Non-multimodal binaries containing null bytes are safely blocked from direct references and skipped during directory traversal.

#### Safety Limits & Configuration
Configurable under the `mentions` section in `~/.ollama-agent/settings.yaml`:

```yaml
mentions:
  max_file_size: 1048576      # Max single file size (default: 1 MB)
  max_files: 100               # Max files loaded in directory traversal (default: 100)
  max_total_size: 10485760     # Max total attached context size (default: 10 MB)
  max_completions: 200         # Max autocompletion candidates (default: 200)
```

#### Decorator & Syntax Safety
Common programming decorators (e.g. `@staticmethod`, `@property`, `@decorator`) that do not exist as files on disk are recognized and treated as literal text. If a missing path contains directory separators (e.g. `@src/missing.py`) or a file extension (e.g. `@app.py`), the agent immediately halts and reports a clear `File or directory not found` error.

---

### Human-in-the-Loop (HITL) & YOLO Mode

To ensure safety when interacting with your local system, Ollama Agent enforces a Human-in-the-Loop confirmation policy before executing potentially sensitive operations (such as running shell commands via `execute` or modifying files via `write_file` and `edit_file`).

```text
╭─ ⚠ Action Approval Required ────────────────────────────────────────────────╮
│ Tool: execute                                                               │
│ Arguments: {'command': 'pytest tests/'}                                     │
╰─────────────────────────────────────────────────────────────────────────────╯
 Approve (y)    Reject (n)    Allow Session (a)    Cancel (c)
```

The approval dialog provides full keyboard-first navigation without requiring a mouse:

* **Direct Keyboard Shortcuts**:
  - **Approve (`y`)**: Authorize this single tool execution.
  - **Reject (`n`)**: Block execution and provide feedback to the agent so it can select an alternative approach.
  - **Allow Session (`a`)**: Approve this call and automatically authorize all subsequent calls for this specific tool for the remainder of the active session.
  - **Cancel (`c`)**: Abort the execution and return focus to the prompt input.
* **Arrow & Tab Navigation**: Use `←` / `→` / `↑` / `↓` arrow keys or `Tab` / `Shift+Tab` to cycle focus between buttons with high-contrast visual indicators.
* **Default Action (`Enter` / `Space`)**: The `Approve (y)` button is focused automatically upon appearance; pressing `Enter` or `Space` immediately executes the focused button.
* **Live Footer Status**: The bottom status bar automatically switches to display the confirmation key guide whenever an approval is pending.

#### YOLO Mode
When you want autonomous execution without confirmation pauses:
* **CLI Flag**: Start the agent with `-y` or `--yolo` (e.g. `ollama-agent -y`).
* **REPL Slash Command**: Toggle dynamically with `/yolo` or set explicitly via `/yolo on` and `/yolo off`.

When YOLO mode is active:
1. Tool approval prompts are bypassed automatically.
2. The header displays `YOLO: ON` with a highlighted badge.
3. The prompt chevron (`❯ `) and input box border change color to **red** for clear visual status.

---

### System Clipboard Integration

Ollama Agent features seamless cross-platform clipboard integration:

* **Copy Selection**: Select text with your mouse or keyboard in the TUI, or press `Super+C`, `Ctrl+Shift+C`, or `Ctrl+Insert` to copy directly to the OS system clipboard.
* **Paste Input**: Use `Super+V`, `Ctrl+V`, or `Shift+Insert` to paste clipboard text into the prompt.
* **Native Tool Backends**: Uses `pbcopy`/`pbpaste` on macOS, `wl-copy`/`wl-paste` on Linux Wayland, `xclip`/`xsel` on Linux X11, and `clip`/PowerShell on Windows.

---

### Thinking / Reasoning Effort Controls

The `--effort` flag (and `model.reasoning_effort` in `settings.yaml`) controls model reasoning traces via Ollama's native thinking capabilities:

| Model Family | `--effort` Value | Ollama API Parameter | Behavior |
| :--- | :--- | :--- | :--- |
| **Qwen3.8 Series** | `xhigh` | `"xhigh"` | Default level. Thorough reasoning for complex analysis. |
| **Qwen3.8 Series** | `medium` | `"medium"` | Balanced reasoning optimizing accuracy and speed. |
| **Qwen3.8 Series** | `low` | `"low"` | Efficient reasoning optimizing for speed and cost. |
| **Qwen3.8 Series** | `enabled` | `"xhigh"` | Enables reasoning with Qwen3.8 default `xhigh` level. |
| **Qwen3.8 Series** | `hide` | `true` | Generates reasoning trace but collapses/hides it from the UI. |
| **Qwen3.8 Series** | `disabled` | `false` | Disables reasoning trace generation at the model level. |
| **GPT-OSS** | `low` / `medium` / `high` / `xhigh` | `"low"` / `"medium"` / `"high"` / `"xhigh"` | Sets thinking trace depth. GPT-OSS accepts string effort levels. |
| **GPT-OSS** | `enabled` | `"medium"` | Enables thinking with default `medium` level. |
| **GPT-OSS** | `hide` | *(omitted)* | Uses model default effort and hides reasoning trace in UI. |
| **GPT-OSS** | `disabled` | *(omitted)* | GPT-OSS cannot disable thinking; emits warning, uses default effort, and hides reasoning trace in UI. |
| **Binary Reasoning Models**<br>*(Qwen 2.5 / 3, Gemma 4, DeepSeek R1, DeepSeek-v3.1)* | `low` / `medium` / `high` / `xhigh` / `enabled` | `true` | Enables native reasoning generation. |
| **Binary Reasoning Models**<br>*(Qwen 2.5 / 3, Gemma 4, DeepSeek R1, DeepSeek-v3.1)* | `hide` | `true` | Generates reasoning trace but collapses/hides it from the UI. |
| **Binary Reasoning Models**<br>*(Qwen 2.5 / 3, Gemma 4, DeepSeek R1, DeepSeek-v3.1)* | `disabled` | `false` | Disables reasoning trace generation at the model level. |
| **Non-Thinking Models** | *(any)* | *(omitted)* | Setting is ignored gracefully. |
