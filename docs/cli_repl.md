# CLI & REPL Interface Guide

`ollama-agent` offers two operational interfaces for interacting with local LLMs: an interactive terminal user interface (**REPL**) powered by Textual and a non-interactive command-line interface (**CLI**) for automation and single-shot queries.

---

## Interactive REPL vs Non-Interactive CLI Mode

### Interactive REPL Mode

The REPL (Read-Eval-Print Loop) is the default mode when launching `ollama-agent` without positional prompt arguments. It provides a rich terminal interface with:

* **Persistent Session State**: Multi-turn conversation state retained in memory and checkpointed using SQLite (`history.db`).
* **Rich Markdown Formatting**: Real-time streaming output with styled code blocks, system notices, and thinking blocks.
* **Interactive Tool Approvals**: Modern TUI widgets for Human-in-the-Loop (HITL) tool execution confirmation.
* **Autocomplete**: Tab-autocompletion for slash commands and local file/directory paths (`@-mentions`).
* **Dynamic Slash Commands**: In-session model switching, RAG loading, task management, and skill configuration.

To start the REPL:
```bash
ollama-agent
```

### Non-Interactive CLI Mode

Non-interactive mode allows single-shot prompt execution directly from your terminal or shell scripts. When given a prompt via `-p` or `--prompt`, `ollama-agent` runs the input, processes tool calls, streams the output directly to standard output, and exits with a status code.

To execute a non-interactive prompt:
```bash
ollama-agent -p "Summarize the commits from the last 24 hours."
```

You can pass configuration overrides directly on the CLI:
```bash
ollama-agent -m "gemma4:26b" -e "high" -t 60 --yolo -p "Perform system diagnostic and print report"
```

---

## Slash Commands Reference

Slash commands provide session management, model control, task execution, document retrieval, and skill management directly inside the REPL.

| Command | Usage | Description |
| :--- | :--- | :--- |
| `/help` | `/help` | Displays help message listing all available slash commands categorized by section. |
| `/model` | `/model [list \| set <model>]` | Lists available local Ollama models or switches the active model dynamically. |
| `/task` | `/task [list \| create <id> \| run <id> [-y] \| delete <id>]` | Manages saved prompt tasks (listing, modal creation, execution, deletion). |
| `/skill` | `/skill [list \| show <id> \| create <id> \| delete <id>]` | Manages agent skills (listing, detail inspection, modal creation, deletion). |
| `/rag` | `/rag [status \| list \| create <name> \| delete <name> \| load <name> \| unload \| add <path> [--dir]]` | Manages RAG vector stores (database creation, ingestion, loading/unloading, status). |
| `/yolo` | `/yolo [on\|off]` | Toggles YOLO mode or sets it explicitly (`on` or `off`). |
| `/new` | `/new` | Starts a new chat session (clears current conversation state and context). |
| `/clear` | `/clear` | Clears all rendered messages from the terminal screen viewport. |
| `/exit` / `/quit` | `/exit` or `/quit` | Gracefully closes connections and terminates the REPL application. |

---

## File & Directory Context Loading (`@-mentions`)

The `@-mention` syntax allows you to attach context directly from your local filesystem into user prompts.

### Usage Syntax

* **Single File**: `@src/main.py` or `@"path with spaces/file.txt"`
* **Directory Traversal**: `@.` or `@src/` (recursively scans and attaches all supported text and multimodal files).

When attached, text files are embedded as formatted XML blocks appended to the user prompt:
```xml
<context_file path="src/main.py">
... file contents ...
</context_file>
```

Multimodal binary files (e.g. images, audio, video, PDFs) are converted into base64 payload objects and passed as native media content blocks to the model.

### Interactive REPL Autocompletion

In REPL mode, typing `@` activates an interactive completion menu:
* Files and directories are matched dynamically against the workspace path.
* Pressing `Down` / `Up` navigates candidates.
* Pressing `Tab` or `Enter` inserts the highlighted completion candidate.
* Paths containing whitespace or special characters are automatically quoted.

### Safety Limits

Safety limits prevent prompt blowing and out-of-memory errors. The default limits are:

* `max_file_size`: **1 MB** (`1_048_576` bytes) - Individual files exceeding this size are rejected.
* `max_files`: **100** - Maximum number of files processed during a directory traversal mention.
* `max_total_size`: **10 MB** (`10_485_760` bytes) - Maximum combined size of all attached files in a single prompt.
* `max_completions`: **200** - Maximum autocomplete suggestions displayed in the REPL dropdown.

These settings are configured under the `mentions` key in `~/.ollama-agent/settings.yaml`.

### Decorator & Path Validation Safety

To prevent false positives when pasting code containing Python decorators or social media mentions (e.g. `@staticmethod`, `@classmethod`, `@dataclass`):
* Unquoted mentions that do not match existing filesystem paths, do **not** contain path separators (`/` or `\`), and lack common file extensions are treated as literal text rather than missing files.
* Mentions containing path separators, explicit extensions, or enclosed in quotes that do not exist on disk trigger an explicit `PromptProcessingError: File or directory not found`.

---

## Human-in-the-Loop (HITL) & YOLO Mode

To protect system integrity when running tool executions (such as local shell commands or file modifications), `ollama-agent` incorporates a Human-in-the-Loop authorization mechanism.

### Tool Approval Modal Actions

When an active tool invocation requires confirmation, execution pauses and presents an inline interactive prompt widget with four action buttons:

1. **Approve**: Authorizes the single pending tool call execution.
2. **Reject**: Aborts the pending tool call and sends an explicit rejection message back to the LLM, enabling it to retry or propose a safe alternative.
3. **Allow Session**: Authorizes the pending tool call and adds the tool name to `auto_approved_tools` for the remainder of the active REPL session.
4. **Cancel**: Cancels execution immediately without resuming the agent stream, returning cursor focus back to the REPL input bar.

### YOLO Mode

YOLO mode completely bypasses tool confirmation prompts, allowing autonomous execution of tool calls.

* **Activation**:
  * **CLI Flag**: `ollama-agent -y` or `ollama-agent --yolo`
  * **Slash Command**: `/yolo`, `/yolo on`, or `/yolo off`
* **Visual Indicator**: When YOLO mode is active:
  * Prompt indicator symbol switches from **blue** (`#89b4fa`) to **red** (`#f38ba8`).
  * Header bar updates status display to **`YOLO: On`** in red text.

---

## Reasoning Effort Controls

`ollama-agent` provides reasoning effort control for LLMs supporting thinking traces (e.g. DeepSeek R1, Qwen 3, GPT-OSS).

Pass the `--effort` CLI option or set `reasoning_effort` in settings:
```bash
ollama-agent -e high -p "Analyze this complex algorithm"
```

### Supported Effort Values

* `low`: Low reasoning effort (concise reasoning trace).
* `medium`: Default reasoning effort level.
* `high`: Maximum reasoning effort for deep problem analysis.
* `disabled`: Disables reasoning traces at model level where supported.
* `hide`: Enables reasoning traces but suppresses their display in the user interface.
* `enabled`: Enables reasoning traces using the default medium effort level.

### Model Compatibility Table

| Model Family | `--effort` Value | Ollama Native `think` Parameter | Behavior Notes |
| :--- | :--- | :--- | :--- |
| **GPT-OSS** | `low` / `medium` / `high` | `"low"` / `"medium"` / `"high"` | Sets explicit thinking trace effort level string. |
| **GPT-OSS** | `disabled` | `None` (omitted) | GPT-OSS cannot disable thinking natively in Ollama. Generates default thinking trace, but UI hides trace output and displays warning. |
| **GPT-OSS** | `hide` | `None` (omitted) | Thinking trace generated by model but hidden from UI. |
| **GPT-OSS** | `enabled` | `"medium"` | Explicitly enables thinking using standard medium effort. |
| **Other Thinking Models** (DeepSeek R1, Qwen 3, etc.) | `low` / `medium` / `high` / `enabled` | `True` | Enables thinking mode natively. |
| **Other Thinking Models** | `disabled` | `False` | Disables thinking mode natively in Ollama API payload. |
| **Other Thinking Models** | `hide` | `True` | Enables thinking mode natively, but hides thinking trace collapsible block in UI. |
| **Standard Models** (no thinking support) | *(any)* | `None` (omitted) | Reasoning parameter is ignored. |
