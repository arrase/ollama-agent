# Saved Tasks & Automation

**Tasks** represent saved, re-executable automation routines containing pre-configured prompt templates, dynamic input parameters, designated models, and reasoning effort levels. They allow users to package repetitive workflows (such as code reviews, release note generation, documentation updates, or security audits) into structured, parameterized commands.

---

## 1. Task Storage Format & Virtual Mount

Each task is stored as an individual YAML file under `~/.ollama-agent/tasks/<task_id>.yaml` and mounted into the agent's virtual filesystem at `/tasks/<task_id>.yaml`:

```yaml
title: "Single File Code Review"
prompt: |
  Review the source code in @{{ target_file }}.
  Focus on identifying potential runtime defects, security vulnerabilities, and code smell.
  {% if strict %}
  Enforce strict PEP 8 compliance and flag all styling deviations.
  {% else %}
  Focus primarily on correctness, performance bottlenecks, and architectural issues.
  {% endif %}
model: "qwen3.8:27b"
reasoning_effort: "high"
inputs:
  target_file:
    description: "Relative path of the source file to review"
    type: "string"
    required: true
  strict:
    description: "Enable strict style and linting review"
    type: "boolean"
    default: false
```

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `title` | `string` | **Yes** | Human-readable title displayed in lists and execution summaries. |
| `prompt` | `string` | **Yes** | Multi-line instruction template to evaluate. Supports full Jinja2 templating. |
| `model` | `string` | **Yes** | Specific Ollama model designated for this task (e.g. `"qwen3.8:27b"`, `"gemma4:26b"`). |
| `reasoning_effort` | `string` | **Yes** | Reasoning effort setting (`low`, `medium`, `high`, `xhigh`, `disabled`, `hide`, `enabled`). |
| `inputs` | `mapping` | No | Map of expected dynamic parameters, their data types, and default values. |

#### Input Schema (`inputs.<name>`)

| Property | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `description` | `string` | `""` | Human explanation of the expected parameter. |
| `type` | `string` | `"string"` | Expected type: `"string"`, `"boolean"`, or `"number"`. |
| `required` | `boolean` | `false` | When `true`, invocation fails if the variable is omitted. |
| `default` | `any` | `null` | Fallback value used when the parameter is not supplied. |

- **Task Identifier (`<task_id>`)**: File basename without `.yaml`. Must contain only alphanumeric characters, underscores, and hyphens (`[A-Za-z0-9_-]+`).
- **Unique Prefix Resolution**: In commands taking `<id>`, you can provide a unique prefix of the ID. If ambiguous or not found, an error is raised.

---

## 2. Jinja2 Templating & Dynamic Context

Task prompt templates are rendered using a strict Jinja2 environment (`trim_blocks=True`, `lstrip_blocks=True`):

* **Variable Interpolation**: `{{ target_file }}` dynamically injects variable values into the instruction body.
* **Dynamic File `@-mentions`**: `@{{ target_file }}` evaluates the variable first, producing an `@-mention` (such as `@src/parser.py`), which the prompt processor expands into complete file contents with binary safety checks.
* **Conditional Logic**: `{% if strict %}...{% else %}...{% endif %}` controls prompt instructions based on boolean flags.
* **Iteration**: `{% for item in items %}...{% endfor %}` formats collections or structured data lists.
* **Filters & Defaults**: `{{ branch | default('main') }}` provides fallbacks directly in the template.

---

## 3. Input Type Validation & Coercion

Variables passed to tasks (via CLI or REPL) are automatically validated and coerced to their declared schema types:

| Declared Type | Accepted Inputs | Output Value / Behavior |
| :--- | :--- | :--- |
| **`string`** | Any text value | Preserved as string representation (`str(val)`). |
| **`boolean`** | `true`, `1`, `yes` (case-insensitive) / `True` | `True` |
| | `false`, `0`, `no` (case-insensitive) / `False` | `False` |
| | Any other value | Fails immediately with `ValueError: Invalid boolean value`. |
| **`number`** | `"42"`, `42` / `"3.14"`, `3.14` | Coerced to `int` or `float`. |
| | Non-numeric string or boolean | Fails immediately with `ValueError: Invalid number value`. |

> [!IMPORTANT]
> **Fail-Fast Safety**: If a required input parameter (`required: true`) is omitted during invocation and lacks a default value, execution terminates immediately with a clear error: `Missing required input: <name>`.

---

## 4. Executing Tasks (CLI & REPL)

### Running via CLI

Tasks can be executed directly from your terminal, with variables passed as positional arguments or `--var` flags:

```bash
# Execute with positional key=value assignments
ollama-agent task run code-review target_file=src/app.py strict=true -y

# Execute with --var flags
ollama-agent task run code-review --var target_file=src/app.py --var strict=true

# Mixed positional and flag syntax
ollama-agent task run code-review target_file=src/app.py --var strict=false
```

### Running via REPL

In the interactive REPL, variables are passed as positional `key=value` pairs:

```text
/task run code-review target_file=src/app.py strict=true
/task run code-review target_file=src/app.py strict=true -y
```

### Execution Lifecycle
When a task is executed:
1. `TaskManager` loads and validates the YAML definition.
2. Supplied input variables are validated, type-coerced, and defaults are applied.
3. The Jinja2 template is rendered into the final prompt text.
4. The runtime temporarily binds the task's configured model and reasoning effort.
5. The prompt streams live (in REPL or CLI) with full tool access and dynamic context expansion.
6. Upon completion, prior model parameters, session states, and YOLO modes are restored cleanly.

---

## 5. Task Management Commands

| Action | CLI Command | REPL Slash Command | Description |
| :--- | :--- | :--- | :--- |
| **List Tasks** | `ollama-agent task list` | `/task list` (or `/task`) | List all saved tasks and their configured models. |
| **Create Task** | `ollama-agent task create <id> -t <title> -p <prompt> -m <model> [-e <effort>] [--force]` | `/task create [<id>]` | Save a new task template via CLI or conversational interview. |
| **Run Task** | `ollama-agent task run <id> [key=value ...] [-y]` | `/task run <id> [key=value ...] [-y]` | Execute a saved task with dynamic variable bindings. |
| **Delete Task** | `ollama-agent task delete <id>` | `/task delete <id>` | Permanently remove a saved task YAML file. |

> [!TIP]
> **Conversational Creation (`/task create`)**: Running `/task create` in the REPL engages the agent using the built-in `task-creator` skill. The agent interviews you about what the task should do, identifies variables, and writes the validated YAML definition automatically.
