---
name: task-creator
description: Guide and schema for creating reusable saved tasks. Use whenever creating, saving, or defining an automated task.
---

# Task Creator

A saved task is a pre-configured, self-contained prompt template and configuration that can be executed autonomously on demand via `/task run <task_id> [key=value ...]` or `ollama-agent task run <task_id> [key=value ...]`. Tasks reside in `/tasks/<task_id>.yaml` (persisted in `~/.ollama-agent/tasks/<task_id>.yaml`).

## Task YAML Schema

Each task is saved as a single YAML file: `/tasks/<task_id>.yaml`.

```yaml
title: "Clear, descriptive title of the task"
prompt: "Jinja2 prompt template with {{ variable }}, {% if %}, {% for %}, and @mentions..."
model: "" # Optional specific model name, or empty string to use the active model
reasoning_effort: "medium" # Reasoning effort level: low, medium, high, xhigh, disabled, hide, enabled
inputs:
  target_file:
    description: "Target file to analyze"
    type: "string"
    required: true
  strict:
    description: "Enable strict mode"
    type: "boolean"
    default: false
```

### Fields Specification

- `title` (string, required): Short descriptive title for display in `/task list`.
- `prompt` (string, required): Jinja2-compatible prompt template. Because tasks run autonomously, the prompt should provide all necessary context, step-by-step instructions, and expected deliverables.
- `model` (string, optional): Specific Ollama model name, or `""` to use the session's active model.
- `reasoning_effort` (string, optional): Reasoning effort (`low`, `medium`, `high`, `xhigh`, `disabled`, `hide`, `enabled`).
- `inputs` (mapping, optional): Dictionary of input parameter definitions:
  - `description` (string, optional): Human-readable explanation of the input parameter.
  - `type` (string, optional): Data type for value coercion (`string`, `boolean`, `number`). Default is `string`.
  - `required` (boolean, optional): Whether the variable must be provided when running the task. Default is `false`.
  - `default` (any, optional): Default fallback value when the variable is omitted at runtime.

### Jinja2 Templating & Dynamic Context

Prompts support full Jinja2 templating combined with `@-mentions`:
- **Variable Substitution**: `{{ target_file }}` interpolates the provided variable value.
- **Dynamic `@-mentions`**: `@{{ target_file }}` embeds the file content dynamically into prompt context once rendered.
- **Conditionals**: `{% if strict %}Enforce strict PEP 8 and fail on any warning.{% endif %}` toggles instructions dynamically.
- **Loops & Filters**: `{% for item in items %}...{% endfor %}` and `{{ branch | default('main') }}`.

## Task ID Rules
- Must contain only letters, numbers, underscores, and hyphens (e.g. `code-review`, `generate-changelog`, `db_migration_check`).

## Interactive Creation Workflow for the Agent

When helping a user create a task:
1. **Clarify Objective & Inputs**: Understand what repeatable task or workflow the user wants to automate and identify any dynamic parameters (`inputs`).
2. **Formulate Prompt Template**: Draft a high-quality Jinja2 prompt template ensuring clear instructions, appropriate `@-mentions` (e.g. `@{{ file }}`), conditionals, and deliverables.
3. **Write File**: Use `write_file` to write the YAML file to `/tasks/<task_id>.yaml`.
4. **Confirm**: Display the task summary and instruct the user on how to run it with `/task run <task_id> [key=value ...]`.

## Examples

### 1. Parameterized Code Review Task
```yaml
title: "Single File Code Review"
prompt: |
  Review the code in @{{ target_file }}.
  Focus on bugs, performance, security issues, and style.
  {% if strict %}
  Apply strict zero-tolerance linting and flag any minor deviations.
  {% else %}
  Focus primarily on critical defects and architectural concerns.
  {% endif %}
model: ""
reasoning_effort: "high"
inputs:
  target_file:
    description: "Relative path of the source file to review"
    type: "string"
    required: true
  strict:
    description: "Whether to enable strict review mode"
    type: "boolean"
    default: false
```

### 2. Static Task without Inputs
```yaml
title: "Repository Tree Analyzer"
prompt: "List the repository structure and describe the purpose of each top-level directory."
model: "gemma4:26b"
reasoning_effort: "medium"
```
