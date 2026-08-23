---
name: task-creator
description: Guide and schema for creating reusable saved tasks. Use whenever creating, saving, or defining an automated task.
---

# Task Creator

A saved task is a pre-configured, self-contained prompt and configuration that can be executed autonomously on demand via `/task run <task_id>` or `ollama-agent task run <task_id>`. Tasks reside in `/tasks/<task_id>.yaml` (persisted in `~/.ollama-agent/tasks/<task_id>.yaml`).

## Task YAML Schema

Each task is saved as a single YAML file: `/tasks/<task_id>.yaml`.

```yaml
title: "Clear, descriptive title of the task"
prompt: "Comprehensive, fully self-contained prompt that describes the exact objective, context, requirements, constraints, and success criteria for the agent to execute."
model: "" # Optional specific model name, or empty string to use the active model
reasoning_effort: "medium" # Reasoning effort level: low, medium, high, default
```

### Fields Specification
- `title` (string, required): Short descriptive title for display in `/task list`.
- `prompt` (string, required): Self-contained prompt. Because tasks run autonomously, the prompt should provide all necessary context, step-by-step instructions, and expected deliverables.
- `model` (string, optional): Specific Ollama model name, or `""` to use the session's active model.
- `reasoning_effort` (string, optional): Reasoning effort (`low`, `medium`, `high`, `default`).

## Task ID Rules
- Must contain only letters, numbers, underscores, and hyphens (e.g. `code-review`, `generate-changelog`, `db_migration_check`).

## Interactive Creation Workflow for the Agent

When helping a user create a task:
1. **Clarify Objective**: Understand what repeatable task or workflow the user wants to automate.
2. **Formulate Prompt**: Draft a high-quality, comprehensive prompt ensuring it includes clear goals, constraints, and output expectations.
3. **Write File**: Use `write_file` to write the YAML file to `/tasks/<task_id>.yaml`.
4. **Confirm**: Display the task summary and instruct the user on how to run it with `/task run <task_id>`.
