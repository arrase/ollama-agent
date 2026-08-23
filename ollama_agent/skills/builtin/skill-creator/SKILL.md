---
name: skill-creator
description: Comprehensive guide and best practices for creating new agent skills following the Agent Skills standard. Use whenever creating, scaffolding, or designing a new skill.
---

# Skill Creator

A skill is a modular, self-contained package of capabilities and instructions that extends the agent's abilities. Skills follow the standard Agent Skills specification and reside in `/skills/<skill_id>/` (persisted in `~/.ollama-agent/skills/<skill_id>/`).

## Directory Structure

A skill directory must contain a `SKILL.md` file and may include optional subdirectories:

```text
/skills/<skill-id>/
├── SKILL.md                 # Required: Frontmatter YAML + instructions
├── scripts/                 # Optional: Executable Python or Bash helper scripts
├── references/              # Optional: Deep reference documentation, schemas, API specs
└── examples/                # Optional: Input/output samples, templates, test cases
```

## Critical Decision: When to Create `scripts/` vs Pure `SKILL.md`

When designing a skill, always evaluate whether helper scripts are needed:

1. **Create Helper Scripts (`scripts/`) when:**
   - The task involves deterministic algorithms, parsing structured formats (JSON, AST, XML, CSV), regex transformations, or data validation.
   - The task performs repetitive Git workflows, CLI tool wrapping, API fetching, or complex file operations.
   - The operation is error-prone or wasteful to do purely through LLM token generation.
   - *Language*: Write modular, self-contained Python scripts (or Bash if simple shell piping is sufficient). Ensure scripts handle errors gracefully and output clear diagnostic messages.

2. **Use Pure Markdown (`SKILL.md`) when:**
   - The skill focuses on reasoning guidelines, architectural reviews, code style heuristics, tone/formatting advice, or general workflow orchestration without deterministic code execution.

## SKILL.md Specification

### 1. Frontmatter (YAML)
Every `SKILL.md` must start with YAML frontmatter:
```yaml
---
name: skill-id-or-name
description: A clear, concise 1-3 sentence summary of what this skill does and the exact trigger conditions for when the agent should activate it.
---
```
> **IMPORTANT (Level 1 Discovery)**: The `description` is loaded into the agent's discovery prompt before the full skill content is read. It MUST explicitly state **what** the skill does and **when** the agent should load and follow it.

### 2. Document Sections
Organize the body of `SKILL.md` cleanly:
- `## Overview`: High-level summary of the skill's purpose.
- `## When to Use`: Specific triggers, keywords, file patterns, or scenarios.
- `## Prerequisites & Tools`: Required environment variables, packages, or tools.
- `## Step-by-Step Workflow`: Clear, numbered, deterministic steps for the agent to follow.
- `## Helper Scripts`: If `scripts/` exist, document the exact execution commands (e.g. `python /skills/<skill-id>/scripts/<script_name>.py <args>`), expected parameters, and output format.
- `## Examples & Edge Cases`: Concrete examples of typical inputs, expected outputs, and common pitfalls.

## Interactive Creation Workflow for the Agent

When helping a user create a skill:
1. **Interview & Clarify**: If the user's request lacks details, ask concise clarifying questions in the language of the conversation about the goal, target tools, language preferences, and whether helper scripts are needed.
2. **Determine Structure**: Decide on the skill ID (lowercase, alphanumeric, hyphens/underscores only), title, description, and whether `scripts/` or `references/` are required.
3. **Draft & Create Files**: Use `write_file` to write:
   - `/skills/<skill_id>/SKILL.md`
   - Any necessary `/skills/<skill_id>/scripts/<script_name>.py` (or other helper files)
4. **Confirm**: Report the created skill ID and summarize the files created and how the skill will assist future tasks.
