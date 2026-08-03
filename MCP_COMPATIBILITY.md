# MCP Dependency Compatibility Note

## Overview
This document records an issue encountered when updating dependencies for `ollama-agent` and explains why the `mcp` package version constraint is explicitly set in `pyproject.toml`.

---

## Issue Description

When upgrading project dependencies, launching `ollama-agent` produced the following traceback:

```text
Traceback (most recent call last):
  File "/home/arrase/.local/bin/ollama-agent", line 3, in <module>
    from ollama_agent.main import main
  ...
  File "/home/arrase/.local/share/pipx/venvs/ollama-agent/lib/python3.14/site-packages/langchain_mcp_adapters/callbacks.py", line 8, in <module>
    from mcp.shared.context import RequestContext as MCPRequestContext
ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'
```

---

## Root Cause

1. **Breaking API Changes in `mcp` 2.0.0**: The `mcp` package introduced major breaking changes in version `2.0.0`, moving/removing `RequestContext` from `mcp.shared.context`.
2. **`langchain-mcp-adapters` Dependency**: `langchain-mcp-adapters` (v0.3.1) relies internally on `mcp` 1.x imports (`from mcp.shared.context import RequestContext`).
3. **Unbounded PyPI Metadata**: `langchain-mcp-adapters` (v0.3.1) specifies `mcp>=1.24.0` in its package metadata without capping the upper bound (`<2.0.0`). Consequently, running `pip install --upgrade` or creating a fresh virtual environment defaults to pulling `mcp` 2.0.0+, breaking `langchain-mcp-adapters`.

---

## Current Solution

In [`pyproject.toml`](./pyproject.toml), the `mcp` package is explicitly constrained:

```toml
dependencies = [
    ...
    "langchain-mcp-adapters>=0.3.1",
    "mcp>=1.24.0,<2.0.0",
]
```

This prevents `pip` / `pipx` from upgrading `mcp` to version `2.0.0` or higher, preserving compatibility with `langchain-mcp-adapters`.

---

## Future Action Plan

When a new version of `langchain-mcp-adapters` is released that supports `mcp` 2.x:
1. Upgrade `langchain-mcp-adapters` in `pyproject.toml`.
2. Test whether `mcp` 2.x works properly.
3. Update or remove the `<2.0.0` constraint on `mcp` once compatibility is verified.
