---
name: mcp-configurator
description: Comprehensive guide and schema for configuring Model Context Protocol (MCP) servers for the main agent and subagents. Use whenever adding, editing, verifying, or troubleshooting MCP servers and tools.
---

# MCP Configurator

Model Context Protocol (MCP) connects `ollama-agent` to external tools, databases, and developer services. In `ollama-agent`, MCP servers can be attached globally to the main orchestrator agent or scoped locally to specific subagents.

## MCP Configuration Scopes

| Scope | Configuration File | Virtual Agent Path | Description |
| :--- | :--- | :--- | :--- |
| **Main Agent (Global)** | `~/.ollama-agent/mcp.json` | `/agent/mcp.json` | Global MCP tools available directly to the main agent orchestrator. |
| **Subagent (Scoped)** | `~/.ollama-agent/settings.yaml` | `/agent/settings.yaml` | Dedicated MCP tools available only within an isolated subagent's execution graph. |

---

## 1. Main Agent MCP Configuration (`/agent/mcp.json`)

The global configuration file resides at `/agent/mcp.json` (persisted at `~/.ollama-agent/mcp.json`). It uses JSON format with a top-level `"mcpServers"` object.

### Schema & Supported Transports

#### A. Standard I/O Subprocess (`stdio`)
Runs an executable as a local subprocess communicating via stdio:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
      "cwd": "/path/to/directory",
      "env": {
        "DEBUG": "true"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"]
    }
  }
}
```

- `command` (*string*, required): Executable command (e.g. `npx`, `uvx`, `python`, `node`, `docker`, or binary path).
- `args` (*array of strings*, optional): Command-line arguments.
- `cwd` (*string*, optional): Working directory for the spawned process.
- `env` (*object*, optional): Environment variables. Values support `${VAR}` and `%VAR%` expansion from the host environment.
- `transport` (*string*, optional): Defaults to `"stdio"` when `command` is present.

#### B. Remote Network Transports (`http`, `sse`, `websocket`, `streamable_http`)
Connects to a remote or local HTTP, Server-Sent Events (SSE), or WebSocket endpoint:

```json
{
  "mcpServers": {
    "remote-service": {
      "type": "http",
      "url": "https://mcp.example.com/api",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
      },
      "timeout": 30,
      "sse_read_timeout": 300
    },
    "sse-service": {
      "transport": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

- `url` / `httpUrl` (*string*, required): Target server URL endpoint.
- `transport` / `type` (*string*, optional): `"http"`, `"sse"`, `"websocket"`, `"streamable_http"`, or `"streamable-http"` (defaults to `"http"`).
- `headers` (*object*, optional): HTTP request headers (e.g. authorization tokens).
- `timeout` (*number*, optional): Request timeout in seconds.
- `sse_read_timeout` (*number*, optional): SSE stream read timeout in seconds.
- `session_kwargs` (*object*, optional): Additional low-level client session parameters.

---

## 2. Subagent MCP Configuration (`/agent/settings.yaml`)

Subagents run in isolated graphs and can possess their own dedicated MCP toolsets defined in `/agent/settings.yaml` (persisted at `~/.ollama-agent/settings.yaml`) under the `subagents` list:

```yaml
subagents:
  - name: "database-expert"
    description: "Specialist for executing database queries, inspecting schema, and running migrations."
    system_prompt: "You are a database administrator. Execute SQL queries and analyze schema changes."
    model: "qwen2.5-coder:32b" # Optional: inherits session model if omitted
    context_window: 32768       # Optional: inherits session context if omitted
    mcp_servers:
      - name: "postgres"
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
        env:
          PGPASSWORD: "${DB_PASSWORD}"
```

### Subagent MCP Server Fields:
- `name` (*string*, required): Identifier for the server.
- `command` (*string*, required): Subprocess executable command.
- `args` (*list of strings*, optional): Arguments passed to the executable.
- `env` (*dict*, optional): Environment variables with `${VAR}` and `%VAR%` expansion.

---

## 3. Environment Variable Expansion & Secrets

- Environment variables in `"env"` blocks and `"headers"` can reference host system variables using `${VAR_NAME}` or `%VAR_NAME%` syntax.
- `ollama-agent` resolves these dynamically from the system environment (`os.environ`).
- **Fail-Fast**: If a referenced variable is missing from the host environment, an error is raised immediately.
- **Best Practice**: Always use `${VAR_NAME}` references for sensitive tokens and API keys rather than hardcoding credentials into configuration files.

---

## 4. Inspecting & Verifying MCP Servers

Users can test connections and verify discovered tools:

- **In the Interactive REPL**:
  ```text
  /mcp
  /mcp list
  /mcp reload
  ```
- **In the Terminal CLI**:
  ```bash
  ollama-agent mcp list
  ```

Output displays a formatted table with:
- Status (🟢 `● Active` with tool names and count, or 🔴 `● Failed` with connection error)
- Server name and scope
- Transport type (`stdio`, `http`, `sse`)
- Target command / URL

---

## 5. Interactive Agent Workflow

When a user requests adding, modifying, or troubleshooting an MCP server:

1. **Clarify Scope & Details**:
   - Determine if the MCP server is for the main agent (`/agent/mcp.json`) or a subagent (`/agent/settings.yaml`).
   - Identify the package or service (e.g. npm package `@modelcontextprotocol/...`, python package via `uvx`, or remote URL).
   - Check if any required environment variables or credentials are required.
2. **Read Current Configuration**:
   - Read `/agent/mcp.json` (or `/agent/settings.yaml`) to preserve existing settings.
   - If `/agent/mcp.json` does not exist, start with `{"mcpServers": {}}`.
3. **Format & Write Configuration**:
   - Use `write_file` or `edit_file` to update `/agent/mcp.json` or `/agent/settings.yaml`.
   - Ensure the JSON/YAML is valid and properly structured.
4. **Confirm & Prompt for `/mcp reload`**:
   - Confirm what server was added, updated, or removed.
   - **Mandatory Step**: Explicitly instruct the user to run `/mcp reload` in the chat to immediately reload the MCP servers and rebuild the tool graph in the current session.
   - Mention that `/mcp reload` will verify connections and display the updated list of available tools.

