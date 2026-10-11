# Model Context Protocol (MCP) Integration

Give your local AI agent real-world superpowers. The **Model Context Protocol (MCP)** is an open industry standard that connects Ollama Agent directly to external developer tools, live internet search, databases, filesystems, and remote APIs—without writing Python glue code or modifying agent internals.

Whether you need your agent to search the live web via Brave Search, query a production PostgreSQL database, inspect a Git repository, or interact with private enterprise webhooks, MCP makes it plug-and-play.

```mermaid
flowchart LR
    User(["Your Prompt"]) --> Agent["Ollama Agent"]
    
    subgraph LocalTools ["Local Subprocess Transports (stdio)"]
        Agent -->|"npx / uvx / docker"| Search["Brave Search<br/>(Live Web Results)"]
        Agent -->|"npx"| FS["Filesystem Server<br/>(Extra Folders)"]
        Agent -->|"uvx"| Git["Git Server<br/>(Diffs, Commits, History)"]
        Agent -->|"uvx"| DB["Database Inspector<br/>(Postgres / SQLite)"]
    end
    
    subgraph RemoteTools ["Remote Network Transports (HTTP / SSE / WS)"]
        Agent -->|"Bearer Auth / SSE"| RemoteAPI["Enterprise Services<br/>& Cloud Endpoints"]
    end
```

---

## 30-Second Quickstart

Get your first MCP server running in three simple steps.

=== "Step 1: Create Configuration"
    Create or open your global MCP configuration file at `~/.ollama-agent/mcp.json`:

    ```bash
    mkdir -p ~/.ollama-agent
    nano ~/.ollama-agent/mcp.json
    ```

=== "Step 2: Add a Server"
    Paste the following configuration to enable live internet search via Brave Search (or filesystem browsing for a custom folder):

    ```json
    {
      "mcpServers": {
        "brave-search": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-brave-search"],
          "env": {
            "BRAVE_API_KEY": "${BRAVE_API_KEY}"
          }
        }
      }
    }
    ```

    Export your API key in your shell:
    ```bash
    export BRAVE_API_KEY="your_api_key_here"
    ```

=== "Step 3: Launch and Use"
    Launch the interactive REPL:

    ```bash
    ollama-agent
    ```

    Verify the tools are loaded by typing `/mcp`:
    ```text
    /mcp
    ```

    Now ask your agent to use its new tool:
    > *"Search the web for the latest Python 3.13 release features and summarize the top three highlights."*

---

## Configuration File Format (`mcp.json`)

All global MCP servers are defined in `~/.ollama-agent/mcp.json`. The file contains a top-level `"mcpServers"` object where each key is a unique server identifier:

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    },
    "git-tools": {
      "command": "uvx",
      "args": ["mcp-server-git"]
    },
    "enterprise-api": {
      "type": "http",
      "url": "https://mcp.internal.company.com/v1",
      "headers": {
        "Authorization": "Bearer ${COMPANY_API_TOKEN}"
      },
      "timeout": 30
    }
  }
}
```

### Environment Variable Expansion

Never hardcode sensitive API keys or passwords directly into configuration files. Ollama Agent automatically resolves environment variables at runtime inside `"env"` maps and `"headers"` values using either `${VAR_NAME}` or `%VAR_NAME%` syntax:

```json
"env": {
  "BRAVE_API_KEY": "${BRAVE_API_KEY}",
  "DATABASE_PASSWORD": "${DB_PASSWORD}"
}
```

* **Dynamic Host Resolution**: Variables are evaluated directly against your current shell environment (`export KEY=value`).
* **Fail-Fast Security**: If a declared variable is unset or empty, Ollama Agent halts immediately with a clear `MCPConfigError` indicating exactly which variable is missing, preventing broken tool runs downstream.

---

## Supported Transports

Ollama Agent supports two connection architectures: **Local Subprocesses (`stdio`)** and **Remote Network Services (`http`, `sse`, `websocket`, `streamable_http`)**.

### 1. Local Subprocess Transport (`stdio`)

Spawns an executable command (like Node.js packages via `npx`, Python packages via `uvx`, Docker containers, or local binaries) and communicates through standard input and output streams.

```json
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"],
      "cwd": "/home/user/my-project",
      "env": {
        "GIT_PYTHON_REFRESH": "quiet"
      }
    }
  }
}
```

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `command` | `string` | **Yes** | Executable name or absolute binary path (e.g., `npx`, `uvx`, `python`, `docker`). |
| `args` | `array of strings` | No | Command-line arguments passed to the process (e.g. `["-y", "@package/name"]`). |
| `cwd` | `string` | No | Working directory for the spawned process. |
| `env` | `object` | No | Key-value pairs of environment variables. Supports `${VAR}` expansion. |
| `transport` | `string` | No | Defaults to `"stdio"` when `command` is present. |

!!! tip "Zero-Install Subprocesses"
    Using `npx -y` (Node.js) or `uvx` (Python) downloads and executes MCP packages on the fly in temporary virtual environments without cluttering your global package manager.

### 2. Remote Network Transports (`http`, `sse`, `websocket`, `streamable_http`)

Connects to a remote or cloud-hosted MCP endpoint using HTTP POST, Server-Sent Events (SSE), or WebSockets.

```json
{
  "mcpServers": {
    "cloud-agent": {
      "type": "http",
      "url": "https://mcp.internal.company.com/api",
      "headers": {
        "Authorization": "Bearer ${CORP_MCP_TOKEN}"
      },
      "timeout": 30,
      "sse_read_timeout": 300
    }
  }
}
```

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `url` | `string` | **Yes** | Target endpoint URL of the remote MCP server. |
| `type` / `transport` | `string` | No | Protocol type: `"http"`, `"sse"`, `"websocket"`, `"streamable_http"`, or `"streamable-http"`. Default is `"http"`. |
| `headers` | `object` | No | Custom HTTP request headers (e.g., API keys, Bearer tokens). |
| `timeout` | `number` | No | Initial connection and request timeout in seconds (default: 10s). |
| `sse_read_timeout` | `number` | No | Maximum read timeout in seconds for streaming SSE responses. |

---

## 5 Popular Copy-Paste Server Recipes

Copy these tested recipes directly into your `~/.ollama-agent/mcp.json` file.

### Recipe 1: Brave Search (Live Internet Access)

Give your agent the ability to search the web for up-to-date documentation, breaking news, and current technical references.

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    }
  }
}
```

* **Tools Provided**: `brave_web_search`, `brave_local_search`.
* **Prerequisites**: Obtain a free API key from [Brave Search API](https://brave.com/search/api/) and run `export BRAVE_API_KEY="your-key"`.

### Recipe 2: Additional Filesystem Directories

While Ollama Agent automatically manages your current project directory, you can grant it secure access to additional document folders or reference libraries.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/user/documents",
        "/home/user/downloads"
      ]
    }
  }
}
```

* **Tools Provided**: `read_file`, `write_file`, `list_directory`, `move_file`, `search_files`.
* **Safety Note**: The server will strictly disallow directory traversal outside of the allowed paths listed in `args`.

### Recipe 3: Git Repository Management

Inspect commits, generate accurate diffs, check branches, and analyze repository status without shell escaping errors.

```json
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"]
    }
  }
}
```

* **Tools Provided**: `git_status`, `git_diff`, `git_log`, `git_commit`, `git_branches`.
* **Prerequisites**: Requires `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

### Recipe 4: Database Inspector (PostgreSQL & SQLite)

Allow your agent to analyze schemas, debug database queries, and inspect table contents safely.

=== "PostgreSQL"
    ```json
    {
      "mcpServers": {
        "postgres": {
          "command": "npx",
          "args": [
            "-y",
            "@modelcontextprotocol/server-postgres",
            "postgresql://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}"
          ]
        }
      }
    }
    ```

=== "SQLite"
    ```json
    {
      "mcpServers": {
        "sqlite": {
          "command": "uvx",
          "args": [
            "mcp-server-sqlite",
            "--db-path",
            "/home/user/data/app.db"
          ]
        }
      }
    }
    ```

* **Tools Provided**: `query`, `read_query`, `list_tables`, `describe_table`.

### Recipe 5: Remote API with Bearer Token

Connect to a remote enterprise MCP gateway or custom cloud microservice over HTTP or Server-Sent Events (SSE).

```json
{
  "mcpServers": {
    "corp-gateway": {
      "type": "sse",
      "url": "https://mcp.internal.company.com/sse",
      "headers": {
        "Authorization": "Bearer ${CORP_API_TOKEN}",
        "X-Environment": "production"
      },
      "timeout": 30,
      "sse_read_timeout": 300
    }
  }
}
```

* **Tools Provided**: Dynamic endpoints published by your remote gateway.
* **Prerequisites**: Export your token (`export CORP_API_TOKEN="..."`).

---

## Using & Inspecting MCP Tools

### Inspecting Server Status

You can check server connectivity, transport types, and discovered tools at any time.

=== "Interactive REPL"
    Type `/mcp` or `/mcp list` inside the REPL:

    ```text
    /mcp
    ```

    This prints a clean status table:
    ```text
    ╭────────────────────── Model Context Protocol (MCP) Servers ──────────────────────╮
    │ Status   │ Server        │ Type  │ Target / Command            │ Tools / Details   │
    ├──────────┼───────────────┼───────┼─────────────────────────────┼───────────────────┤
    │ ● Active │ brave-search  │ stdio │ npx -y @model...            │ 2 tools: search.. │
    │ ● Active │ git           │ stdio │ uvx mcp-server-git          │ 5 tools: git_...  │
    │ ● Failed │ remote-api    │ http  │ https://mcp.corp.internal   │ Connection timed  │
    ╰──────────┴───────────────┴───────┴─────────────────────────────┴───────────────────╯
    ```

=== "CLI Command"
    From your terminal, run:

    ```bash
    ollama-agent mcp list
    ```

Status badges clearly indicate operational readiness:
- 🟢 **`● Active`**: Server connected and ready, showing the total count and names of available tools.
- 🔴 **`● Failed`**: Connection failed, showing the specific error or timeout reason.

### Mid-Session Live Reloading (`/mcp reload`)

If you edit `~/.ollama-agent/mcp.json` or start a new local server while chatting, **you do not need to restart the application**. Run:

```text
/mcp reload
```

Ollama Agent cleanly reconnects to your servers, discovers new tools, and updates the active session **while completely preserving your existing chat history**.

### Natural Language Configuration (`mcp-configurator`)

Ollama Agent includes a built-in assistant skill named `mcp-configurator`. Rather than manually editing JSON syntax, you can instruct the agent in plain English:

> *"Add the GitHub MCP server to my config with GITHUB_TOKEN."*  
> *"Set up an SQLite database inspector pointing to /tmp/analytics.db."*

The agent checks required flags, guides you through required environment variables, and safely updates `~/.ollama-agent/mcp.json`. For more details, see the [Skills Guide](skills.md).

### Scoping MCP Tools to Subagents

You don't have to load every tool into your primary agent. You can assign dedicated MCP servers to isolated **Subagents** in `~/.ollama-agent/settings.yaml`:

```yaml
subagents:
  - name: "database-auditor"
    description: "Specialist in analyzing SQL schemas and optimizing queries."
    model: "qwen2.5-coder:32b"
    mcp_servers:
      - name: "postgres"
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/prod"]
```

This keeps your main agent's context window clean and focused while empowering specialized workers with deep toolsets. Learn more in the [Subagents Guide](subagents.md).

### Diagnostic Logging (`~/.ollama-agent/mcp.log`)

Subprocess tools often print diagnostics, npm warnings, or update notices to standard error (`stderr`). To keep your interactive terminal user interface completely clean and avoid display corruption, Ollama Agent automatically routes all `stderr` output from `stdio` servers to:

```text
~/.ollama-agent/mcp.log
```

When diagnosing a misbehaving server, tail this log file in a separate terminal:

```bash
tail -f ~/.ollama-agent/mcp.log
```

### Tool Execution Timeout (`builtin_tool_timeout`)

By default, tool executions are permitted to run for up to 30 seconds before timing out. If you run long database migrations, heavy git operations, or slow remote webhooks, you can adjust this limit:

=== "In settings.yaml"
    Open `~/.ollama-agent/settings.yaml`:
    ```yaml
    runtime:
      builtin_tool_timeout: 60  # Timeout in seconds
    ```

=== "Via CLI Option"
    Specify the timeout when launching:
    ```bash
    ollama-agent --builtin-tool-timeout 60
    ```

---

## Troubleshooting Common Errors

### 1. Missing Executable (`npx` / `uvx` / `docker`)

**Symptom**: Server status displays `● Failed` with `FileNotFoundError: [Errno 2] No such file or directory: 'uvx'`.

**Cause**: The tool runner (`npx`, `uvx`, or `docker`) is not installed or not present in your system `PATH`.

**Solution**:
* For Node.js packages: install Node.js and npm (`sudo apt install nodejs npm` or via `nvm`).
* For Python packages: install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
* Alternatively, provide the absolute path in `mcp.json`:
  ```json
  "command": "/home/user/.cargo/bin/uvx"
  ```

### 2. Missing Environment Variables

**Symptom**: Agent start fails with `MCPConfigError: MCP server 'brave-search': missing environment variable 'BRAVE_API_KEY'`.

**Cause**: A variable referenced as `${VAR_NAME}` in `mcp.json` is not exported in your current shell session.

**Solution**:
1. Export the variable before launching:
   ```bash
   export BRAVE_API_KEY="your-key-here"
   ```
2. Or add it to your shell profile (`~/.bashrc` or `~/.zshrc`).

### 3. Remote Connection & Timeout Errors

**Symptom**: Server status displays `● Failed` with `Connection timed out (10s)` or HTTP 401/403.

**Cause**: The remote server is unreachable, the endpoint URL is incorrect, or authentication headers are rejected.

**Solution**:
* Test connectivity using `curl`:
  ```bash
  curl -i -H "Authorization: Bearer $CORP_API_TOKEN" https://mcp.internal.company.com/api
  ```
* Increase the timeout in `mcp.json`:
  ```json
  "timeout": 30,
  "sse_read_timeout": 300
  ```

### 4. Malformed JSON Configuration

**Symptom**: Startup warning `Failed to load MCP config ... expected a JSON object`.

**Cause**: Trailing commas, unquoted keys, or syntax errors in `~/.ollama-agent/mcp.json`.

**Solution**: Validate your configuration file with Python's built-in JSON linter:
```bash
python3 -m json.tool ~/.ollama-agent/mcp.json > /dev/null
```

---

## Pro Tips & Best Practices

1. **Keep Secrets Out of Config Files**: Always use `${SECRET_NAME}` references rather than hardcoding credentials into `mcp.json`.
2. **Scope Heavy Tools to Subagents**: Instead of loading 30+ tools into your global agent, isolate database or specialized code-analysis tools into dedicated subagents (see [Subagents Guide](subagents.md)).
3. **Use `/mcp reload` While Testing**: When modifying arguments or adding new tools, use `/mcp reload` inside the REPL to instantly test changes without losing context.
4. **Tail `mcp.log` for Troubleshooting**: If an npm or python server fails to start, check `~/.ollama-agent/mcp.log` for the exact subprocess crash stack trace.

---

## Related Documentation

* [CLI & REPL User Guide](cli_repl.md) — Learn how to execute commands and navigate the interactive TUI.
* [Subagents Guide](subagents.md) — Configure specialized agents with dedicated MCP toolsets.
* [Skills System](skills.md) — Discover built-in skills like `mcp-configurator` to manage tools naturally.
* [Configuration Reference](configuration.md) — Full reference for `settings.yaml` options.
