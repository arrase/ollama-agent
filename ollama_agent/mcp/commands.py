"""MCP server status commands for CLI and REPL."""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.console import Console
from rich.table import Table

from .lifecycle import _infer_transport
from .types import DEFAULT_MCP_CONFIG_PATH, RunningMCPServer

logger = logging.getLogger(__name__)


def list_mcp_servers(
    console: Console,
    mcp_servers: list[RunningMCPServer],
    mcp_config_path: Any = None,
) -> None:
    """Display connection status and tool counts for configured MCP servers."""
    config_path = mcp_config_path or DEFAULT_MCP_CONFIG_PATH
    if not config_path.exists():
        console.print("[yellow]No MCP servers configured.[/yellow]")
        return

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]Failed to read MCP config: {exc}[/red]")
        return

    servers_cfg: dict[str, Any] = data.get("mcpServers", {})
    if not isinstance(servers_cfg, dict) or not servers_cfg:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        return

    running = {srv.name: srv for srv in mcp_servers}

    table = Table(title="MCP Servers", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Transport")
    table.add_column("Status")
    table.add_column("Tools", style="magenta", justify="right")

    for name, cfg in servers_cfg.items():
        transport = _infer_transport(cfg) if isinstance(cfg, dict) else "?"
        if name in running:
            status = "[green]Connected[/green]"
            tools = str(len(running[name].subagent.get("tools", [])))
        else:
            status = "[red]Error[/red]"
            tools = "-"
        table.add_row(name, transport, status, tools)

    console.print(table)


def show_mcp_server(
    console: Console,
    mcp_servers: list[RunningMCPServer],
    name: str,
) -> None:
    """Display the tools available for a specific MCP server."""
    running = {srv.name: srv for srv in mcp_servers}
    if name not in running:
        console.print(f"[red]MCP server not found or not connected:[/red] {name}")
        return

    tools = running[name].subagent.get("tools", [])
    if not tools:
        console.print(f"[yellow]No tools available for MCP server:[/yellow] {name}")
        return

    table = Table(
        title=f"MCP Server: {name}", show_header=True, header_style="bold magenta"
    )
    table.add_column("Tool", style="cyan")
    table.add_column("Description")

    max_desc = 80
    for tool in tools:
        desc = (tool.description or "").replace("\n", " ").strip()
        if len(desc) > max_desc:
            desc = desc[:max_desc] + "..."
        table.add_row(tool.name, desc)

    console.print(table)
