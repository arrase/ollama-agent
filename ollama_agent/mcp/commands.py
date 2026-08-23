"""Shared MCP management commands used by CLI and REPL."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..i18n import _
from ..settings import Settings
from .loader import _build_mcp_connection, get_mcp_config_path


@dataclass(slots=True)
class MCPServerStatus:
    """Status snapshot of an MCP server connection."""

    name: str
    transport: str
    target: str
    status: str  # "active" | "failed"
    tools: list[str]
    error: str = ""


async def check_mcp_server(
    name: str,
    cfg: dict[str, Any],
    *,
    timeout: float = 10.0,
) -> MCPServerStatus:
    """Connect to an MCP server and probe available tools."""
    transport = cfg.get("transport") or cfg.get("type") or ("stdio" if cfg.get("command") else "http")
    target = (
        cfg.get("url")
        or cfg.get("httpUrl")
        or f"{cfg.get('command', '')} {' '.join(cfg.get('args', []))}".strip()
    )

    conn = _build_mcp_connection(cfg)
    if not conn:
        return MCPServerStatus(
            name=name,
            transport=transport,
            target=target,
            status="failed",
            tools=[],
            error=_("Invalid configuration or missing environment variables"),
        )

    try:
        client = MultiServerMCPClient({name: conn})  # type: ignore[dict-item,arg-type]
        tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
        tool_names = [t.name for t in tools]
        return MCPServerStatus(
            name=name,
            transport=conn.get("transport", transport),
            target=target,
            status="active",
            tools=tool_names,
            error="",
        )
    except asyncio.TimeoutError:
        return MCPServerStatus(
            name=name,
            transport=conn.get("transport", transport),
            target=target,
            status="failed",
            tools=[],
            error=_("Connection timed out ({timeout}s)", timeout=int(timeout)),
        )
    except Exception as exc:
        return MCPServerStatus(
            name=name,
            transport=conn.get("transport", transport),
            target=target,
            status="failed",
            tools=[],
            error=str(exc),
        )


async def list_mcp_servers(
    console: Console,
    settings: Settings | None = None,
) -> None:
    """List all configured MCP servers and display their connection status."""
    config_path = get_mcp_config_path()
    servers_cfg: dict[str, dict[str, Any]] = {}

    if config_path.exists():
        try:
            raw_json = await asyncio.to_thread(config_path.read_text, encoding="utf-8")
            data = json.loads(raw_json)
            raw_servers = data.get("mcpServers") or data.get("servers") or {}
            if isinstance(raw_servers, dict):
                servers_cfg.update(
                    {k: v for k, v in raw_servers.items() if isinstance(v, dict)}
                )
        except (json.JSONDecodeError, OSError) as exc:
            console.print(
                f"[red]{_('Error reading MCP configuration {path}: {exc}', path=config_path, exc=exc)}[/red]"
            )
            return

    subagent_servers: dict[str, dict[str, Any]] = {}
    if settings:
        for sa in settings.subagents:
            for srv in sa.mcp_servers:
                subagent_servers[f"{srv.name} ({sa.name})"] = {
                    "command": srv.command,
                    "args": srv.args,
                    "env": srv.env,
                    "transport": "stdio",
                }

    all_servers = {**servers_cfg, **subagent_servers}
    if not all_servers:
        console.print(
            f"[yellow]{_('No MCP servers configured.')}[/yellow]\n"
            f"[dim]{_('Configure servers in {path}', path=config_path)}[/dim]"
        )
        return

    tasks = [
        check_mcp_server(name, cfg)
        for name, cfg in all_servers.items()
    ]
    statuses: list[MCPServerStatus] = await asyncio.gather(*tasks)

    table = Table(
        title=_("Model Context Protocol (MCP) Servers"),
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column(_("Status"), style="bold", justify="center")
    table.add_column(_("Server"), style="bold cyan")
    table.add_column(_("Type"), style="blue")
    table.add_column(_("Target / Command"), style="dim")
    table.add_column(_("Tools / Details"))

    for st in statuses:
        if st.status == "active":
            status_text = "[bold green]● Active[/bold green]"
            if st.tools:
                tools_str = (
                    f"[green]{len(st.tools)} {_('tools')}:[/green] "
                    f"[dim]{', '.join(st.tools)}[/dim]"
                )
            else:
                tools_str = f"[green]{_('0 tools available')}[/green]"
        else:
            status_text = "[bold red]● Failed[/bold red]"
            tools_str = f"[red]{st.error}[/red]"

        table.add_row(
            status_text,
            st.name,
            st.transport,
            st.target,
            tools_str,
        )

    console.print(table)
