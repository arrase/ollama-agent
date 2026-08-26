"""Shared MCP management commands used by CLI and REPL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console
from rich.table import Table

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..i18n import _
from ..settings import MCP_PATH, Settings
from .loader import MCPConfigError, _build_mcp_connection, _read_main_config

if TYPE_CHECKING:
    from ..agent import AgentRuntime


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
    try:
        conn = _build_mcp_connection(name, cfg)
    except MCPConfigError as exc:
        return MCPServerStatus(name=name, transport="", target="", status="failed", tools=[], error=str(exc))

    transport = str(conn["transport"])
    if "command" in conn:
        target = f"{conn['command']} {' '.join(conn['args'])}".strip()
    else:
        target = str(conn["url"])

    try:
        client = MultiServerMCPClient({name: conn})  # type: ignore[dict-item,arg-type]
        tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
    except asyncio.TimeoutError:
        return MCPServerStatus(
            name=name,
            transport=transport,
            target=target,
            status="failed",
            tools=[],
            error=_("Connection timed out ({timeout}s)", timeout=int(timeout)),
        )
    except Exception as exc:
        return MCPServerStatus(name=name, transport=transport, target=target, status="failed", tools=[], error=str(exc))

    return MCPServerStatus(
        name=name,
        transport=transport,
        target=target,
        status="active",
        tools=[t.name for t in tools],
        error="",
    )


async def list_mcp_servers(
    console: Console,
    settings: Settings | None = None,
) -> None:
    """List all configured MCP servers and display their connection status."""
    try:
        servers_cfg = await _read_main_config()
    except MCPConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    subagent_servers: dict[str, dict[str, Any]] = {}
    if settings:
        for sa in settings.subagents:
            for srv in sa.mcp_servers:
                subagent_servers[f"{srv.name} ({sa.name})"] = {
                    "command": srv.command,
                    "args": srv.args,
                    "env": srv.env,
                }

    all_servers = {**servers_cfg, **subagent_servers}
    if not all_servers:
        console.print(
            f"[yellow]{_('No MCP servers configured.')}[/yellow]\n"
            f"[dim]{_('Configure servers in {path}', path=MCP_PATH)}[/dim]"
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


async def reload_mcp_servers(
    console: Console,
    runtime: AgentRuntime,
) -> None:
    """Reload MCP servers and rebuild the agent graph."""
    await runtime.reload()
    console.print(f"[green]✓ {_('MCP servers reloaded successfully.')}[/green]")
    await list_mcp_servers(console, settings=runtime.settings)

