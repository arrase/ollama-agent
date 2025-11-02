"""MCP servers configuration and initialization."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

logger = logging.getLogger(__name__)

DEFAULT_MCP_CONFIG_PATH = Path.home() / ".ollama-agent" / "mcp_servers.json"


def _create_stdio_server(name: str, config: dict[str, Any]) -> Optional[MCPServerStdio]:
    """Create stdio MCP server from config."""
    if "command" not in config:
        return None
    return MCPServerStdio(
        name=name,
        params={"command": config["command"], "args": config.get("args", [])},  # type: ignore[arg-type]
        cache_tools_list=config.get("cache_tools_list", True),
        max_retry_attempts=config.get("max_retry_attempts", 3)
    )


def _create_http_server(name: str, config: dict[str, Any]) -> Optional[MCPServer]:
    """Create HTTP (SSE or streamable) MCP server from config."""
    if "httpUrl" not in config:
        return None
    
    server_type = config.get("type", "streamable_http")
    params = {
        "url": config["httpUrl"],
        "headers": config.get("headers", {}),
        "timeout": config.get("timeout", 10)
    }
    
    if server_type == "sse":
        return MCPServerSse(
            name=name,
            params=params,  # type: ignore[arg-type]
            cache_tools_list=config.get("cache_tools_list", True),
            max_retry_attempts=config.get("max_retry_attempts", 3)
        )
    else:  # streamable_http
        return MCPServerStreamableHttp(
            name=name,
            params=params,  # type: ignore[arg-type]
            cache_tools_list=config.get("cache_tools_list", True),
            max_retry_attempts=config.get("max_retry_attempts", 3)
        )


async def initialize_mcp_servers(config_path: Optional[Path] = None) -> list[MCPServer]:
    """
    Initialize MCP servers from JSON config file.
    
    Supports standard format:
    {
      "mcpServers": {
        "name": { "command": "...", "args": [...] },  // stdio
        "name": { "httpUrl": "...", "headers": {...} }  // http
      }
    }
    """
    config_path = config_path or DEFAULT_MCP_CONFIG_PATH
    
    if not config_path.exists():
        return []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "mcpServers" not in data:
            logger.warning(f"Invalid MCP config format in {config_path}")
            return []
        
        servers: list[MCPServer] = []
        for name, config in data["mcpServers"].items():
            # Try stdio first, then http
            server = _create_stdio_server(name, config) or _create_http_server(name, config)
            
            if server:
                try:
                    await server.__aenter__()  # type: ignore[attr-defined]
                    servers.append(server)
                    logger.info(f"Initialized MCP server: {name}")
                except Exception as e:
                    logger.error(f"Failed to connect to MCP server '{name}': {e}")
        
        return servers
        
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error loading MCP config: {e}")
        return []


async def cleanup_mcp_servers(servers: list[MCPServer]) -> None:
    """Cleanup MCP server connections."""
    for server in servers:
        try:
            await server.__aexit__(None, None, None)  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug(f"Error cleaning up MCP server: {e}")
