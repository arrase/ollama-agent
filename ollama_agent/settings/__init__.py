"""Settings package for application configuration."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config
    from .mcp import RunningMCPServer


_EXPORTS = {
    "Config": (".config", "Config"),
    "get_config": (".config", "get_config"),
    "load_instructions": (".config", "load_instructions"),
    "reset_config": (".config", "reset_config"),
    "RunningMCPServer": (".mcp", "RunningMCPServer"),
    "cleanup_mcp_servers": (".mcp", "cleanup_mcp_servers"),
    "initialize_mcp_servers": (".mcp", "initialize_mcp_servers"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "Config",
    "RunningMCPServer",
    "cleanup_mcp_servers",
    "get_config",
    "initialize_mcp_servers",
    "load_instructions",
    "reset_config",
]
