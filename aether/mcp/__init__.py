"""Aether MCP server.

Exposes the belief substrate as Model Context Protocol tools so any
MCP-speaking AI shell (Claude Code, Cursor, Cline, Continue, Goose,
Zed, LM Studio, ...) can use the same persistent belief state.

Run as a stdio server:
    python -m aether.mcp

State lives in ~/.aether/mcp_state.json by default; override with
the AETHER_STATE_PATH environment variable. State is loaded on
startup and saved after every write.

Imports are lazy so this package can be imported without the optional
[mcp] extra installed. `aether.mcp.state` and `aether.mcp.server`
are importable directly when their dependencies are available.
"""

from __future__ import annotations

__all__ = ["build_server", "run"]


def __getattr__(name: str):
    # Lazy proxy: only pull in `.server` (which requires the `mcp`
    # package) when the caller actually asks for it.
    if name in __all__:
        from .server import build_server, run  # noqa: F401
        return {"build_server": build_server, "run": run}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
