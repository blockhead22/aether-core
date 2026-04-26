"""Aether MCP server.

Exposes the belief substrate as Model Context Protocol tools so any
MCP-speaking AI shell (Claude Code, Cursor, Cline, Continue, Goose,
Zed, LM Studio, ...) can use the same persistent belief state.

Run as a stdio server:
    python -m aether.mcp

State lives in ~/.aether/mcp_state.json by default; override with
the AETHER_STATE_PATH environment variable. State is loaded on
startup and saved after every write.
"""

from .server import build_server, run

__all__ = ["build_server", "run"]
