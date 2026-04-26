"""Entrypoint: `python -m aether.mcp`.

Starts a stdio MCP server with the Aether belief substrate.
Plug it into Claude Code, Cursor, or any MCP-speaking client.
"""

from .server import run


if __name__ == "__main__":
    run()
