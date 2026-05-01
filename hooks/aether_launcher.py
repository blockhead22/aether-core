#!/usr/bin/env python3
"""hooks/aether_launcher.py — find a Python that has aether installed
and re-exec the requested command on that interpreter.

Why this exists. Claude Code's `.mcp.json` and `hooks.json` both
specify the interpreter as a literal `command`. On a default
Homebrew macOS install (Python 3.11+), that interpreter is PEP 668
locked — `pip install aether-core` fails — so the user works around
it by creating a venv at `~/.aether-venv/`. But Claude Code's hook
config still hardcodes `python` and points at the PEP 668 lockout.
Result: the MCP server can't start, the auto-ingest Stop hook
silently fails, the SessionStart hook can't pip-install — all of it
broken on the platform the README claims supports laymen.

This launcher fixes that. It's pure stdlib so any 3.10+ Python
that exists on the user's PATH can run it. The `command` in MCP /
hook config calls the launcher; the launcher then discovers a
Python that actually has aether installed and uses `os.execvp` to
replace itself, so MCP's stdio protocol is unaffected.

Discovery order (first match wins):

  1. ``$AETHER_PYTHON`` — explicit override. Used as-is, not
     validated. The user opted in.
  2. ``~/.aether-venv/bin/python`` (Unix) or
     ``~/.aether-venv/Scripts/python.exe`` (Windows) — the
     conventional location our SessionStart hook creates.
  3. ``~/aether-venv/...`` — non-dotted variant, common style.
  4. ``$VIRTUAL_ENV/bin/python`` — if the user has a venv
     activated and aether is importable in it.
  5. The launcher's own ``sys.executable`` — for the case where
     aether is installed in the system Python the user invoked us
     with.

If none have aether importable, the launcher exits 127 with a clear
remediation message on stderr — never silently. This is the inverse
of the SessionStart hook's old "swallow the error" behavior.

Usage examples:

    # As an MCP server entrypoint (.mcp.json):
    {"command": "python3",
     "args": ["${CLAUDE_PLUGIN_ROOT}/hooks/aether_launcher.py",
              "-m", "aether.mcp"]}

    # As a hook wrapper (hooks/hooks.json):
    {"command":
       "python3 \\"${CLAUDE_PLUGIN_ROOT}/hooks/aether_launcher.py\\""
       " \\"${CLAUDE_PLUGIN_ROOT}/hooks/session_start.py\\""}

    # Manual smoke test from a shell:
    python3 hooks/aether_launcher.py -c "import aether; print(aether.__version__)"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _venv_python(venv: Path) -> Path:
    """Path to the python interpreter inside a venv on this OS."""
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _aether_importable(py: str) -> bool:
    """True if running ``py -c 'import aether'`` succeeds."""
    if not py:
        return False
    if not Path(py).exists():
        return False
    try:
        result = subprocess.run(
            [py, "-c", "import aether"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def find_python() -> Optional[str]:
    """Return a Python path with aether importable, or None."""
    # 1. Explicit env var — trust the user without validation.
    explicit = os.environ.get("AETHER_PYTHON", "").strip()
    if explicit:
        return explicit

    # 2. + 3. Conventional venv locations.
    home = Path.home()
    for venv in (home / ".aether-venv", home / "aether-venv"):
        py = _venv_python(venv)
        if _aether_importable(str(py)):
            return str(py)

    # 4. Active virtual environment.
    active = os.environ.get("VIRTUAL_ENV", "").strip()
    if active:
        py = _venv_python(Path(active))
        if _aether_importable(str(py)):
            return str(py)

    # 5. Our own interpreter.
    if _aether_importable(sys.executable):
        return sys.executable

    return None


def main(argv: List[str]) -> int:
    if not argv:
        print(
            "aether_launcher: no command given. "
            "Pass the args you want to forward, e.g. `-m aether.mcp`.",
            file=sys.stderr,
        )
        return 2

    py = find_python()
    if not py:
        print(
            "aether_launcher: could not find a Python with aether-core "
            "installed.\n\n"
            "Searched (in order):\n"
            "  - $AETHER_PYTHON env var\n"
            "  - ~/.aether-venv/bin/python (Unix) or "
            "~/.aether-venv/Scripts/python.exe (Windows)\n"
            "  - ~/aether-venv/...\n"
            "  - $VIRTUAL_ENV/bin/python\n"
            f"  - this launcher's own interpreter ({sys.executable})\n\n"
            "Fix:\n"
            "  pip install 'aether-core[mcp,graph,ml]'\n"
            "If your platform is PEP 668 locked (Homebrew Python, recent "
            "Debian/Ubuntu), create a venv:\n"
            "  python3 -m venv ~/.aether-venv\n"
            "  ~/.aether-venv/bin/pip install 'aether-core[mcp,graph,ml]'\n"
            "Then either restart Claude Code or set:\n"
            "  export AETHER_PYTHON=~/.aether-venv/bin/python",
            file=sys.stderr,
        )
        return 127

    # Replace this process so MCP stdio passes through untouched.
    try:
        os.execvp(py, [py] + argv)
    except OSError as e:
        print(f"aether_launcher: execvp({py}) failed: {e}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
