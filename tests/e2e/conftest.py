"""End-to-end test fixtures.

Difference from in-process slice tests in `tests/`: those import
`aether` from the development checkout and call functions directly. The
e2e suite drives Aether the way a real user does — pip-install in a
fresh venv, spawn the MCP server as a subprocess, talk over the JSON-RPC
stdio wire that real clients use.

What this catches that slice tests miss:

- Broken or missing `extras_require` (the silent kind that only fails
  on a clean install).
- Console-script entry points that didn't get installed.
- Wire-protocol regressions (tool registration, schema, response shape).
- `~/.aether/` state-path interactions across process boundaries.

Cost: a session-scoped venv build (~30-60s the first time). After that
every e2e test reuses the same venv but gets a fresh state file.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import venv
from contextlib import asynccontextmanager
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run_pip(py: Path, *args: str) -> None:
    """pip install with captured output — surface failures with full context."""
    result = subprocess.run(
        [str(py), "-m", "pip", "install", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "pip install failed:\n"
            f"  cmd: {' '.join(args)}\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )


def build_aether_venv(venv_dir: Path, extras: str = "mcp,graph") -> Path:
    """Create a fresh venv at `venv_dir` and pip-install aether-core[extras].

    Returns the venv's Python executable. Used by both the session-scoped
    `aether_venv` fixture and ad-hoc tests that want to exercise a
    specific extras combination (e.g. an `[mcp]`-alone install to
    document a known gap).
    """
    venv.create(venv_dir, with_pip=True, clear=True)
    py = _venv_python(venv_dir)
    _run_pip(py, "--upgrade", "pip", "--quiet")
    _run_pip(py, f"{REPO_ROOT}[{extras}]", "--quiet")
    return py


@pytest.fixture(scope="session")
def aether_venv(tmp_path_factory) -> Path:
    """Session-scoped venv with `[mcp,graph]` installed.

    `[ml]` is intentionally excluded — sentence-transformers is ~1GB
    and the cold/warm-mode test owns that install path. `[mcp,graph]`
    matches what `NEXT_SESSION.md` calls the working install for the
    full-loop scenario.
    """
    return build_aether_venv(tmp_path_factory.mktemp("aether_venv"))


@pytest.fixture
def aether_state_path(tmp_path) -> Path:
    """Per-test isolated state file. Pass via `AETHER_STATE_PATH` env."""
    return tmp_path / "aether_state.json"


@asynccontextmanager
async def mcp_session(venv_python: Path, state_path: Path):
    """Spawn `python -m aether.mcp` in `venv_python` and yield an
    initialized `mcp.ClientSession`.

    Each call to this helper gives the server its own state file, so
    tests can't leak memories across each other.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=str(venv_python),
        args=["-m", "aether.mcp"],
        env={
            **os.environ,
            "AETHER_STATE_PATH": str(state_path),
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def run_async(coro):
    """Run an async coroutine from a sync test body.

    Avoids a hard dependency on pytest-asyncio. Each test that needs
    the MCP wire wraps its body in `async def run(): ...` and calls
    `run_async(run())` at the end.
    """
    return asyncio.run(coro)
