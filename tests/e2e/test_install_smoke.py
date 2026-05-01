"""Smoke tests for the install path itself.

These run *inside* the freshly-created venv, not the dev environment.
Passing here means a real user doing `pip install aether-core[mcp]`
or `[mcp,graph]` in a clean environment will get a working install.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .conftest import build_aether_venv


pytestmark = pytest.mark.e2e


def _run_in_venv(py: Path, code: str) -> subprocess.CompletedProcess:
    """Run a one-liner in the venv's interpreter with output captured."""
    return subprocess.run(
        [str(py), "-c", code],
        capture_output=True,
        text=True,
    )


def test_aether_imports_in_fresh_venv(aether_venv):
    """`import aether` works on a clean install with no dev tooling."""
    result = _run_in_venv(aether_venv, "import aether; print(aether.__version__)")
    assert result.returncode == 0, (
        f"import aether failed:\n  stdout: {result.stdout}\n  stderr: {result.stderr}"
    )
    assert result.stdout.strip() == "0.12.18", (
        f"version mismatch: got {result.stdout.strip()!r}, expected '0.12.18'"
    )


def test_server_builds_under_recommended_install(aether_venv):
    """`[mcp,graph]` extras resolve and `build_server()` succeeds."""
    result = _run_in_venv(
        aether_venv,
        "from aether.mcp.server import build_server; "
        "from aether.mcp.state import StateStore; "
        "import tempfile; from pathlib import Path; "
        "p = Path(tempfile.mkdtemp()) / 'state.json'; "
        "build_server(store=StateStore(state_path=str(p))); print('ok')",
    )
    assert result.returncode == 0, (
        f"server build failed:\n  stdout: {result.stdout}\n  stderr: {result.stderr}"
    )
    assert "ok" in result.stdout


def test_console_scripts_installed(aether_venv):
    """`aether-mcp` and `aether` console scripts land on the venv path."""
    bin_dir = aether_venv.parent
    suffix = ".exe" if aether_venv.suffix == ".exe" else ""
    for name in ("aether-mcp", "aether"):
        script = bin_dir / f"{name}{suffix}"
        assert script.exists(), f"console script missing: {script}"


def test_mcp_only_install_yields_working_server(tmp_path_factory):
    """Regression test: `[mcp]` alone must give a usable server.

    Originally an xfail finding — `[mcp]` didn't pull `networkx` so the
    server crashed on first StateStore construction. The fix declared
    `networkx` as a `[mcp]` dependency. This test pins that contract:
    if anyone splits the extras again, this fails immediately.
    """
    venv_dir = tmp_path_factory.mktemp("aether_venv_mcp_only")
    py = build_aether_venv(venv_dir, extras="mcp")
    result = _run_in_venv(
        py,
        "from aether.mcp.server import build_server; "
        "from aether.mcp.state import StateStore; "
        "import tempfile; from pathlib import Path; "
        "p = Path(tempfile.mkdtemp()) / 'state.json'; "
        "build_server(store=StateStore(state_path=str(p))); print('ok')",
    )
    assert result.returncode == 0, (
        f"[mcp]-only install should yield a working server.\n"
        f"  stdout: {result.stdout}\n  stderr: {result.stderr}"
    )
    assert "ok" in result.stdout
