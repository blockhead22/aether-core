"""v0.12.18: cross-Python launcher + PEP 668 fallback.

A user testing v0.12.17 on a clean M2 Mac surfaced two real upstream
bugs in how aether's plugin handed off to Python:

  1. `.mcp.json` and the hook commands shipped `"command": "python"`,
     which on Homebrew macOS resolves to a PEP 668-locked Python.
     Result: MCP server never started, Stop hook silently failed,
     SessionStart pip-install hit `externally-managed-environment`
     and emitted an empty context, never seeding the substrate.
  2. The user fixed it locally with a venv at `~/.aether-venv/` but
     the plugin had no way to discover it.

Fix:

  - `hooks/aether_launcher.py` — pure-stdlib script that finds a
    Python with aether importable (AETHER_PYTHON env var,
    ~/.aether-venv, ~/aether-venv, $VIRTUAL_ENV, sys.executable in
    that order) and `os.execvp`s the requested command on it.
    `.mcp.json` and `hooks.json` invoke the launcher.
  - `hooks/session_start.py` detects PEP 668 from pip stderr, falls
    back to creating ~/.aether-venv on the fly, re-execs self via
    that venv's Python so the rest of the script sees aether.
  - SessionStart now logs the LAST 800 chars of pip stderr on
    failure (was: just the Python exception type name) so future
    install issues are diagnosable from the log file alone.

These tests cover the launcher discovery + PEP 668 detection. The
re-exec / venv-creation paths require a real subprocess and aren't
unit-testable cleanly; smoke-tested manually.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------------------
# aether_launcher.py — load it as a module without going through the package.
# It's a top-level script in hooks/, not part of `aether/`.
# --------------------------------------------------------------------------

def _load_launcher():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "aether_launcher_mod", repo_root / "hooks" / "aether_launcher.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_session_start():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "session_start_mod", repo_root / "hooks" / "session_start.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# Launcher: find_python discovery order
# ==========================================================================

class TestLauncherDiscoveryOrder:
    def test_aether_python_env_var_takes_priority(self, monkeypatch):
        """When AETHER_PYTHON is set, the launcher trusts it without
        validating — power-user opt-in."""
        mod = _load_launcher()
        monkeypatch.setenv("AETHER_PYTHON", "/totally/fake/python")
        # Don't even need _aether_importable to succeed — env var wins.
        assert mod.find_python() == "/totally/fake/python"

    def test_falls_through_to_sys_executable_when_aether_importable(
        self, monkeypatch, tmp_path
    ):
        """When no env var and no venv, but aether IS importable in
        the launcher's own Python — use that."""
        mod = _load_launcher()
        monkeypatch.delenv("AETHER_PYTHON", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        # Force home() to point at empty tmp_path so venv lookups miss.
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch.object(mod, "_aether_importable") as mock_imp:
                # First few calls (venv candidates) miss; sys.executable hits.
                mock_imp.side_effect = lambda p: p == sys.executable
                assert mod.find_python() == sys.executable

    def test_returns_none_when_nowhere_has_aether(self, monkeypatch, tmp_path):
        mod = _load_launcher()
        monkeypatch.delenv("AETHER_PYTHON", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch.object(mod, "_aether_importable", return_value=False):
                assert mod.find_python() is None

    def test_prefers_aether_venv_over_sys_executable(self, monkeypatch, tmp_path):
        """If both ~/.aether-venv and sys.executable have aether, the
        venv wins. Reflects user intent — they made the venv on purpose."""
        mod = _load_launcher()
        monkeypatch.delenv("AETHER_PYTHON", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        # Build a fake venv tree.
        venv = tmp_path / ".aether-venv"
        bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
        bin_dir.mkdir(parents=True)
        py_name = "python.exe" if sys.platform == "win32" else "python"
        venv_py = bin_dir / py_name
        venv_py.touch()
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch.object(mod, "_aether_importable", return_value=True):
                # Both paths "have aether" per the mock; venv_py listed first.
                assert mod.find_python() == str(venv_py)


class TestLauncherMain:
    def test_no_args_returns_2(self, capsys):
        mod = _load_launcher()
        rc = mod.main([])
        assert rc == 2
        err = capsys.readouterr().err
        assert "no command given" in err

    def test_no_python_with_aether_returns_127(self, capsys, monkeypatch, tmp_path):
        mod = _load_launcher()
        monkeypatch.delenv("AETHER_PYTHON", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch.object(mod, "_aether_importable", return_value=False):
                rc = mod.main(["-m", "aether.mcp"])
        assert rc == 127
        err = capsys.readouterr().err
        # Diagnostic message must mention the search order + remediation.
        assert "AETHER_PYTHON" in err
        assert "venv" in err.lower()
        assert "pip install" in err


# ==========================================================================
# SessionStart: PEP 668 detection
# ==========================================================================

class TestPEP668Detection:
    def test_externally_managed_string_triggers_detection(self):
        mod = _load_session_start()
        mod._LAST_PIP_STDERR = (
            "error: externally-managed-environment\n"
            "× This environment is externally managed"
        )
        assert mod._is_pep668_error() is True

    def test_other_pip_error_is_not_pep668(self):
        mod = _load_session_start()
        mod._LAST_PIP_STDERR = "ERROR: Could not find a version that satisfies the requirement"
        assert mod._is_pep668_error() is False

    def test_empty_stderr_is_not_pep668(self):
        mod = _load_session_start()
        mod._LAST_PIP_STDERR = ""
        assert mod._is_pep668_error() is False

    def test_case_insensitive(self):
        mod = _load_session_start()
        mod._LAST_PIP_STDERR = "EXTERNALLY-MANAGED-ENVIRONMENT detected"
        assert mod._is_pep668_error() is True


class TestPipInstallCapturesStderr:
    """The bug we're fixing: previously the hook caught
    CalledProcessError and logged only the exception type, not pip's
    actual stderr — making remote diagnosis impossible. The new
    `_pip_install` populates `_LAST_PIP_STDERR` so the caller can
    classify the failure (PEP 668 vs network vs other)."""

    def test_pip_stderr_captured_on_failure(self, monkeypatch):
        mod = _load_session_start()
        mod._LAST_PIP_STDERR = ""

        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stderr = "error: externally-managed-environment\n"
        with patch("subprocess.run", return_value=fake_result):
            ok = mod._pip_install("/fake/python", upgrade=False)
        assert ok is False
        assert "externally-managed-environment" in mod._LAST_PIP_STDERR

    def test_pip_success_returns_true(self):
        mod = _load_session_start()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stderr = ""
        with patch("subprocess.run", return_value=fake_result):
            ok = mod._pip_install("/fake/python", upgrade=False)
        assert ok is True


# ==========================================================================
# SessionStart: _maybe_reexec_via_aether_python doesn't re-exec when not needed
# ==========================================================================

class TestReexecGate:
    def test_no_reexec_when_aether_already_importable(self, monkeypatch):
        """If aether is importable in current Python, _maybe_reexec
        returns silently — no os.execvp call. Critical: a buggy
        re-exec would loop forever."""
        mod = _load_session_start()
        # aether is importable in this test env — just call the function.
        # Patch execvp to detect any unwanted call.
        with patch("os.execvp") as mock_execvp:
            mod._maybe_reexec_via_aether_python()
        assert mock_execvp.call_count == 0
