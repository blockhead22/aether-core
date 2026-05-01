"""Tests for v0.7.0:
    - Repo-aware substrate discovery (.aether/state.json walks up tree)
    - aether init / status / check / contradictions CLI
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Tests that build a real StateStore need networkx for MemoryGraph.
# Mark them so they skip cleanly when the [graph] extra isn't installed.
try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX,
    reason="networkx required (install [graph] extra)",
)

from aether.mcp.state import _find_repo_state, _default_state_path
from aether.cli import build_parser, cmd_init, cmd_check  # noqa: F401


# --------------------------------------------------------------------------
# Repo discovery
# --------------------------------------------------------------------------

class TestRepoDiscovery:
    def test_find_repo_state_locates_dotaether(self, tmp_path):
        repo = tmp_path / "myproj"
        (repo / ".aether").mkdir(parents=True)
        (repo / ".aether" / "state.json").write_text("{}")
        nested = repo / "src" / "deep"
        nested.mkdir(parents=True)

        found = _find_repo_state(start_dir=str(nested))
        assert found is not None
        assert found.endswith("state.json")

    def test_find_repo_state_returns_none_when_absent(self, tmp_path):
        cwd = tmp_path / "noaether"
        cwd.mkdir()
        found = _find_repo_state(start_dir=str(cwd))
        assert found is None

    def test_default_path_prefers_repo_when_in_tree(self, tmp_path, monkeypatch):
        repo = tmp_path / "myproj"
        (repo / ".aether").mkdir(parents=True)
        (repo / ".aether" / "state.json").write_text("{}")

        monkeypatch.chdir(repo)
        monkeypatch.delenv("AETHER_STATE_PATH", raising=False)
        monkeypatch.delenv("AETHER_NO_REPO_DISCOVERY", raising=False)

        path = _default_state_path()
        assert ".aether" in path
        assert path.endswith("state.json")

    def test_no_repo_discovery_env_var_disables(self, tmp_path, monkeypatch):
        repo = tmp_path / "myproj"
        (repo / ".aether").mkdir(parents=True)
        (repo / ".aether" / "state.json").write_text("{}")

        monkeypatch.chdir(repo)
        monkeypatch.delenv("AETHER_STATE_PATH", raising=False)
        monkeypatch.setenv("AETHER_NO_REPO_DISCOVERY", "1")

        path = _default_state_path()
        # Should NOT use repo path; should fall back to user-global
        assert ".aether" in path
        # Acceptable: the user-global also has .aether/, but it should NOT
        # be the repo directory.
        assert str(repo) not in path


# --------------------------------------------------------------------------
# CLI: init
# --------------------------------------------------------------------------

class TestCLIInit:
    def test_init_creates_dotaether(self, tmp_path):
        target = tmp_path / "newproj"
        target.mkdir()

        # v0.12.13: pass --no-defaults so this test still snapshots the
        # empty-substrate shape. The default-policy seeding is exercised
        # in test_v1213_default_policy_beliefs.py.
        parser = build_parser()
        args = parser.parse_args(["init", "--dir", str(target), "--no-defaults"])
        rc = args.func(args)
        assert rc == 0

        d = target / ".aether"
        assert d.exists()
        assert (d / "state.json").exists()
        assert (d / "state_trust_history.json").exists()
        assert (d / "README.md").exists()
        assert (d / ".gitignore").exists()

        # state.json should be a valid empty graph
        data = json.loads((d / "state.json").read_text())
        assert data == {"nodes": [], "edges": []}

    def test_init_refuses_existing_without_force(self, tmp_path):
        target = tmp_path / "existing"
        (target / ".aether").mkdir(parents=True)

        parser = build_parser()
        args = parser.parse_args(["init", "--dir", str(target)])
        rc = args.func(args)
        assert rc == 1


# --------------------------------------------------------------------------
# CLI: check (the big one)
# --------------------------------------------------------------------------

@needs_networkx
class TestCLICheck:
    def test_check_returns_zero_when_substrate_supports(
        self, tmp_path, monkeypatch
    ):
        # Set up an empty .aether/ in cwd
        repo = tmp_path / "proj"
        (repo / ".aether").mkdir(parents=True)
        (repo / ".aether" / "state.json").write_text(
            json.dumps({"nodes": [], "edges": []}),
        )
        monkeypatch.chdir(repo)
        monkeypatch.setenv("AETHER_STATE_PATH",
                           str(repo / ".aether" / "state.json"))

        # Hedged commit message + empty substrate => SAFE
        msg_file = repo / "MSG"
        msg_file.write_text(
            "Possibly fixes the foo issue. Likely covers the bar case."
        )

        parser = build_parser()
        args = parser.parse_args([
            "check", "--message-file", str(msg_file),
            "--format", "json",
        ])
        rc = args.func(args)
        # SAFE returns 0
        assert rc == 0

    def test_check_blocks_when_substrate_contradicts(
        self, tmp_path, monkeypatch, capsys
    ):
        # Pre-populate substrate with a strong belief
        from aether.mcp.state import StateStore
        path = tmp_path / "state.json"
        store = StateStore(state_path=str(path))
        store.add_memory("Main branch is protected, never force push", trust=0.95)

        # Now point CLI at the same substrate
        monkeypatch.setenv("AETHER_STATE_PATH", str(path))
        msg_file = tmp_path / "MSG"
        msg_file.write_text(
            "We will absolutely force push to main now. The answer is "
            "definitely to force push and there is no question about it."
        )

        parser = build_parser()
        args = parser.parse_args([
            "check", "--message-file", str(msg_file),
            "--fail-severity", "ELEVATED",
        ])
        rc = args.func(args)
        # ELEVATED or worse -> non-zero exit
        assert rc != 0


# --------------------------------------------------------------------------
# Subprocess smoke (use console script if installed)
# --------------------------------------------------------------------------

class TestCLISubprocess:
    def test_aether_init_via_python_module(self, tmp_path, monkeypatch):
        target = tmp_path / "subprocproj"
        target.mkdir()
        result = subprocess.run(
            [sys.executable, "-m", "aether.cli", "init",
             "--dir", str(target)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (target / ".aether" / "state.json").exists()
