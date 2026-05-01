"""v0.12.14: layman onboarding — Track 1 of the roadmap.

Goal: a Claude Code user with `claude plugin install aether-core`
should not need a single manual step beyond that. This release ships
the plumbing for that:

  - hooks/session_start.py: replaces the inline pip-install one-liner.
    Auto-installs aether-core, kicks off encoder warmup in the
    background, ensures ~/.aether/mcp_state.json exists with default
    policy beliefs, emits a first-run welcome message via the SessionStart
    additionalContext protocol, falls back to a brief status line on
    subsequent runs.

  - aether status / aether doctor: report a one-line "update available"
    notice when the installed version is behind the latest on PyPI.
    Cached for 24h at ~/.aether/.pypi_version_cache.json. Skip with
    AETHER_NO_UPDATE_CHECK=1.

  - aether uninstall-cleanup: explicit, dry-run-able way to remove
    ~/.aether/ after plugin uninstall. Default is dry-run; pass --yes
    to actually remove. Pass --keep-substrate to preserve mcp_state.json
    + backups while clearing logs / markers / caches.

These tests pin the contracts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aether.cli import (
    _check_pypi_version,
    _format_version_drift_line,
    _gather_cleanup_paths,
    cmd_uninstall_cleanup,
)


# ==========================================================================
# Update notification (Track 1 #6)
# ==========================================================================

class TestPypiVersionCheck:
    def test_returns_none_when_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("AETHER_NO_UPDATE_CHECK", "1")
        assert _check_pypi_version() is None

    def test_uses_cache_when_recent(self, tmp_path, monkeypatch):
        cache = tmp_path / ".aether" / ".pypi_version_cache.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"ts": time.time(), "latest": "9.9.9"}))
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = _check_pypi_version()
        # Cache hit — should return cached value without network.
        assert result == "9.9.9"

    def test_ignores_stale_cache(self, tmp_path, monkeypatch):
        cache = tmp_path / ".aether" / ".pypi_version_cache.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Cache from 2 days ago — should be ignored.
        cache.write_text(json.dumps({"ts": time.time() - 86400 * 2, "latest": "1.0.0"}))
        # Block actual network so we don't depend on PyPI.
        monkeypatch.setenv("AETHER_NO_UPDATE_CHECK", "0")
        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch("aether.cli.urlopen", side_effect=Exception("no network")):
            result = _check_pypi_version()
        # Stale cache + no network → None, not the stale value.
        assert result != "1.0.0"


class TestVersionDriftLine:
    def test_none_when_versions_match(self, monkeypatch):
        monkeypatch.setenv("AETHER_NO_UPDATE_CHECK", "1")
        # Disabled — returns None regardless.
        assert _format_version_drift_line() is None

    def test_emits_line_when_behind(self, monkeypatch):
        with patch("aether.cli._check_pypi_version", return_value="99.99.99"):
            line = _format_version_drift_line()
        assert line is not None
        assert "99.99.99" in line
        assert "pip install -U" in line


# ==========================================================================
# Uninstall cleanup (Track 1 #7)
# ==========================================================================

class TestGatherCleanupPaths:
    def test_returns_empty_when_dir_missing(self, tmp_path):
        targets = _gather_cleanup_paths(tmp_path / "nope", keep_substrate=False)
        assert targets == []

    def test_full_removal_returns_dir_itself(self, tmp_path):
        aether_dir = tmp_path / ".aether"
        aether_dir.mkdir()
        (aether_dir / "mcp_state.json").write_text("{}")
        targets = _gather_cleanup_paths(aether_dir, keep_substrate=False)
        # Full removal = remove the whole directory in one shot.
        assert targets == [aether_dir]

    def test_keep_substrate_preserves_state_files(self, tmp_path):
        aether_dir = tmp_path / ".aether"
        aether_dir.mkdir()
        (aether_dir / "mcp_state.json").write_text("{}")
        (aether_dir / "state_trust_history.json").write_text("{}")
        (aether_dir / "state_receipts.json").write_text("{}")
        (aether_dir / "backups").mkdir()
        (aether_dir / "auto_ingest.log").write_text("log")
        (aether_dir / ".first_run_complete").write_text("ts")
        (aether_dir / "mcp_state.pre_v0_test.json").write_text("{}")

        targets = _gather_cleanup_paths(aether_dir, keep_substrate=True)
        target_names = {t.name for t in targets}
        # Substrate files preserved.
        assert "mcp_state.json" not in target_names
        assert "state_trust_history.json" not in target_names
        assert "state_receipts.json" not in target_names
        assert "backups" not in target_names
        assert "mcp_state.pre_v0_test.json" not in target_names
        # Logs / markers selected for removal.
        assert "auto_ingest.log" in target_names
        assert ".first_run_complete" in target_names


class TestUninstallCleanupCmd:
    def _make_args(self, yes=False, keep_substrate=False):
        ns = argparse.Namespace()
        ns.yes = yes
        ns.keep_substrate = keep_substrate
        return ns

    def test_dry_run_does_not_remove(self, tmp_path, capsys):
        aether_dir = tmp_path / ".aether"
        aether_dir.mkdir()
        (aether_dir / "mcp_state.json").write_text("{}")

        with patch("pathlib.Path.home", return_value=tmp_path):
            rc = cmd_uninstall_cleanup(self._make_args(yes=False))
        assert rc == 0
        # File still exists — dry run.
        assert (aether_dir / "mcp_state.json").exists()
        out = capsys.readouterr().out
        assert "dry-run" in out.lower()

    def test_yes_flag_actually_removes(self, tmp_path, capsys):
        aether_dir = tmp_path / ".aether"
        aether_dir.mkdir()
        (aether_dir / "mcp_state.json").write_text("{}")
        (aether_dir / "auto_ingest.log").write_text("log")

        with patch("pathlib.Path.home", return_value=tmp_path):
            rc = cmd_uninstall_cleanup(self._make_args(yes=True))
        assert rc == 0
        # Whole directory gone.
        assert not aether_dir.exists()

    def test_keep_substrate_preserves_state(self, tmp_path):
        aether_dir = tmp_path / ".aether"
        aether_dir.mkdir()
        (aether_dir / "mcp_state.json").write_text('{"nodes":[]}')
        (aether_dir / "auto_ingest.log").write_text("log")
        (aether_dir / ".first_run_complete").write_text("ts")

        with patch("pathlib.Path.home", return_value=tmp_path):
            rc = cmd_uninstall_cleanup(self._make_args(yes=True, keep_substrate=True))
        assert rc == 0
        # State preserved, logs gone.
        assert (aether_dir / "mcp_state.json").exists()
        assert not (aether_dir / "auto_ingest.log").exists()
        assert not (aether_dir / ".first_run_complete").exists()

    def test_empty_dir_is_a_no_op(self, tmp_path, capsys):
        # No ~/.aether/ at all.
        with patch("pathlib.Path.home", return_value=tmp_path):
            rc = cmd_uninstall_cleanup(self._make_args(yes=True))
        assert rc == 0
        out = capsys.readouterr().out
        assert "nothing to remove" in out.lower()


# ==========================================================================
# SessionStart hook helpers (Track 1 #1, #2, #4)
# ==========================================================================

class TestSessionStartHelpers:
    """The hook script is at hooks/session_start.py, not importable as a
    module from aether/. We test its helpers by importing the file
    directly via importlib, then exercising the functions in isolation.
    """

    def _load_hook_module(self):
        import importlib.util
        repo_root = Path(__file__).resolve().parent.parent
        hook_path = repo_root / "hooks" / "session_start.py"
        spec = importlib.util.spec_from_file_location("session_start_hook", hook_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_first_run_detection(self, tmp_path):
        mod = self._load_hook_module()
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert mod._is_first_run() is True
            mod._mark_first_run_complete()
            assert mod._is_first_run() is False

    def test_welcome_message_contains_key_facts(self):
        mod = self._load_hook_module()
        info = {"existed": False, "seeded": 7, "memory_count": 7, "encoder_mode": "warming"}
        msg = mod._welcome_message(info)
        # Must mention the key affordances so the user knows what to do.
        assert "Aether" in msg
        assert "7 default policy beliefs" in msg
        assert "/aether-status" in msg
        assert "aether doctor" in msg

    def test_status_message_is_brief(self):
        mod = self._load_hook_module()
        info = {"existed": True, "seeded": 0, "memory_count": 127, "encoder_mode": "warm"}
        msg = mod._status_message(info)
        # Brief = doesn't include onboarding stuff.
        assert "127 memories" in msg
        assert "warm" in msg
        assert "/aether-status" not in msg
