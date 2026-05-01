"""v0.12.11: ``aether doctor --report`` bundle for GitHub issue submission.

Production-readiness gap #4: when a stranger hits a problem, getting
useful issue reports requires asking them for `aether doctor --format
json`, OS, python version, install path, log tails — multiple
back-and-forths before reproduction is even possible.

The fix: ``aether doctor --report`` outputs a single self-contained
markdown bundle with environment, doctor results, and tails of the
three diagnostic log files in `~/.aether/`. The new
`.github/ISSUE_TEMPLATE/bug_report.yml` requires this bundle in every
new bug, so the issue arrives reproducible.

These tests pin the bundle's shape and the privacy contract (no state
file contents, no backup contents).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")

from aether.cli import _build_report, _read_log_tail


def _stub_results():
    return [
        {"status": "ok", "name": "install_imports", "messages": ["all good"]},
        {"status": "ok", "name": "state_file", "messages": ["129 nodes"]},
        {"status": "warn", "name": "encoder", "messages": ["cold mode"]},
    ]


class TestBuildReport:
    def test_includes_environment_section(self):
        out = _build_report(_stub_results(), state_path=None)
        assert "## Environment" in out
        assert "aether-core:" in out
        assert "python:" in out
        assert "platform:" in out

    def test_includes_doctor_section(self):
        out = _build_report(_stub_results(), state_path=None)
        assert "## `aether doctor`" in out
        assert "install_imports" in out
        assert "state_file" in out
        assert "encoder" in out
        # Sigils render in the bundle.
        assert "[ OK ]" in out
        assert "[WARN]" in out

    def test_summary_counts_match_results(self):
        out = _build_report(_stub_results(), state_path=None)
        # 2 ok, 1 warn, 0 fail
        assert "summary: 2 ok, 1 warn, 0 fail" in out

    def test_does_not_dump_state_file_contents(self, tmp_path, monkeypatch):
        """The report MUST NOT include substrate memory contents.
        The state file path metadata is fine; the contents are private.
        """
        state_path = tmp_path / "state.json"
        secret_text = "USER_FAVORITE_COLOR_IS_PURPLE_DO_NOT_LEAK"
        state_path.write_text(
            '{"nodes": [{"text": "' + secret_text + '"}], "edges": []}'
        )
        out = _build_report(_stub_results(), state_path=str(state_path))
        assert secret_text not in out

    def test_redaction_review_disclaimer_present(self):
        """Bundle nudges the user to review before pasting."""
        out = _build_report(_stub_results(), state_path=None)
        assert "review" in out.lower() or "Review" in out


class TestLogTailHelper:
    def test_returns_none_for_missing_file(self, tmp_path):
        assert _read_log_tail(tmp_path / "nope.log") is None

    def test_returns_last_n_lines(self, tmp_path):
        log = tmp_path / "demo.log"
        log.write_text("\n".join(f"line {i}" for i in range(20)))
        out = _read_log_tail(log, lines=5)
        assert out is not None
        # Last 5 of lines 0..19 -> lines 15..19.
        assert "line 19" in out
        assert "line 15" in out
        # Lines older than the cutoff are not present.
        assert "line 14" not in out
        assert "line 10" not in out

    def test_returns_short_log_intact(self, tmp_path):
        log = tmp_path / "short.log"
        log.write_text("only line")
        out = _read_log_tail(log, lines=10)
        assert out == "only line"


class TestReportIncludesAvailableLogs:
    def test_includes_log_section_when_log_exists(self, tmp_path, monkeypatch):
        # Redirect HOME to a tmp_path with one of the diagnostic logs.
        aether_dir = tmp_path / ".aether"
        aether_dir.mkdir()
        log = aether_dir / "encoder_warmup.log"
        log.write_text("2026-04-30 22:30:00  load_attempt model=foo\n"
                       "2026-04-30 22:30:18  load_ok size=22MB\n")
        with patch("pathlib.Path.home", return_value=tmp_path):
            out = _build_report(_stub_results(), state_path=None)
        assert "Encoder warmup log" in out
        assert "load_ok" in out

    def test_skips_missing_log_section(self, tmp_path):
        # No logs present in the empty .aether dir -> no log sections.
        aether_dir = tmp_path / ".aether"
        aether_dir.mkdir()
        with patch("pathlib.Path.home", return_value=tmp_path):
            out = _build_report(_stub_results(), state_path=None)
        assert "Encoder warmup log" not in out
        assert "Auto-ingest log" not in out
