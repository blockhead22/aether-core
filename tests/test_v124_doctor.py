"""v0.12.4: `aether doctor` diagnostic command.

This subcommand runs a battery of checks designed to catch the kinds
of silent-breakage bugs we hit across v0.9-v0.12 (F#1: missing
extra; F#3: corrupt nodes; F#7: stale auto-ingest). Each check
returns ok / warn / fail with an actionable message.

These tests pin the contract:
- The command runs without crashing in normal conditions.
- JSON output is well-formed.
- A corrupt state file is detected.
- Exit code reflects the worst severity (fail -> 1, otherwise 0).
"""

from __future__ import annotations

import json
import io
import sys
from pathlib import Path

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx required (install [graph] extra)",
)

from aether.cli import (
    _doctor_install_imports,
    _doctor_state_file,
    _doctor_substrate_activity,
    cmd_doctor,
)


def _capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        rv = fn(*args, **kwargs)
    finally:
        sys.stdout = saved
    return rv, buf.getvalue()


class TestIndividualChecks:
    """Each check should run cleanly and return a status dict."""

    def test_install_imports_returns_status(self):
        result = _doctor_install_imports()
        assert result["status"] in ("ok", "warn", "fail")
        assert result["name"] == "install_imports"
        assert isinstance(result["messages"], list)
        assert result["messages"], "expected at least one message"

    def test_state_file_missing_warns(self, tmp_path):
        result = _doctor_state_file(str(tmp_path / "nonexistent.json"))
        assert result["status"] == "warn"
        # Doesn't crash on missing — it's the fresh-install case.
        assert any("not exist" in m for m in result["messages"])

    def test_state_file_corrupt_node_fails(self, tmp_path):
        """F#3 detection: nodes missing required fields trip the check."""
        state = tmp_path / "state.json"
        state.write_text(json.dumps({
            "nodes": [
                {"id": "good_node", "memory_id": "good_node", "text": "fine",
                 "created_at": 1.0},
                {"id": "zombie_node"},  # missing memory_id, text, created_at
            ],
            "edges": [],
        }))
        result = _doctor_state_file(str(state))
        assert result["status"] == "fail"
        assert any("corrupt node" in m.lower() for m in result["messages"])
        assert any("zombie_node" in m for m in result["messages"])

    def test_state_file_clean_passes(self, tmp_path):
        state = tmp_path / "state.json"
        state.write_text(json.dumps({
            "aether_version": "0.12.4",
            "nodes": [
                {"id": "n1", "memory_id": "n1", "text": "anchor",
                 "created_at": 1.0},
            ],
            "edges": [],
        }))
        result = _doctor_state_file(str(state))
        assert result["status"] == "ok"

    def test_state_file_unparseable_fails(self, tmp_path):
        state = tmp_path / "broken.json"
        state.write_text("{not valid json,,")
        result = _doctor_state_file(str(state))
        assert result["status"] == "fail"
        assert any("corrupt" in m.lower() for m in result["messages"])

    def test_substrate_activity_missing_warns(self, tmp_path):
        result = _doctor_substrate_activity(str(tmp_path / "nope.json"))
        assert result["status"] == "warn"


@needs_networkx
class TestDoctorCommand:
    """End-to-end: cmd_doctor() runs, prints, and exits with the right code."""

    def _args(self, **kw):
        class A:
            pass
        a = A()
        a.format = kw.get("format", "text")
        a.state_path = kw.get("state_path")
        return a

    def test_runs_without_crashing(self, tmp_path):
        # Empty tmp_path means state_file warns rather than fails;
        # other checks may warn or pass depending on environment.
        # Either way, the command must not crash.
        rv, out = _capture_stdout(cmd_doctor, self._args(state_path=str(tmp_path / "s.json")))
        assert rv in (0, 1)
        assert "summary:" in out

    def test_json_output_is_well_formed(self, tmp_path):
        rv, out = _capture_stdout(
            cmd_doctor,
            self._args(format="json", state_path=str(tmp_path / "s.json")),
        )
        payload = json.loads(out)
        assert "checks" in payload
        assert isinstance(payload["checks"], list)
        for c in payload["checks"]:
            assert c["status"] in ("ok", "warn", "fail")
            assert isinstance(c["messages"], list)
            assert "name" in c

    def test_exit_code_reflects_failures(self, tmp_path):
        """If a state file is corrupt (F#3 shape), doctor exits 1."""
        state = tmp_path / "state.json"
        state.write_text(json.dumps({
            "nodes": [{"id": "zombie"}],  # missing required fields
            "edges": [],
        }))
        rv, _ = _capture_stdout(
            cmd_doctor,
            self._args(format="json", state_path=str(state)),
        )
        assert rv == 1
