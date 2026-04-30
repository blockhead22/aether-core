"""F#7 fix: StateStore picks up external writes to the state file.

Background — F#7 surfaced 2026-04-30:
    The Stop hook writes facts directly to ~/.aether/mcp_state.json
    while the running MCP server holds graph state in memory. Without
    coordination, the server's saves silently overwrite hook-written
    memories (and vice versa). Result: in a real session, the auto-
    ingest hook works in isolation but loses writes the moment the
    server's StateStore touches the file again.

The fix:
    MemoryGraph tracks `_loaded_mtime` after each load/save and
    exposes `is_stale_on_disk()`. StateStore gains a `_sync_first`
    decorator that reloads from disk before every public method runs.
    External writes now surface on the very next tool call instead of
    being clobbered.

These tests prove the contract:
    - External adds become visible to the next read.
    - Server writes preserve external memories that arrived between
      saves.
    - The cheap path (no external write) doesn't reload unnecessarily.
"""

from __future__ import annotations

import json
import os
import time
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

from aether.mcp.state import StateStore


def _write_external_memory(state_path: Path, memory_id: str, text: str) -> None:
    """Simulate the Stop hook: append a memory to the JSON file
    without going through the StateStore in memory.

    Bumps the file mtime forward so `is_stale_on_disk()` returns True
    on the next StateStore call. We can't rely on natural mtime drift
    because tests run faster than filesystem mtime resolution on some
    platforms.
    """
    with state_path.open("r") as f:
        data = json.load(f)
    ts = time.time()
    data.setdefault("nodes", []).append({
        "id": memory_id,
        "memory_id": memory_id,
        "text": text,
        "created_at": ts,
        "memory_type": "fact",
        "belnap_state": "T",
        "trust": 0.85,
        "confidence": 0.85,
        "valid_at": ts,
        "invalid_at": None,
        "superseded_by": None,
        "tags": ["source:external_test"],
    })
    with state_path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    # Force a future mtime so the staleness check fires deterministically
    # even on filesystems with second-precision mtime.
    future = time.time() + 1
    os.utime(state_path, (future, future))


@needs_networkx
class TestExternalWriteSync:
    """The F#7 contract: external writers surface on the next call."""

    def test_external_add_becomes_visible_on_next_read(self, tmp_path):
        """The core F#7 scenario: hook writes a memory while the server
        is idle; the next read tool sees it without restart."""
        state_path = tmp_path / "state.json"
        store = StateStore(state_path=str(state_path))

        # Server-side write — establishes baseline mtime.
        store.add_memory(text="Server-written memory", trust=0.7)
        baseline = store.stats()["memory_count"]

        # External writer (simulates the Stop hook).
        _write_external_memory(
            state_path,
            memory_id="ext_test_1",
            text="Hook-written memory that arrived between server calls",
        )

        # Next tool call must reflect the external write.
        result = store.stats()
        assert result["memory_count"] == baseline + 1, (
            f"external write not picked up: stats() returned "
            f"{result['memory_count']} memories, expected {baseline + 1}"
        )

        # And the specific memory must be retrievable.
        detail = store.memory_detail("ext_test_1")
        assert "error" not in detail, (
            f"externally-added memory not found via memory_detail: {detail}"
        )

    def test_server_save_does_not_clobber_external_memory(self, tmp_path):
        """The original F#7 bug: server.save() overwrites hook writes
        because the in-memory state was stale.

        After the fix, the server reloads before its next mutation, so
        the post-save file contains BOTH the external write and the
        server's new write.
        """
        state_path = tmp_path / "state.json"
        store = StateStore(state_path=str(state_path))

        store.add_memory(text="Initial server-written memory", trust=0.7)

        _write_external_memory(
            state_path,
            memory_id="ext_test_2",
            text="External memory — must survive the next server save",
        )

        # Server adds a third memory. Without the F#7 fix, the save()
        # would dump the stale in-memory state and lose ext_test_2.
        result = store.add_memory(
            text="Server memory written after the external write", trust=0.7,
        )
        new_id = result["memory_id"]

        # Re-read the file directly to confirm both survived.
        with state_path.open("r") as f:
            data = json.load(f)
        ids = {n.get("id") for n in data["nodes"]}

        assert "ext_test_2" in ids, (
            f"server.save() clobbered the external memory. "
            f"Disk currently has: {sorted(ids)}"
        )
        assert new_id in ids, (
            f"server's own write also missing — disk has: {sorted(ids)}"
        )

    def test_no_reload_when_file_unchanged(self, tmp_path, monkeypatch):
        """Defensive: the cheap path (no external write) must not
        trigger a json.load on every call. Otherwise tools that read
        N times pay N file reads instead of 1.
        """
        state_path = tmp_path / "state.json"
        store = StateStore(state_path=str(state_path))

        store.add_memory(text="Setup memory", trust=0.7)

        load_count = {"n": 0}
        original_load = store.graph.load

        def counting_load(*args, **kwargs):
            load_count["n"] += 1
            return original_load(*args, **kwargs)

        monkeypatch.setattr(store.graph, "load", counting_load)

        # Fire several reads with no external write between them.
        for _ in range(5):
            store.stats()
            store.search("anything", limit=1)

        assert load_count["n"] == 0, (
            f"is_stale_on_disk() incorrectly triggered {load_count['n']} "
            f"reload(s) when nothing changed externally"
        )

    def test_is_stale_on_disk_after_external_touch(self, tmp_path):
        """Direct contract test on MemoryGraph: external mtime bump
        flips is_stale_on_disk to True; a save() flips it back."""
        state_path = tmp_path / "state.json"
        store = StateStore(state_path=str(state_path))
        store.add_memory(text="Anchor", trust=0.7)

        assert not store.graph.is_stale_on_disk(), (
            "freshly-saved file should not appear stale"
        )

        # External writer bumps mtime.
        future = time.time() + 1
        os.utime(state_path, (future, future))

        assert store.graph.is_stale_on_disk(), (
            "external mtime bump should make is_stale_on_disk return True"
        )

        # A subsequent server save resets the baseline.
        store.add_memory(text="Post-staleness write", trust=0.7)

        assert not store.graph.is_stale_on_disk(), (
            "save() should reset _loaded_mtime so we no longer see "
            "our own write as external"
        )
