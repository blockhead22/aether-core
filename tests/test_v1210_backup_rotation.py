"""v0.12.10: rotating backups + atomic write for state-file saves.

Production-readiness gap: the substrate's single state file at
``~/.aether/mcp_state.json`` was the entire substrate. A corrupt
write, a botched import, or a process killed mid-write would take
the substrate with it. The bulk-import script wrote one-off backups
manually (``mcp_state.pre_*``) but the live save path had no rotation.

The fix:
    Every ``MemoryGraph.save()`` snapshots the previous state file to
    a sibling ``backups/`` directory under a timestamped filename
    before overwriting, then atomic-writes via ``{path}.tmp`` +
    ``os.replace``. The rotation prunes to the most recent N (default
    5; ``AETHER_BACKUP_KEEP``). ``AETHER_DISABLE_BACKUPS=1`` skips
    the rotation entirely.

These tests pin the contract:
    - First save creates no backup (nothing to back up).
    - Second save creates one backup.
    - Subsequent saves keep the most recent N, drop older.
    - AETHER_DISABLE_BACKUPS=1 short-circuits.
    - AETHER_BACKUP_KEEP=N is honored.
    - Atomic write does not leave a .tmp file behind on success.
    - The .tmp leftover from a crashed write is harmless on the
      next save (gets overwritten via os.replace).
"""

from __future__ import annotations

import json
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

from aether.memory.graph import (
    MemoryGraph,
    MemoryNode,
    _atomic_write_json,
    _backup_keep,
    _rotate_backups,
)


def _add(graph: MemoryGraph, text: str, trust: float = 0.8) -> str:
    """Test helper: build a MemoryNode and add it. Production paths
    (StateStore.add_memory, ingest_turn) build the node themselves;
    inside graph.py it's the dataclass-direct API."""
    import uuid
    node = MemoryNode(
        memory_id=f"mem_{uuid.uuid4().hex[:8]}",
        text=text,
        created_at=time.time(),
        trust=trust,
        confidence=trust,
    )
    return graph.add_memory(node)


# ==========================================================================
# Backup rotation
# ==========================================================================

@needs_networkx
class TestBackupRotation:
    def _list_backups(self, state_path: Path) -> list[Path]:
        backup_dir = state_path.parent / "backups"
        if not backup_dir.exists():
            return []
        return sorted(
            backup_dir.glob(f"{state_path.stem}.*.json"),
            key=lambda p: p.stat().st_mtime,
        )

    def test_first_save_creates_no_backup(self, tmp_path):
        """No prior file -> nothing to back up. Saves cleanly."""
        state_path = tmp_path / "state.json"
        graph = MemoryGraph(persist_path=str(state_path))
        _add(graph, "initial")
        graph.save()
        assert state_path.exists()
        assert self._list_backups(state_path) == []

    def test_second_save_creates_one_backup(self, tmp_path):
        state_path = tmp_path / "state.json"
        graph = MemoryGraph(persist_path=str(state_path))
        _add(graph, "first")
        graph.save()
        # Sleep a moment so backup mtime is unambiguously after first save.
        time.sleep(0.01)
        _add(graph, "second")
        graph.save()
        backups = self._list_backups(state_path)
        assert len(backups) == 1
        # The backup contains the *previous* state (before the second save).
        with backups[0].open() as f:
            backup_data = json.load(f)
        assert len(backup_data["nodes"]) == 1

    def test_rotation_keeps_only_n_most_recent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AETHER_BACKUP_KEEP", "3")
        state_path = tmp_path / "state.json"
        graph = MemoryGraph(persist_path=str(state_path))
        # 6 saves -> 5 rotations (the first save creates no backup).
        # With keep=3, we should end up with 3 backups.
        for i in range(6):
            _add(graph, f"memory {i}")
            graph.save()
            time.sleep(0.01)
        backups = self._list_backups(state_path)
        assert len(backups) == 3, (
            f"expected 3 backups with AETHER_BACKUP_KEEP=3, got {len(backups)}: "
            f"{[b.name for b in backups]}"
        )

    def test_disable_backups_env_skips_rotation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AETHER_DISABLE_BACKUPS", "1")
        state_path = tmp_path / "state.json"
        graph = MemoryGraph(persist_path=str(state_path))
        for i in range(3):
            _add(graph, f"memory {i}")
            graph.save()
            time.sleep(0.01)
        assert self._list_backups(state_path) == []

    def test_keep_zero_disables_rotation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AETHER_BACKUP_KEEP", "0")
        state_path = tmp_path / "state.json"
        graph = MemoryGraph(persist_path=str(state_path))
        for i in range(3):
            _add(graph, f"memory {i}")
            graph.save()
            time.sleep(0.01)
        assert self._list_backups(state_path) == []

    def test_invalid_keep_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AETHER_BACKUP_KEEP", "not-a-number")
        # Should default to 5 silently rather than crash.
        assert _backup_keep() == 5


# ==========================================================================
# Atomic write
# ==========================================================================

@needs_networkx
class TestAtomicWrite:
    def test_no_tmp_leftover_on_success(self, tmp_path):
        state_path = tmp_path / "state.json"
        graph = MemoryGraph(persist_path=str(state_path))
        _add(graph, "x")
        graph.save()
        assert state_path.exists()
        assert not (tmp_path / "state.json.tmp").exists(), (
            "atomic write left a .tmp file behind"
        )

    def test_recovers_from_stale_tmp_leftover(self, tmp_path):
        """If a previous run crashed mid-write, a .tmp file remains.
        The next save must proceed (overwriting the .tmp on the next
        atomic write, then os.replace). The substrate isn't corrupted."""
        state_path = tmp_path / "state.json"
        graph = MemoryGraph(persist_path=str(state_path))
        _add(graph, "x")
        graph.save()
        # Simulate a crashed prior write.
        (tmp_path / "state.json.tmp").write_text("garbage from a crash")
        _add(graph, "y")
        graph.save()  # Should not raise
        with state_path.open() as f:
            data = json.load(f)
        assert len(data["nodes"]) == 2
        assert not (tmp_path / "state.json.tmp").exists()

    def test_atomic_write_helper_round_trips(self, tmp_path):
        path = tmp_path / "data.json"
        _atomic_write_json(str(path), {"hello": "world", "n": 42})
        with path.open() as f:
            data = json.load(f)
        assert data == {"hello": "world", "n": 42}
        assert not path.with_suffix(".json.tmp").exists()


# ==========================================================================
# Helper unit tests
# ==========================================================================

@needs_networkx
class TestRotateBackupsHelper:
    def test_returns_none_when_state_file_missing(self, tmp_path):
        # File doesn't exist yet - no rotation to do.
        result = _rotate_backups(str(tmp_path / "missing.json"))
        assert result is None

    def test_returns_path_string_on_success(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text('{"nodes": [], "edges": []}')
        result = _rotate_backups(str(state_path))
        assert result is not None
        assert Path(result).exists()
        assert "backups" in result
