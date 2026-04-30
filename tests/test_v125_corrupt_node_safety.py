"""F#3 fix: substrate read tools must not crash on corrupt nodes.

Background:
    Pre-v0.10.1 substrates can carry stub nodes with missing required
    fields (memory_id, text, created_at) — the metadata-collision bug
    created the canonical example, an `id="backfill"` node with no
    other fields. MemoryNode.from_dict() raises TypeError on those.

    Before this fix, `aether_memory_detail`, `aether_lineage`, and
    `aether_cascade_preview` all crashed when fed the corrupt node id.
    The substrate-assisted dev loop tried to introspect such a node
    on 2026-04-29 and got TypeErrors instead of useful diagnostics.

The fix:
    `MemoryGraph.get_memory()` catches deserialization errors and
    returns None — same shape as "node not found." Every downstream
    caller already handles None, so the runtime degrades gracefully
    to "unknown memory_id" instead of crashing.

    `aether doctor` surfaces these corrupt nodes proactively in its
    state_file check; this safety net guarantees they never crash a
    tool even if doctor wasn't run.
"""

from __future__ import annotations

import json

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx required (install [graph] extra)",
)


def _write_corrupt_state(path, corrupt_id="backfill") -> dict:
    """Construct a JSON state file with one valid node + one corrupt
    stub. Mirrors the pre-v0.10.1 metadata-collision shape."""
    data = {
        "aether_version": "test",
        "nodes": [
            {
                "id": "good_node",
                "memory_id": "good_node",
                "text": "valid memory",
                "created_at": 1.0,
                "memory_type": "fact",
                "belnap_state": "T",
                "trust": 0.7,
                "confidence": 0.7,
                "valid_at": 1.0,
                "invalid_at": None,
                "superseded_by": None,
                "tags": [],
            },
            # Corrupt: id only, missing memory_id / text / created_at.
            {"id": corrupt_id},
        ],
        "edges": [],
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return data


@needs_networkx
class TestCorruptNodeSafety:
    """Substrate read tools degrade to 'not found' on corrupt nodes."""

    def test_get_memory_returns_none_for_corrupt_node(self, tmp_path):
        """The single source of safety: MemoryGraph.get_memory() catches
        the TypeError that MemoryNode.from_dict raises."""
        from aether.memory.graph import MemoryGraph

        state = tmp_path / "state.json"
        _write_corrupt_state(state)
        g = MemoryGraph(persist_path=str(state))

        assert g.get_memory("good_node") is not None
        assert g.get_memory("backfill") is None  # was TypeError pre-fix
        assert g.get_memory("totally_missing") is None  # baseline

    def test_memory_detail_does_not_crash_on_corrupt_node(self, tmp_path):
        """`aether_memory_detail` returns a graceful error envelope."""
        from aether.mcp.state import StateStore

        state = tmp_path / "state.json"
        _write_corrupt_state(state)
        s = StateStore(state_path=str(state))

        result = s.memory_detail("backfill")
        assert "error" in result, (
            f"memory_detail should report 'unknown memory_id' for corrupt "
            f"nodes, got {result}"
        )

    def test_lineage_skips_corrupt_node(self, tmp_path):
        """`aether_lineage` skips corrupt ancestors instead of crashing."""
        from aether.mcp.state import StateStore

        state = tmp_path / "state.json"
        _write_corrupt_state(state)
        s = StateStore(state_path=str(state))

        # Lineage on the corrupt node returns an empty/error result
        # rather than crashing.
        result = s.lineage("backfill", hops=3)
        assert isinstance(result, dict)
        # No assertion on shape — just that we got back a dict, not an
        # exception. The contract for lineage on a missing/corrupt node
        # is "no ancestors."

    def test_cascade_preview_does_not_crash_on_corrupt_node(self, tmp_path):
        """`aether_cascade_preview` also returns a graceful response."""
        from aether.mcp.state import StateStore

        state = tmp_path / "state.json"
        _write_corrupt_state(state)
        s = StateStore(state_path=str(state))

        result = s.cascade_preview("backfill", proposed_delta=-1.0)
        assert isinstance(result, dict)

    def test_compute_path_with_corrupt_ancestor_does_not_crash(self, tmp_path):
        """compute_path's sort key reads node.trust — must handle None."""
        from aether.mcp.state import StateStore

        state = tmp_path / "state.json"
        # Build state with a real target + corrupt connected via edge so
        # compute_path traverses both.
        data = _write_corrupt_state(state)
        # Manually add an edge so the corrupt node enters the Dijkstra
        # frontier from the good node's neighborhood.
        data["edges"].append({
            "edge_type": "supports",
            "source": "good_node",
            "target": "backfill",
        })
        with open(state, "w") as f:
            json.dump(data, f)

        s = StateStore(state_path=str(state))
        # Reload to pick up the edge.
        s._sync_from_disk_if_stale()

        # compute_path queries by text, lands on good_node, walks the
        # graph. Must not crash even if the BDG contains a corrupt
        # neighbor.
        result = s.compute_path(query="valid memory", max_tokens=500)
        assert isinstance(result, dict)
