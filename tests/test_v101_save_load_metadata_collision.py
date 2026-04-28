"""Regression tests for v0.10.1: edge metadata key collision in save/load.

Background -- the v0.10.0 production miss (caught by substrate-assisted
dev loop on first call to aether_path during v0.10.1 design):

    aether_path crashed with:
      MemoryNode.__init__() missing 3 required positional arguments:
      'memory_id', 'text', 'created_at'

    Root cause: a save/load asymmetry. In aether/memory/graph.py:save(),
    edges were serialized as:

        data["edges"].append({
            "source": source_id,    # endpoint, set first
            "target": target_id,
            **clean_data,           # <- shadow-overrides "source" if metadata had it
        })

    backfill_edges in v0.9.5 included `"source": "backfill"` in its
    auto-link metadata. The **clean_data spread overrode the endpoint
    "source" key. JSON wrote source="backfill" instead of the real
    node_id. On load(), networkx.add_edge("backfill", real_target, ...)
    auto-created a stub node with id="backfill" and no required fields.
    Subsequent get_memory("backfill") crashed.

    Tests didn't catch it because every persistence test built edges
    via the public API, and no test round-tripped an edge with
    metadata containing keys that shadow the JSON edge schema.

Fix (three-part defense):
    1. graph.py:save() puts endpoints AFTER **clean_data so endpoints win
    2. graph.py:load() skips edges with unknown endpoints (defense in depth)
    3. state.py:backfill_edges renames metadata key 'source' -> 'origin'
       to avoid collision

This file proves all three:
    - Edge metadata containing key "source" round-trips with correct
      endpoints preserved
    - Edge metadata containing key "target" round-trips correctly
    - load() skips edges whose endpoints aren't valid nodes (forward
      compatibility for pre-v0.10.1 substrates with corrupted edges)
    - get_memory() works on every node after a round-trip (no stub nodes)
    - aether_path can walk the BDG without crashing on a substrate
      that had backfill_edges run on it
"""

from __future__ import annotations

import os
import time

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx required (install [graph] extra)",
)

from aether.memory.graph import MemoryGraph, MemoryNode, EdgeType


@needs_networkx
class TestEdgeMetadataKeyCollision:
    """The actual v0.10.1 bug: metadata keys shadowing endpoint keys."""

    def test_edge_metadata_with_source_key_round_trips(self, tmp_path):
        """Edge metadata `{"source": "backfill"}` must NOT overwrite the
        endpoint source field on save.

        v0.10.1 contract: endpoints are protected. Metadata fields whose
        keys collide with the JSON edge schema's reserved keys
        ('source', 'target') are silently dropped on save -- they cannot
        round-trip without breaking the schema. Callers should not use
        these key names in metadata. Use 'origin' / 'destination' / etc.
        instead. This is why v0.10.1 also renamed backfill_edges'
        metadata key from 'source' to 'origin'.
        """
        path = str(tmp_path / "state.json")
        g = MemoryGraph(persist_path=path)
        g.add_memory(MemoryNode("a", "node a", time.time()))
        g.add_memory(MemoryNode("b", "node b", time.time()))
        g.add_edge("a", "b", EdgeType.RELATED_TO,
                   metadata={"source": "backfill", "auto": True})
        g.save(path)

        g2 = MemoryGraph(persist_path=path)
        edges = list(g2.graph.edges(data=True))
        assert len(edges) == 1
        src, tgt, data = edges[0]
        # ENDPOINT preserved -- this is the bug fix
        assert src == "a", f"endpoint source was overwritten: {src}"
        assert tgt == "b"
        # Other metadata fields survive
        assert data.get("auto") is True
        # The colliding "source" metadata field was silently dropped --
        # documented contract, not a bug. Callers must rename their key.
        assert data.get("source") != "backfill", (
            "metadata 'source' key would re-shadow the endpoint on next "
            "load and re-corrupt the substrate. Callers must rename."
        )

    def test_edge_metadata_with_target_key_round_trips(self, tmp_path):
        """Same defense for the target endpoint."""
        path = str(tmp_path / "state.json")
        g = MemoryGraph(persist_path=path)
        g.add_memory(MemoryNode("a", "node a", time.time()))
        g.add_memory(MemoryNode("b", "node b", time.time()))
        g.add_edge("a", "b", EdgeType.RELATED_TO,
                   metadata={"target": "evil_value", "auto": True})
        g.save(path)

        g2 = MemoryGraph(persist_path=path)
        edges = list(g2.graph.edges(data=True))
        src, tgt, data = edges[0]
        assert src == "a"
        assert tgt == "b", f"endpoint target was overwritten: {tgt}"

    def test_round_trip_does_not_create_stub_nodes(self, tmp_path):
        """The smoking gun: after a round-trip with collision-prone
        metadata, no stub nodes should appear."""
        path = str(tmp_path / "state.json")
        g = MemoryGraph(persist_path=path)
        g.add_memory(MemoryNode("a", "node a", time.time()))
        g.add_memory(MemoryNode("b", "node b", time.time()))
        g.add_edge("a", "b", EdgeType.RELATED_TO,
                   metadata={"source": "backfill"})
        g.save(path)

        g2 = MemoryGraph(persist_path=path)
        # Only the original two nodes -- no "backfill" stub
        assert set(g2.graph.nodes()) == {"a", "b"}

    def test_get_memory_works_on_all_nodes_after_collision_round_trip(
        self, tmp_path,
    ):
        """If the bug existed, get_memory("backfill") would crash with
        the canonical AttributeError. This is the exact production
        repro from the v0.10.1 incident."""
        path = str(tmp_path / "state.json")
        g = MemoryGraph(persist_path=path)
        g.add_memory(MemoryNode("a", "node a", time.time()))
        g.add_memory(MemoryNode("b", "node b", time.time()))
        g.add_edge("a", "b", EdgeType.RELATED_TO,
                   metadata={"source": "backfill"})
        g.save(path)

        g2 = MemoryGraph(persist_path=path)
        # Every node in the graph must have constructible MemoryNode data
        for nid in g2.graph.nodes():
            node = g2.get_memory(nid)
            assert node is not None
            assert node.memory_id == nid
            assert node.text  # non-empty


@needs_networkx
class TestLoadDefenseInDepth:
    """load() must skip edges with unknown endpoints, even if a
    pre-v0.10.1 corrupted JSON file is loaded by a v0.10.1+ runtime."""

    def test_load_skips_edge_with_unknown_source(self, tmp_path):
        """Simulate a pre-fix corrupted state file: edge with source=
        a string that isn't a real node id."""
        import json
        path = str(tmp_path / "state.json")
        corrupt = {
            "nodes": [
                {"id": "a", "memory_id": "a", "text": "node a",
                 "created_at": time.time()},
                {"id": "b", "memory_id": "b", "text": "node b",
                 "created_at": time.time()},
            ],
            "edges": [
                # Corrupt: source isn't in the nodes list
                {"source": "phantom", "target": "b",
                 "edge_type": "related_to", "created_at": time.time()},
                # Valid: should still load
                {"source": "a", "target": "b",
                 "edge_type": "related_to", "created_at": time.time()},
            ],
        }
        with open(path, "w") as f:
            json.dump(corrupt, f)

        g = MemoryGraph(persist_path=path)
        # Phantom is not a node
        assert "phantom" not in g.graph
        # Only the valid edge survived
        assert len(g.graph.edges()) == 1
        src, tgt = list(g.graph.edges())[0]
        assert src == "a" and tgt == "b"

    def test_load_skips_edge_with_unknown_target(self, tmp_path):
        import json
        path = str(tmp_path / "state.json")
        corrupt = {
            "nodes": [
                {"id": "a", "memory_id": "a", "text": "x",
                 "created_at": time.time()},
            ],
            "edges": [
                {"source": "a", "target": "phantom",
                 "edge_type": "related_to", "created_at": time.time()},
            ],
        }
        with open(path, "w") as f:
            json.dump(corrupt, f)

        g = MemoryGraph(persist_path=path)
        assert "phantom" not in g.graph
        assert len(g.graph.edges()) == 0


@needs_networkx
class TestBackfillRenamedKey:
    """Verify backfill_edges no longer uses 'source' as a metadata key."""

    def test_backfill_metadata_uses_origin_not_source(self, tmp_path):
        """When backfill_edges adds RELATED_TO edges, the metadata should
        use 'origin' (not 'source') as the key naming the source of the
        edge wiring."""
        pytest.importorskip("mcp")
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"))
        if s._encoder is not None:
            s._encoder._load()
        # Two clearly-related memories, no auto-link (disable detection)
        s.add_memory("the aether substrate stores belief state across sessions",
                     trust=0.9, detect_contradictions=False)
        s.add_memory("the aether substrate persists belief state between sessions",
                     trust=0.9, detect_contradictions=False)

        result = s.backfill_edges()
        if result["added"] == 0:
            pytest.skip("substring similarity below threshold; can't verify metadata")

        # Find the RELATED_TO edge and inspect its metadata
        related_edges = [
            (u, v, d) for u, v, d in s.graph.graph.edges(data=True)
            if d.get("edge_type") == "related_to"
        ]
        assert len(related_edges) >= 1
        u, v, d = related_edges[0]
        # Metadata must use 'origin', not 'source'
        assert d.get("origin") == "backfill"
        # 'source' key in metadata would collide with the JSON schema --
        # must not be present
        assert "source" not in d


@needs_networkx
class TestAetherPathOnRealisticSubstrate:
    """End-to-end: a substrate with backfilled edges round-trips and
    aether_path runs without crashing."""

    def test_compute_path_after_save_load_with_backfill(self, tmp_path):
        pytest.importorskip("mcp")
        from aether.mcp.state import StateStore

        path = str(tmp_path / "state.json")
        s = StateStore(state_path=path)
        if s._encoder is not None:
            s._encoder._load()

        # Build a small substrate with a backfilled edge
        s.add_memory("Aether is a belief substrate library",
                     trust=0.9, detect_contradictions=False)
        s.add_memory("Aether is a belief substrate for AI agents",
                     trust=0.85, detect_contradictions=False)
        s.backfill_edges()

        # Restart the store -- forces a save/load round-trip
        s2 = StateStore(state_path=path)

        # No phantom nodes
        for nid in s2.graph.graph.nodes():
            node = s2.graph.get_memory(nid)
            assert node is not None, f"phantom node detected: {nid}"

        # aether_path runs without crashing (the v0.10.0 production miss)
        result = s2.compute_path("Aether substrate")
        assert result["method"] in ("dijkstra", "no_target", "no_substrate")
        # Graph walk completed without raising
