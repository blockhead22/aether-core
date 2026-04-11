"""Tests for crt.memory.graph — memory graph and BDG."""

import json
import os
import tempfile
import time

import pytest

from crt.memory.graph import (
    MemoryType,
    BelnapState,
    EdgeType,
    Disposition,
    MemoryNode,
    ContradictionEdge,
    MemoryGraph,
    CascadeResult,
    BeliefDependencyGraph,
)

# Skip all tests if networkx not installed
nx = pytest.importorskip("networkx")


class TestMemoryGraph:
    def test_add_and_get_memory(self):
        graph = MemoryGraph()
        node = MemoryNode(
            memory_id="m1",
            text="User lives in Seattle",
            created_at=time.time(),
            memory_type="fact",
            trust=0.8,
        )
        graph.add_memory(node)
        retrieved = graph.get_memory("m1")
        assert retrieved is not None
        assert retrieved.text == "User lives in Seattle"
        assert retrieved.trust == 0.8

    def test_get_nonexistent_returns_none(self):
        graph = MemoryGraph()
        assert graph.get_memory("nonexistent") is None

    def test_add_edge(self):
        graph = MemoryGraph()
        graph.add_memory(MemoryNode("m1", "fact1", time.time()))
        graph.add_memory(MemoryNode("m2", "fact2", time.time()))
        graph.add_edge("m1", "m2", EdgeType.RELATED_TO, {"similarity": 0.85})
        stats = graph.stats()
        assert stats["edges"] == 1

    def test_add_contradiction(self):
        graph = MemoryGraph()
        graph.add_memory(MemoryNode("m1", "User is 30", time.time()))
        graph.add_memory(MemoryNode("m2", "User is 32", time.time()))
        edge = ContradictionEdge(
            disposition=Disposition.HELD.value,
            nli_score=0.9,
            detected_at=time.time(),
        )
        graph.add_contradiction("m1", "m2", edge)
        contras = graph.get_contradictions("m1")
        assert len(contras) == 1
        held = graph.get_held_contradictions()
        assert len(held) == 1

    def test_deprecate(self):
        graph = MemoryGraph()
        graph.add_memory(MemoryNode("m1", "old fact", time.time()))
        graph.add_memory(MemoryNode("m2", "new fact", time.time()))
        graph.deprecate("m1", "m2", reason="superseded")
        node = graph.get_memory("m1")
        assert node.belnap_state == BelnapState.FALSE.value
        assert node.superseded_by == "m2"

    def test_neighbors(self):
        graph = MemoryGraph()
        for i in range(4):
            graph.add_memory(MemoryNode(f"m{i}", f"fact{i}", time.time()))
        graph.add_edge("m0", "m1", EdgeType.RELATED_TO)
        graph.add_edge("m1", "m2", EdgeType.RELATED_TO)
        graph.add_edge("m2", "m3", EdgeType.RELATED_TO)
        neighbors_1hop = graph.get_neighbors("m0", hops=1)
        assert "m1" in neighbors_1hop
        neighbors_2hop = graph.get_neighbors("m0", hops=2)
        assert "m2" in neighbors_2hop

    def test_contradiction_density(self):
        graph = MemoryGraph()
        graph.add_memory(MemoryNode("m1", "a", time.time()))
        graph.add_memory(MemoryNode("m2", "b", time.time()))
        graph.add_memory(MemoryNode("m3", "c", time.time()))
        edge = ContradictionEdge(disposition=Disposition.HELD.value)
        graph.add_contradiction("m1", "m2", edge)
        graph.add_contradiction("m1", "m3", edge)
        assert graph.contradiction_density("m1") == 2
        assert graph.contradiction_density("m2") == 1

    def test_save_and_load(self):
        graph = MemoryGraph()
        graph.add_memory(MemoryNode("m1", "fact1", 1000.0, trust=0.9))
        graph.add_memory(MemoryNode("m2", "fact2", 2000.0, trust=0.7))
        graph.add_edge("m1", "m2", EdgeType.SUPPORTS)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            graph.save(path)
            loaded = MemoryGraph(persist_path=path)
            assert loaded.get_memory("m1") is not None
            assert loaded.get_memory("m2") is not None
            assert loaded.stats()["edges"] == 1
        finally:
            os.unlink(path)

    def test_stats(self):
        graph = MemoryGraph()
        graph.add_memory(MemoryNode("m1", "a", time.time()))
        graph.add_memory(MemoryNode("m2", "b", time.time()))
        stats = graph.stats()
        assert stats["nodes"] == 2
        assert "edges" in stats
        assert "belnap_states" in stats


class TestBeliefDependencyGraph:
    def _make_chain_bdg(self):
        """A -> B -> C chain."""
        from dataclasses import dataclass

        @dataclass
        class FakeLocus:
            memory_id: str
            text: str
            memory_type: str = "fact"
            alpha: float = 0.5

        bdg = BeliefDependencyGraph()
        for name in ["A", "B", "C"]:
            bdg.add_belief(FakeLocus(name, f"belief {name}"))
        bdg.add_dependency("A", "B", EdgeType.SUPPORTS, weight=0.8)
        bdg.add_dependency("B", "C", EdgeType.SUPPORTS, weight=0.6)
        return bdg

    def test_cascade_propagation(self):
        bdg = self._make_chain_bdg()
        result = bdg.propagate_cascade("A", delta_0=1.0)
        assert "A" in result.affected_nodes
        assert "B" in result.affected_nodes
        assert result.impacts["B"] < result.impacts["A"]

    def test_cascade_damping(self):
        bdg = self._make_chain_bdg()
        result = bdg.propagate_cascade("A", delta_0=1.0, lipschitz_constant=0.5)
        # With L=0.5, impact should decay rapidly
        if "C" in result.impacts:
            assert result.impacts["C"] < result.impacts["A"] * 0.5

    def test_backward_propagation(self):
        bdg = self._make_chain_bdg()
        result = bdg.propagate_backward("C", loss=1.0)
        assert "C" in result.affected_nodes
        assert "B" in result.affected_nodes

    def test_held_nodes_block_cascade(self):
        bdg = self._make_chain_bdg()
        result = bdg.propagate_cascade(
            "A", delta_0=1.0, held_nodes={"B"},
        )
        assert "B" in result.blocked_by_firewall

    def test_graph_properties(self):
        bdg = self._make_chain_bdg()
        assert bdg.num_nodes == 3
        assert bdg.num_edges == 2
        assert bdg.is_dag

    def test_cascade_result_pressure(self):
        bdg = self._make_chain_bdg()
        result = bdg.propagate_cascade("A", delta_0=1.0)
        assert isinstance(result.max_pressure, float)
        assert isinstance(result.avg_pressure, float)

    def test_effective_reachable_set(self):
        bdg = self._make_chain_bdg()
        reachable = bdg.effective_reachable_set("A", held_nodes=set())
        assert "B" in reachable
        assert "C" in reachable

    def test_reachable_with_held_blocks(self):
        bdg = self._make_chain_bdg()
        reachable = bdg.effective_reachable_set("A", held_nodes={"B"})
        # B is reachable but shouldn't propagate through
        assert "B" in reachable
        assert "C" not in reachable
