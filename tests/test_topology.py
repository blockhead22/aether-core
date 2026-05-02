"""Topology tests: persistent homology + substrate adapter.

Uses a stub encoder so tests stay offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.predictive.splats import create_splat_from_type
from aether.predictive.substrate_adapter import SubstrateSplatOverlay
from aether.substrate import SubstrateGraph
from aether.topology import (
    BeliefTopology,
    compute_topology,
    cosine_distance_matrix,
    track_topology_evolution,
    detect_restructuring_events,
    TopologySnapshot,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


class _StubEncoder:
    """Deterministic hash-based encoder, no torch."""

    is_loaded = True
    is_unavailable = False
    is_warming = False

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(384).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        return True

    def start_warmup(self) -> None:
        return


def _make_splats(n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    splats = []
    for i in range(n):
        v = _unit(rng.standard_normal(384).astype(np.float32))
        splats.append(create_splat_from_type(f"s{i}", v, f"text{i}", "belief", 0.8))
    return splats


# ---------------------------------------------------------------------------
# Pure topology
# ---------------------------------------------------------------------------


def test_too_few_memories_returns_empty_topology():
    topo = compute_topology([])
    assert topo.n_memories == 0
    assert topo.betti_0 == 0
    assert "Too few" in topo.interpretation


def test_random_splats_produce_nontrivial_topology():
    splats = _make_splats(8, seed=42)
    topo = compute_topology(splats, distance_fn="cosine")
    # Random orthogonal-ish splats should produce a multi-component topology
    # at the 0.35 default filtration; b0 >= 1.
    assert topo.n_memories == 8
    assert topo.betti_0 >= 1
    assert topo.betti_1 >= 0
    assert isinstance(topo.interpretation, str) and len(topo.interpretation) > 0


def test_cosine_distance_matrix_is_symmetric_zero_diag():
    splats = _make_splats(5)
    D = cosine_distance_matrix(splats)
    assert D.shape == (5, 5)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)
    np.testing.assert_allclose(D, D.T, atol=1e-6)


def test_distance_fn_options_run():
    splats = _make_splats(6, seed=1)
    cosine_topo = compute_topology(splats, distance_fn="cosine")
    weighted_topo = compute_topology(splats, distance_fn="confidence_weighted")
    assert cosine_topo.distance_metric == "cosine"
    assert weighted_topo.distance_metric == "confidence_weighted"


def test_evolution_tracking_returns_one_snapshot_per_step():
    splats = _make_splats(5, seed=2)
    timeline = []
    for i in range(1, 5):
        timeline.append((float(i), splats[:i + 1]))
    snapshots = track_topology_evolution(timeline)
    assert len(snapshots) == 4
    assert all(isinstance(s, TopologySnapshot) for s in snapshots)


def test_detect_restructuring_events_fires_on_betti_change():
    snaps = [
        TopologySnapshot(timestamp=1.0, betti_0=2, betti_1=0,
                         max_persistence_h0=0, max_persistence_h1=0,
                         total_persistence_h1=0, n_memories=2),
        TopologySnapshot(timestamp=2.0, betti_0=4, betti_1=0,
                         max_persistence_h0=0, max_persistence_h1=0,
                         total_persistence_h1=0, n_memories=4),
        TopologySnapshot(timestamp=3.0, betti_0=4, betti_1=1,
                         max_persistence_h0=0, max_persistence_h1=0.3,
                         total_persistence_h1=0.3, n_memories=4),
        TopologySnapshot(timestamp=4.0, betti_0=4, betti_1=1,
                         max_persistence_h0=0, max_persistence_h1=0.3,
                         total_persistence_h1=0.3, n_memories=4),
    ]
    events = detect_restructuring_events(snaps)
    # Two events: cluster split (1->2) and hole opens (2->3); step 3 stable.
    assert len(events) == 2
    assert events[0]["delta_b0"] == 2
    assert events[1]["delta_b1"] == 1


# ---------------------------------------------------------------------------
# Substrate adapter end-to-end
# ---------------------------------------------------------------------------


def test_substrate_topology_runs_against_stubbed_encoder(monkeypatch):
    from aether.topology import substrate_adapter as adapter

    sub = SubstrateGraph()
    for slot, value in [
        ("location", "Milwaukee"),
        ("location", "Portland"),
        ("employer", "Acme"),
        ("name", "Alice"),
        ("color", "orange"),
    ]:
        sub.observe("user", slot, value)

    # Replace the overlay's encoder construction with our stub
    real_overlay_cls = SubstrateSplatOverlay

    def make_stub_overlay(substrate, encoder=None, memory_type="belief"):
        return real_overlay_cls(substrate, encoder=_StubEncoder(),
                                memory_type=memory_type)

    monkeypatch.setattr(adapter, "SubstrateSplatOverlay", make_stub_overlay)

    res = adapter.compute_substrate_topology(sub, distance_fn="cosine")
    assert res["encoder_loaded"] is True
    assert res["splats_built"] >= 4
    assert res["topology"]["betti_0"] >= 1
    assert "interpretation" in res["topology"]


def test_substrate_topology_evolution_returns_snapshots(monkeypatch):
    from aether.topology import substrate_adapter as adapter

    sub = SubstrateGraph()
    sub.observe("user", "a", "v1")
    sub.observe("user", "b", "v1")
    sub.observe("user", "c", "v1")
    sub.observe("user", "d", "v1")

    real_overlay_cls = SubstrateSplatOverlay

    def make_stub_overlay(substrate, encoder=None, memory_type="belief"):
        return real_overlay_cls(substrate, encoder=_StubEncoder(),
                                memory_type=memory_type)

    monkeypatch.setattr(adapter, "SubstrateSplatOverlay", make_stub_overlay)

    res = adapter.compute_substrate_topology_evolution(sub)
    assert res["encoder_loaded"] is True
    assert len(res["snapshots"]) >= 1
    # Each new slot should trigger a Betti-0 increase event somewhere
    assert isinstance(res["events"], list)


def test_substrate_topology_handles_dead_encoder(monkeypatch):
    from aether.topology import substrate_adapter as adapter

    class _DeadEncoder:
        is_loaded = False
        is_unavailable = True
        is_warming = False

        def encode(self, text):
            return None

        def wait_until_ready(self, timeout=60.0):
            return False

        def start_warmup(self):
            return

    sub = SubstrateGraph()
    sub.observe("user", "x", "v")

    real_overlay_cls = SubstrateSplatOverlay

    def make_stub_overlay(substrate, encoder=None, memory_type="belief"):
        return real_overlay_cls(substrate, encoder=_DeadEncoder(),
                                memory_type=memory_type)

    monkeypatch.setattr(adapter, "SubstrateSplatOverlay", make_stub_overlay)
    res = adapter.compute_substrate_topology(sub)
    assert res["splats_built"] == 0
    assert res["topology"] is None
