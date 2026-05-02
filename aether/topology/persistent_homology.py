"""Persistent homology over memory-splat point clouds.

Trimmed from D:/CRT/compression_lab/belief_topology.py:

* Distance matrix builders (cosine + confidence-weighted variants)
* compute_topology() -> BeliefTopology with Betti numbers + persistence
* interpret_topology() -> human-readable summary
* track_topology_evolution + detect_restructuring_events for time series

Requires the [topology] extra (ripser + persim). All imports are
deferred to call-time so importing this module never fails on a
machine without the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aether.predictive.splats import MemorySplat, cosine_similarity


# ---------------------------------------------------------------------------
# Distance matrices
# ---------------------------------------------------------------------------


def cosine_distance_matrix(splats: List[MemorySplat]) -> np.ndarray:
    """Pairwise cosine distance (1 - cos) between splat centers."""
    n = len(splats)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - cosine_similarity(splats[i], splats[j])
            D[i, j] = d
            D[j, i] = d
    return D


def confidence_weighted_distance(splats: List[MemorySplat]) -> np.ndarray:
    """Cosine distance / (alpha_a * alpha_b). Low-confidence beliefs are pushed
    to the periphery so they contribute less structure."""
    D = cosine_distance_matrix(splats)
    n = len(splats)
    for i in range(n):
        for j in range(i + 1, n):
            w = splats[i].alpha * splats[j].alpha
            if w > 1e-8:
                D[i, j] /= w
                D[j, i] /= w
            else:
                D[i, j] = 2.0
                D[j, i] = 2.0
    return D


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TopologicalFeature:
    """One (birth, death) feature from persistent homology."""
    dimension: int
    birth: float
    death: float
    persistence: float


@dataclass
class BeliefTopology:
    """Full topology of a slot-splat point cloud at one moment."""
    n_memories: int
    features: List[TopologicalFeature]
    betti_0: int
    betti_1: int
    betti_2: int
    max_persistence_h0: float
    max_persistence_h1: float
    total_persistence_h0: float
    total_persistence_h1: float
    n_significant_h0: int
    n_significant_h1: int
    interpretation: str
    distance_metric: str = "cosine"
    betti_filtration: float = 0.35


@dataclass
class TopologySnapshot:
    """Topology at one point in time (for evolution tracking)."""
    timestamp: float
    betti_0: int
    betti_1: int
    max_persistence_h0: float
    max_persistence_h1: float
    total_persistence_h1: float
    n_memories: int


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_topology(
    splats: List[MemorySplat],
    distance_fn: str = "cosine",
    max_dim: int = 1,
    significance_threshold: float = 0.05,
    betti_filtration: float = 0.35,
) -> BeliefTopology:
    """Compute persistent homology of a splat point cloud.

    distance_fn: 'cosine' (default) or 'confidence_weighted'
    max_dim: 1 = components + loops (2 = adds voids; rarely useful)
    significance_threshold: persistence below this is considered noise
    betti_filtration: distance scale at which to read off Betti numbers
    """
    if len(splats) < 3:
        return BeliefTopology(
            n_memories=len(splats), features=[], betti_0=len(splats),
            betti_1=0, betti_2=0,
            max_persistence_h0=0.0, max_persistence_h1=0.0,
            total_persistence_h0=0.0, total_persistence_h1=0.0,
            n_significant_h0=0, n_significant_h1=0,
            interpretation="Too few memories for topology (need >= 3).",
            distance_metric=distance_fn,
            betti_filtration=betti_filtration,
        )

    if distance_fn == "confidence_weighted":
        D = confidence_weighted_distance(splats)
    else:
        D = cosine_distance_matrix(splats)

    from ripser import ripser
    diagrams = ripser(D, maxdim=max_dim, distance_matrix=True)["dgms"]

    features: List[TopologicalFeature] = []
    for dim, dgm in enumerate(diagrams):
        for birth, death in dgm:
            persistence = float("inf") if np.isinf(death) else float(death - birth)
            features.append(TopologicalFeature(
                dimension=dim,
                birth=float(birth),
                death=float(death),
                persistence=persistence,
            ))

    betti = [0] * (max_dim + 1)
    for f in features:
        if f.birth <= betti_filtration and (
            f.death > betti_filtration or np.isinf(f.persistence)
        ):
            betti[f.dimension] += 1

    h0_persist = [f.persistence for f in features
                  if f.dimension == 0 and not np.isinf(f.persistence)]
    h1_persist = [f.persistence for f in features
                  if f.dimension == 1 and not np.isinf(f.persistence)]

    max_p_h0 = max(h0_persist) if h0_persist else 0.0
    max_p_h1 = max(h1_persist) if h1_persist else 0.0
    total_p_h0 = sum(h0_persist)
    total_p_h1 = sum(h1_persist)
    n_sig_h0 = sum(1 for p in h0_persist if p > significance_threshold)
    n_sig_h1 = sum(1 for p in h1_persist if p > significance_threshold)

    interp = interpret_topology(
        betti[0], betti[1], betti[2] if len(betti) > 2 else 0,
        n_sig_h0, n_sig_h1, max_p_h0, max_p_h1, len(splats),
    )

    return BeliefTopology(
        n_memories=len(splats),
        features=features,
        betti_0=betti[0],
        betti_1=betti[1],
        betti_2=betti[2] if len(betti) > 2 else 0,
        max_persistence_h0=max_p_h0,
        max_persistence_h1=max_p_h1,
        total_persistence_h0=total_p_h0,
        total_persistence_h1=total_p_h1,
        n_significant_h0=n_sig_h0,
        n_significant_h1=n_sig_h1,
        interpretation=interp,
        distance_metric=distance_fn,
        betti_filtration=betti_filtration,
    )


def interpret_topology(
    b0: int,
    b1: int,
    b2: int,
    n_sig_h0: int,
    n_sig_h1: int,
    max_p_h0: float,
    max_p_h1: float,
    n_total: int,
) -> str:
    """Human-readable summary. The mappings (cluster->compartmentalization,
    hole->avoidance) are interpretive and need empirical validation."""
    parts: List[str] = []
    if b0 == 1:
        parts.append("Beliefs form a single connected cluster (integrated worldview).")
    elif b0 <= 3:
        parts.append(
            f"Beliefs split into {b0} distinct clusters (compartmentalized thinking)."
        )
    else:
        parts.append(
            f"Beliefs fragmented into {b0} clusters (highly compartmentalized "
            "or diverse interests)."
        )

    if n_sig_h0 > 0 and max_p_h0 > 0.3:
        parts.append(
            f"Strong cluster separation (max gap persistence={max_p_h0:.2f}) -- "
            "some belief domains are genuinely isolated from each other."
        )

    if b1 == 0:
        parts.append("No topological holes (beliefs fill their space without gaps).")
    elif b1 <= 2:
        parts.append(
            f"{b1} hole(s) in belief space -- topic(s) circled around but not "
            "directly addressed."
        )
    else:
        parts.append(
            f"{b1} holes -- multiple avoidance patterns or circular reasoning structures."
        )

    if n_sig_h1 > 0 and max_p_h1 > 0.2:
        parts.append(
            f"Persistent hole (persistence={max_p_h1:.2f}) -- stable gap, not noise. "
            "Something consistently avoided or unresolved."
        )

    if n_total > 0:
        ratio = b0 / n_total
        if ratio < 0.1:
            parts.append("High belief density (memories well-connected).")
        elif ratio > 0.5:
            parts.append("Low belief density (many isolated memories).")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Temporal: how does topology evolve?
# ---------------------------------------------------------------------------


def track_topology_evolution(
    splat_timeline: List[Tuple[float, List[MemorySplat]]],
    **kwargs,
) -> List[TopologySnapshot]:
    """Compute one TopologySnapshot per (timestamp, splats) pair."""
    snapshots: List[TopologySnapshot] = []
    for ts, splats in splat_timeline:
        topo = compute_topology(splats, **kwargs)
        snapshots.append(TopologySnapshot(
            timestamp=ts,
            betti_0=topo.betti_0,
            betti_1=topo.betti_1,
            max_persistence_h0=topo.max_persistence_h0,
            max_persistence_h1=topo.max_persistence_h1,
            total_persistence_h1=topo.total_persistence_h1,
            n_memories=topo.n_memories,
        ))
    return snapshots


def detect_restructuring_events(
    snapshots: List[TopologySnapshot],
) -> List[Dict[str, Any]]:
    """Find Betti-number changes between consecutive snapshots.

    delta_b0 < 0: clusters merging (beliefs integrating)
    delta_b0 > 0: clusters splitting (beliefs fragmenting)
    delta_b1 > 0: new holes (avoidance patterns appearing)
    delta_b1 < 0: holes filling (gaps resolved)
    """
    events: List[Dict[str, Any]] = []
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i - 1], snapshots[i]
        db0 = curr.betti_0 - prev.betti_0
        db1 = curr.betti_1 - prev.betti_1
        if db0 == 0 and db1 == 0:
            continue
        desc: List[str] = []
        if db0 < 0:
            desc.append(f"{abs(db0)} cluster(s) merged (beliefs integrating)")
        elif db0 > 0:
            desc.append(f"{db0} new cluster(s) emerged (beliefs fragmenting)")
        if db1 > 0:
            desc.append(f"{db1} new hole(s) opened (avoidance pattern)")
        elif db1 < 0:
            desc.append(f"{abs(db1)} hole(s) closed (gap filled)")
        events.append({
            "timestamp": curr.timestamp,
            "step": i,
            "delta_b0": db0,
            "delta_b1": db1,
            "description": "; ".join(desc),
        })
    return events
