"""Topology adapter over SubstrateGraph.

Borrows the splat overlay from aether.predictive to get one Gaussian
splat per slot, then runs persistent homology over the splat centers.

Two flavors:

* ``compute_substrate_topology`` -- one-shot: current Betti numbers,
  persistence summary, interpretation.
* ``compute_substrate_topology_evolution`` -- time series: replay each
  slot's history and snapshot topology at each event timestamp, then
  surface restructuring events (Betti number changes).

Both return JSON-friendly dicts so they can drop straight into MCP
tool responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aether.predictive.splats import (
    MemorySplat,
    create_splat_from_type,
    update_splat_with_evidence,
)
from aether.predictive.substrate_adapter import SubstrateSplatOverlay
from aether.substrate import SubstrateGraph

from .persistent_homology import (
    BeliefTopology,
    compute_topology,
    track_topology_evolution,
    detect_restructuring_events,
)


def _topology_to_dict(t: BeliefTopology) -> Dict[str, Any]:
    return {
        "n_memories": t.n_memories,
        "betti_0": t.betti_0,
        "betti_1": t.betti_1,
        "betti_2": t.betti_2,
        "max_persistence_h0": t.max_persistence_h0,
        "max_persistence_h1": t.max_persistence_h1,
        "total_persistence_h0": t.total_persistence_h0,
        "total_persistence_h1": t.total_persistence_h1,
        "n_significant_h0": t.n_significant_h0,
        "n_significant_h1": t.n_significant_h1,
        "distance_metric": t.distance_metric,
        "betti_filtration": t.betti_filtration,
        "interpretation": t.interpretation,
    }


def compute_substrate_topology(
    substrate: SubstrateGraph,
    *,
    distance_fn: str = "cosine",
    namespace: Optional[str] = None,
    betti_filtration: float = 0.35,
    wait_for_encoder: bool = False,
) -> Dict[str, Any]:
    """Run persistent homology over current substrate slot-splats.

    namespace: if given, restrict to slots in that namespace
    wait_for_encoder: True for offline scripts; False for MCP (non-blocking)
    """
    overlay = SubstrateSplatOverlay(substrate)
    splats_map = overlay.build(wait_for_encoder=wait_for_encoder)
    if not splats_map:
        return {
            "encoder_loaded": overlay.encoder.is_loaded,
            "encoder_warming": overlay.encoder.is_warming,
            "splats_built": 0,
            "topology": None,
        }

    if namespace:
        splats = [s for sid, s in splats_map.items() if sid.startswith(f"{namespace}:")]
    else:
        splats = list(splats_map.values())

    topo = compute_topology(splats, distance_fn=distance_fn,
                            betti_filtration=betti_filtration)
    return {
        "encoder_loaded": True,
        "splats_built": len(splats_map),
        "splats_used": len(splats),
        "namespace": namespace,
        "topology": _topology_to_dict(topo),
    }


def compute_substrate_topology_evolution(
    substrate: SubstrateGraph,
    *,
    distance_fn: str = "cosine",
    namespace: Optional[str] = None,
    betti_filtration: float = 0.35,
    max_snapshots: int = 50,
) -> Dict[str, Any]:
    """Walk every observation in time order; snapshot topology at each step.

    Returns Betti trajectories + restructuring events (Betti changes).

    For substrates whose data was bulk-migrated (many states sharing a
    timestamp), evolution collapses to a single snapshot — caller should
    interpret the result accordingly.
    """
    overlay = SubstrateSplatOverlay(substrate)
    if not overlay.ensure_encoder(wait=False):
        return {
            "encoder_loaded": False,
            "encoder_warming": overlay.encoder.is_warming,
            "snapshots": [],
            "events": [],
        }

    # Order observations chronologically across the whole graph.
    obs_order: List[Tuple[float, str]] = []  # (observed_at, observation_id)
    for obs in substrate.observations.values():
        obs_order.append((obs.observed_at, obs.observation_id))
    obs_order.sort()

    if not obs_order:
        return {
            "encoder_loaded": True,
            "snapshots": [],
            "events": [],
        }

    # Replay observations into per-slot splats and snapshot topology after each.
    per_slot: Dict[str, MemorySplat] = {}
    timeline: List[Tuple[float, List[MemorySplat]]] = []
    seen_obs = 0

    for ts, observation_id in obs_order:
        observation = substrate.observations.get(observation_id)
        if observation is None:
            continue
        for state_id in observation.emitted_state_ids:
            state = substrate.states.get(state_id)
            if state is None:
                continue
            if namespace and not state.slot_id.startswith(f"{namespace}:"):
                continue
            vec = overlay._embed(state.value)
            if vec is None:
                continue
            existing = per_slot.get(state.slot_id)
            if existing is None:
                splat = create_splat_from_type(
                    memory_id=state.slot_id,
                    embedding=vec,
                    text=state.value,
                    memory_type="belief",
                    confidence=state.trust,
                )
                if splat.trajectory:
                    splat.trajectory[-1]["timestamp"] = state.observed_at
                per_slot[state.slot_id] = splat
            else:
                update_splat_with_evidence(existing, vec, weight=0.3)
                existing.snapshot()
                existing.trajectory[-1]["timestamp"] = state.observed_at
        seen_obs += 1
        timeline.append((ts, list(per_slot.values())))

    if max_snapshots and len(timeline) > max_snapshots:
        # Subsample evenly to keep ripser calls bounded.
        step = max(1, len(timeline) // max_snapshots)
        timeline = timeline[::step]

    snapshots = track_topology_evolution(
        timeline,
        distance_fn=distance_fn,
        betti_filtration=betti_filtration,
    )
    events = detect_restructuring_events(snapshots)

    return {
        "encoder_loaded": True,
        "observations_total": seen_obs,
        "snapshots": [
            {
                "timestamp": s.timestamp,
                "betti_0": s.betti_0,
                "betti_1": s.betti_1,
                "max_persistence_h0": s.max_persistence_h0,
                "max_persistence_h1": s.max_persistence_h1,
                "total_persistence_h1": s.total_persistence_h1,
                "n_memories": s.n_memories,
            }
            for s in snapshots
        ],
        "events": events,
    }
