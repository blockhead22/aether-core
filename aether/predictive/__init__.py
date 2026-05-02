"""Predictive substrate primitives (v0.15+, experimental).

Adds Gaussian memory splats over slot states so the substrate can
extrapolate where beliefs are heading, not just where they currently
sit. Two collaborating pieces:

* ``splats`` — pure math. MemorySplat dataclass, Bhattacharyya overlap,
  KL divergence, evidence updates, trajectory + overlap-trend prediction.
  Ported from D:/CRT/compression_lab/memory_splats.py with no external
  dependencies beyond numpy.
* ``substrate_adapter`` — connects splats to ``aether.substrate``.
  Walks slot histories, builds one splat per slot, advances trajectory
  on each new state, and surfaces predicted contradictions across
  slot pairs.

Status: opt-in (gated by ``AETHER_PREDICTIVE=1`` in any user-facing
caller). Not wired to auto-ingest yet — first prove the prediction
signal is real on the live substrate.
"""

from .splats import (
    MemorySplat,
    create_splat,
    create_splat_from_type,
    bhattacharyya_distance,
    bhattacharyya_coefficient,
    overlap_integral,
    kl_divergence,
    cosine_similarity,
    detect_geometric_contradiction,
    update_splat_confirming,
    update_splat_contradicting,
    update_splat_with_evidence,
    predict_trajectory,
    predict_overlap_trend,
    covariance_velocity,
    GeometricContradictionResult,
)

__all__ = [
    "MemorySplat",
    "create_splat",
    "create_splat_from_type",
    "bhattacharyya_distance",
    "bhattacharyya_coefficient",
    "overlap_integral",
    "kl_divergence",
    "cosine_similarity",
    "detect_geometric_contradiction",
    "update_splat_confirming",
    "update_splat_contradicting",
    "update_splat_with_evidence",
    "predict_trajectory",
    "predict_overlap_trend",
    "covariance_velocity",
    "GeometricContradictionResult",
]
