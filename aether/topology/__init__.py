"""Belief topology via persistent homology (v0.15+, experimental).

Treats slot embedding centers as a point cloud and asks topological
questions about it:

* Beta_0 (connected components) -- how fragmented is the belief space?
* Beta_1 (1-dimensional holes) -- avoidance loops, circular reasoning.
* Persistence -- how "real" is each feature (long persistence = structural).

Distance metric is cosine in 384D (the curse-of-dimensionality finding
from the predictive splat work: Bhattacharyya overlap degenerates,
cosine carries the structural signal).

Ported and trimmed from D:/CRT/compression_lab/belief_topology.py.
Plotting and demo code dropped; pure analysis only.
"""

from .persistent_homology import (
    BeliefTopology,
    TopologicalFeature,
    TopologySnapshot,
    compute_topology,
    interpret_topology,
    cosine_distance_matrix,
    confidence_weighted_distance,
    track_topology_evolution,
    detect_restructuring_events,
)

__all__ = [
    "BeliefTopology",
    "TopologicalFeature",
    "TopologySnapshot",
    "compute_topology",
    "interpret_topology",
    "cosine_distance_matrix",
    "confidence_weighted_distance",
    "track_topology_evolution",
    "detect_restructuring_events",
]
