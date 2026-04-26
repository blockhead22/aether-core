"""Belief Backpropagation Engine -- Error-Driven Trust Evolution.

Core components:
  - EpistemicLoss: computes loss from correction events
  - BackpropResult: structured result of a backward pass
  - DomainVolatility: tracks per-domain correction frequency
  - compute_backward_gradients(): gradient per upstream edge
  - apply_trust_adjustments(): apply corrections to trust state
  - flat_demotion(): simple multiplicative demotion

No external dependencies -- pure stdlib math.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CorrectionEvent:
    """A user correction or contradiction resolution."""
    corrected_node_id: str
    trust_at_assertion: float        # how confident was the system?
    times_corrected: int             # how many times has this node been wrong?
    correction_source: str           # "user" | "system" | "inference"
    time_since_assertion: float      # seconds since the wrong assertion
    domain: str                      # e.g. "employer", "color", "name"


@dataclass
class BackpropResult:
    """Result of a backward pass through the BDG."""
    source: str                                # corrected node
    loss: float                                # epistemic loss at the source
    gradients: Dict[str, float]                # node_id -> gradient magnitude
    trust_adjustments: Dict[str, float]        # node_id -> delta_trust (negative)
    affected_nodes: Set[str]                   # all nodes touched
    depth: int                                 # max backward depth reached
    learning_rate_used: Dict[str, float]       # node_id -> lr applied
    path_trace: Dict[str, List[str]]           # node_id -> path from source


# ---------------------------------------------------------------------------
# Epistemic Loss Function
# ---------------------------------------------------------------------------

class EpistemicLoss:
    """Computes loss from a correction event.

    Higher loss when:
      - System was very confident AND wrong
      - This node has been corrected multiple times
      - User corrected it (highest authority)
      - Error persisted for a long time

    L = trust_at_assertion * (1 + log(1 + times_corrected)) * source_weight * time_decay
    """

    SOURCE_WEIGHTS = {
        "user": 1.0,         # user correction = highest authority
        "system": 0.6,       # system-detected contradiction
        "inference": 0.3,    # inferred from downstream evidence
    }

    def __init__(self, time_decay_halflife: float = 86400.0):
        """
        Args:
            time_decay_halflife: seconds after which time factor = 0.5
                                 (default 24h -- errors that persist longer cost more)
        """
        self.time_decay_halflife = time_decay_halflife

    def compute(self, event: CorrectionEvent) -> float:
        """Compute epistemic loss for a correction event."""
        confidence_factor = event.trust_at_assertion
        repetition_factor = 1.0 + math.log1p(event.times_corrected)
        source_weight = self.SOURCE_WEIGHTS.get(event.correction_source, 0.5)

        if self.time_decay_halflife > 0 and event.time_since_assertion > 0:
            time_factor = 1.0 + min(1.0, event.time_since_assertion / self.time_decay_halflife)
        else:
            time_factor = 1.0

        return confidence_factor * repetition_factor * source_weight * time_factor


# ---------------------------------------------------------------------------
# Domain Volatility
# ---------------------------------------------------------------------------

class DomainVolatility:
    """Tracks per-domain correction frequency and computes volatility scores.

    volatility = corrections / total_assertions * recency_weight

    High volatility -> higher learning rate, more hedging, more verification.
    Low volatility -> lower learning rate, confident assertions.
    """

    def __init__(self, recency_halflife: float = 172800.0):
        """
        Args:
            recency_halflife: seconds after which a correction's recency weight = 0.5
                              (default 48h)
        """
        self.recency_halflife = recency_halflife
        self._history: Dict[str, List[Tuple[float, bool]]] = {}

    def record_assertion(self, domain: str, timestamp: Optional[float] = None):
        """Record that the system made an assertion in this domain."""
        ts = timestamp or time.time()
        if domain not in self._history:
            self._history[domain] = []
        self._history[domain].append((ts, False))

    def record_correction(self, domain: str, timestamp: Optional[float] = None):
        """Record that the system was corrected in this domain."""
        ts = timestamp or time.time()
        if domain not in self._history:
            self._history[domain] = []
        self._history[domain].append((ts, True))

    def get_volatility(self, domain: str, now: Optional[float] = None) -> float:
        """Compute volatility for a domain. Returns [0, 1]."""
        if domain not in self._history or not self._history[domain]:
            return 0.0

        now = now or time.time()
        total_weight = 0.0
        correction_weight = 0.0

        for ts, is_correction in self._history[domain]:
            age = max(0, now - ts)
            w = math.exp(-0.693 * age / self.recency_halflife) if self.recency_halflife > 0 else 1.0
            total_weight += w
            if is_correction:
                correction_weight += w

        if total_weight < 1e-10:
            return 0.0

        return correction_weight / total_weight

    def get_learning_rate(self, domain: str, base_lr: float = 0.1,
                          min_lr: float = 0.02, max_lr: float = 0.5,
                          now: Optional[float] = None) -> float:
        """Adaptive learning rate based on domain volatility."""
        vol = self.get_volatility(domain, now)
        lr = base_lr + vol * (max_lr - base_lr)
        return max(min_lr, min(max_lr, lr))

    def get_all_volatilities(self, now: Optional[float] = None) -> Dict[str, float]:
        """Get volatility scores for all tracked domains."""
        return {domain: self.get_volatility(domain, now) for domain in self._history}

    def seed_history(self, domain: str, assertions: int, corrections: int,
                     base_time: Optional[float] = None, spread: float = 3600.0):
        """Seed synthetic history for testing."""
        base = base_time or time.time()
        total = assertions + corrections
        if total == 0:
            return

        step = spread / max(1, total)
        events = []
        correction_interval = total / max(1, corrections) if corrections > 0 else float('inf')

        correction_count = 0
        for i in range(total):
            ts = base - spread + (i * step)
            if corrections > 0 and (i + 1) % max(1, round(correction_interval)) == 0 and correction_count < corrections:
                events.append((ts, True))
                correction_count += 1
            else:
                events.append((ts, False))

        self._history[domain] = events


# ---------------------------------------------------------------------------
# Backward Gradient Computation
# ---------------------------------------------------------------------------

def compute_backward_gradients(
    graph,               # networkx DiGraph
    corrected_node: str,
    loss: float,
    learning_rates: Dict[str, float],
    damping_factor: float = 0.9,
    cascade_threshold: float = 0.01,
    max_depth: int = 50,
) -> BackpropResult:
    """Compute backward gradients through a belief dependency graph.

    For each upstream node A that supports the corrected node:
        gradient(A) = loss * edge_weight(path) * damping^depth
        trust_adjustment(A) = -learning_rate(A) * gradient(A)

    Uses BFS over in_edges (reverse direction from forward cascade).

    Args:
        graph: networkx DiGraph with weighted edges
        corrected_node: the node that was wrong
        loss: epistemic loss at the corrected node
        learning_rates: per-node learning rates (from DomainVolatility)
        damping_factor: how much gradient decays per hop (default 0.9)
        cascade_threshold: minimum gradient to continue propagating
        max_depth: safety limit
    """
    gradients: Dict[str, float] = {corrected_node: loss}
    trust_adjustments: Dict[str, float] = {}
    path_trace: Dict[str, List[str]] = {corrected_node: [corrected_node]}
    lr_used: Dict[str, float] = {}
    depth_map: Dict[str, int] = {corrected_node: 0}
    max_depth_reached = 0

    queue = deque([(corrected_node, loss, 0)])

    while queue:
        node, gradient_at_node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        if not graph.has_node(node):
            continue

        for pred, _, edata in graph.in_edges(node, data=True):
            if pred == corrected_node:
                continue

            edge_type = edata.get("edge_type", "SUPPORTS")
            if edge_type in ("CONTRADICTS",):
                continue

            w = edata.get("weight", 0.5)
            propagated = gradient_at_node * w * damping_factor

            if propagated < cascade_threshold:
                continue

            new_depth = depth + 1

            if pred in gradients and propagated <= gradients[pred]:
                continue

            gradients[pred] = propagated
            depth_map[pred] = new_depth
            max_depth_reached = max(max_depth_reached, new_depth)

            lr = learning_rates.get(pred, 0.1)
            adjustment = -lr * propagated
            trust_adjustments[pred] = adjustment
            lr_used[pred] = lr

            parent_path = path_trace.get(node, [node])
            path_trace[pred] = parent_path + [pred]

            queue.append((pred, propagated, new_depth))

    return BackpropResult(
        source=corrected_node,
        loss=loss,
        gradients=gradients,
        trust_adjustments=trust_adjustments,
        affected_nodes=set(gradients.keys()),
        depth=max_depth_reached,
        learning_rate_used=lr_used,
        path_trace=path_trace,
    )


# ---------------------------------------------------------------------------
# Trust Update Application
# ---------------------------------------------------------------------------

def apply_trust_adjustments(
    trust_scores: Dict[str, float],
    adjustments: Dict[str, float],
    min_trust: float = 0.0,
    max_trust: float = 1.0,
) -> Dict[str, float]:
    """Apply backward-pass trust adjustments to a trust state.

    Returns new trust scores (does not mutate input).
    """
    new_scores = dict(trust_scores)
    for node_id, delta in adjustments.items():
        if node_id in new_scores:
            new_scores[node_id] = max(min_trust, min(max_trust, new_scores[node_id] + delta))
    return new_scores


def flat_demotion(trust: float, multiplier: float = 0.4) -> float:
    """Simple multiplicative demotion on contradiction."""
    return trust * multiplier
