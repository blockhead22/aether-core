"""Splat overlay on top of SubstrateGraph.

The substrate stores slot states as text + trust + timestamp. To run
predictive contradiction detection (Bhattacharyya overlap trends,
covariance velocity, trajectory extrapolation), each slot needs a
Gaussian splat in embedding space.

This adapter:

* Lazily embeds slot-state values via the process-wide LazyEncoder.
* Maintains one ``MemorySplat`` per slot, keyed by ``slot_id``.
* Replays the slot's history into the splat's trajectory so that
  ``predict_overlap_trend`` and ``covariance_velocity`` have signal
  on a freshly loaded substrate.
* Exposes ``predict_pairwise_contradictions`` and
  ``find_converging_pairs`` over the in-memory splat collection.

Pure overlay — never mutates SubstrateGraph state. If the encoder
isn't warm, builds() returns an empty splat map; callers degrade
gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from aether._lazy_encoder import LazyEncoder
from aether.substrate import SubstrateGraph, SlotState

from .splats import (
    MemorySplat,
    create_splat_from_type,
    update_splat_with_evidence,
    overlap_integral,
    detect_geometric_contradiction,
    predict_overlap_trend,
    covariance_velocity,
    GeometricContradictionResult,
)


# ---------------------------------------------------------------------------
# Cosine-trajectory predictor (works in 384D where overlap_integral degenerates)
#
# In high dimensions, Bhattacharyya overlap between two Gaussians is essentially
# zero unless centers nearly coincide, even for clearly converging trajectories.
# The robust signal in 384D is the cosine-similarity trajectory between centers:
# rising cosine = converging, falling cosine = diverging.
# ---------------------------------------------------------------------------


def cosine_trajectory(a: MemorySplat, b: MemorySplat) -> List[float]:
    """Cosine similarity between centers at each historical snapshot."""
    out: List[float] = []
    n = min(len(a.trajectory), len(b.trajectory))
    for i in range(n):
        mu_a = a.trajectory[i]['mu']
        mu_b = b.trajectory[i]['mu']
        denom = float(np.linalg.norm(mu_a) * np.linalg.norm(mu_b)) + 1e-12
        out.append(float(np.dot(mu_a, mu_b) / denom))
    return out


def predict_cosine_trend(
    a: MemorySplat,
    b: MemorySplat,
    steps: int = 3,
) -> Optional[List[float]]:
    """Predict cosine trajectory N steps ahead by linear extrapolation."""
    history = cosine_trajectory(a, b)
    if len(history) < 2:
        return None
    velocity = history[-1] - history[-2]
    extrapolated = list(history)
    for s in range(steps):
        nxt = extrapolated[-1] + velocity
        extrapolated.append(max(-1.0, min(1.0, nxt)))
    return extrapolated


# ---------------------------------------------------------------------------
# Predicted-contradiction record
# ---------------------------------------------------------------------------


@dataclass
class PredictedContradiction:
    """One predicted-future-conflict pair surfaced by trajectory analysis."""

    slot_id_a: str
    slot_id_b: str
    current_overlap: float          # Bhattacharyya coefficient now
    predicted_overlap: float        # extrapolated N steps ahead
    overlap_velocity: float         # per-step change
    converging: bool                # overlap trending up
    cosine: float                   # center cosine right now
    center_distance: float          # euclidean distance now
    explanation: str

    def to_dict(self) -> dict:
        return {
            "slot_id_a": self.slot_id_a,
            "slot_id_b": self.slot_id_b,
            "current_overlap": self.current_overlap,
            "predicted_overlap": self.predicted_overlap,
            "overlap_velocity": self.overlap_velocity,
            "converging": self.converging,
            "cosine": self.cosine,
            "center_distance": self.center_distance,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SubstrateSplatOverlay:
    """In-memory splat layer over a SubstrateGraph.

    Construct with an existing graph. Call ``build()`` to embed slot
    histories and assemble splats. Call prediction methods after.
    """

    def __init__(
        self,
        substrate: SubstrateGraph,
        encoder: Optional[LazyEncoder] = None,
        memory_type: str = "belief",
    ):
        self.substrate = substrate
        self.encoder = encoder or LazyEncoder()
        self.memory_type = memory_type
        self.splats: Dict[str, MemorySplat] = {}      # slot_id -> splat
        self.last_state_id: Dict[str, str] = {}       # slot_id -> most recent state replayed
        self._unavailable = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def ensure_encoder(self, wait: bool = True, timeout: float = 60.0) -> bool:
        """Make sure the encoder is loaded. Returns True if usable."""
        if self.encoder.is_loaded:
            return True
        if self.encoder.is_unavailable:
            self._unavailable = True
            return False
        if wait:
            self.encoder.wait_until_ready(timeout=timeout)
        else:
            self.encoder.start_warmup()
        if self.encoder.is_unavailable:
            self._unavailable = True
        return self.encoder.is_loaded

    def build(self, wait_for_encoder: bool = True) -> Dict[str, MemorySplat]:
        """Walk every slot's history, replay into splats with trajectories.

        Returns the splat map. Empty if encoder is unavailable.
        """
        if not self.ensure_encoder(wait=wait_for_encoder):
            return {}

        for slot_id, state_ids in self.substrate._states_by_slot.items():
            if not state_ids:
                continue
            self._replay_slot(slot_id, state_ids)
        return self.splats

    def _replay_slot(self, slot_id: str, state_ids: List[str]) -> None:
        """Build/refresh the splat for one slot from its full state history."""
        states = [self.substrate.states[sid] for sid in state_ids
                  if sid in self.substrate.states]
        if not states:
            return

        # First state seeds the splat
        first = states[0]
        first_vec = self._embed(first.value)
        if first_vec is None:
            return
        splat = create_splat_from_type(
            memory_id=slot_id,
            embedding=first_vec,
            text=first.value,
            memory_type=self.memory_type,
            confidence=first.trust,
        )
        # Tag the splat with the original observation timestamp so
        # covariance_velocity reflects observed-time deltas not wall-clock.
        if splat.trajectory:
            splat.trajectory[-1]["timestamp"] = first.observed_at

        # Subsequent states feed evidence updates.
        for state in states[1:]:
            vec = self._embed(state.value)
            if vec is None:
                continue
            update_splat_with_evidence(splat, vec, weight=0.3)
            splat.snapshot()
            splat.trajectory[-1]["timestamp"] = state.observed_at

        self.splats[slot_id] = splat
        self.last_state_id[slot_id] = state_ids[-1]

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """One-shot encode. Returns None if encoder is not ready."""
        try:
            v = self.encoder.encode(text)
        except Exception:
            return None
        if v is None:
            return None
        return np.asarray(v, dtype=np.float32)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_pairwise_contradictions(
        self,
        steps_ahead: int = 3,
        min_current_cosine: float = 0.3,
        require_converging: bool = True,
        same_namespace: bool = True,
        signal: str = "cosine",
    ) -> List[PredictedContradiction]:
        """Find slot pairs whose centers are converging over time.

        - ``steps_ahead``: extrapolation horizon.
        - ``min_current_cosine``: only consider pairs with at least this
          much current center alignment (same broad topic).
        - ``require_converging``: drop stable / diverging pairs.
        - ``same_namespace``: restrict to pairs in the same namespace.
        - ``signal``: ``"cosine"`` (default, robust in 384D) or
          ``"overlap"`` (Bhattacharyya — degenerates to ~0 in high-d).
        """
        results: List[PredictedContradiction] = []
        items = list(self.splats.items())
        for i in range(len(items)):
            slot_id_a, splat_a = items[i]
            for j in range(i + 1, len(items)):
                slot_id_b, splat_b = items[j]
                if same_namespace and slot_id_a.split(":")[0] != slot_id_b.split(":")[0]:
                    continue
                if len(splat_a.trajectory) < 2 or len(splat_b.trajectory) < 2:
                    continue
                # Quick reject: not the same broad topic right now
                cos_now = float(np.dot(splat_a.mu, splat_b.mu) /
                                (np.linalg.norm(splat_a.mu) *
                                 np.linalg.norm(splat_b.mu) + 1e-12))
                if cos_now < min_current_cosine:
                    continue
                if signal == "overlap":
                    trend = predict_overlap_trend(splat_a, splat_b, steps=steps_ahead)
                else:
                    trend = predict_cosine_trend(splat_a, splat_b, steps=steps_ahead)
                if not trend or len(trend) < 3:
                    continue
                history_len = min(len(splat_a.trajectory), len(splat_b.trajectory))
                current = trend[history_len - 1]
                predicted = trend[-1]
                velocity = predicted - current
                converging = velocity > 1e-4
                if require_converging and not converging:
                    continue
                center_distance = float(np.linalg.norm(splat_a.mu - splat_b.mu))
                why = (
                    f"signal={signal}, cos_now={cos_now:.3f}, dist={center_distance:.3f}, "
                    f"value_now={current:.4f}, "
                    f"value_in_{steps_ahead}_steps={predicted:.4f}, "
                    f"velocity={velocity:+.4f}"
                )
                results.append(PredictedContradiction(
                    slot_id_a=slot_id_a,
                    slot_id_b=slot_id_b,
                    current_overlap=current,
                    predicted_overlap=predicted,
                    overlap_velocity=velocity,
                    converging=converging,
                    cosine=cos_now,
                    center_distance=center_distance,
                    explanation=why,
                ))
        results.sort(key=lambda r: r.overlap_velocity, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def velocity_report(self) -> List[Tuple[str, Optional[float]]]:
        """Per-slot covariance velocity. Positive = belief under pressure."""
        rows = []
        for slot_id, splat in self.splats.items():
            rows.append((slot_id, covariance_velocity(splat)))
        rows.sort(
            key=lambda r: (r[1] if r[1] is not None else float("-inf")),
            reverse=True,
        )
        return rows

    def static_pairwise(self, slot_id_a: str, slot_id_b: str) -> Optional[GeometricContradictionResult]:
        """Run the static (non-predictive) geometric detector on two slots."""
        a = self.splats.get(slot_id_a)
        b = self.splats.get(slot_id_b)
        if a is None or b is None:
            return None
        return detect_geometric_contradiction(a, b)
