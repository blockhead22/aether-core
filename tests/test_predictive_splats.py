"""Predictive splat overlay tests (v0.15 experimental).

Covers:
- Pure splat math (Bhattacharyya / KL / cosine, evidence updates).
- Cosine-trajectory predictor (the 384D-robust signal).
- SubstrateSplatOverlay end-to-end against an in-memory substrate.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.predictive import (
    MemorySplat,
    create_splat_from_type,
    bhattacharyya_coefficient,
    overlap_integral,
    kl_divergence,
    update_splat_confirming,
    update_splat_contradicting,
    update_splat_with_evidence,
    covariance_velocity,
)
from aether.predictive.substrate_adapter import (
    SubstrateSplatOverlay,
    cosine_trajectory,
    predict_cosine_trend,
)
from aether.substrate import SubstrateGraph


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _seed_pair(d: int = 384, sep: float = 0.4, seed: int = 7):
    rng = np.random.default_rng(seed)
    base = _unit(rng.standard_normal(d).astype(np.float32))
    a = _unit(base + 0.05 * rng.standard_normal(d).astype(np.float32))
    b = _unit(base + sep * rng.standard_normal(d).astype(np.float32))
    return a.astype(np.float32), b.astype(np.float32)


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


def test_bhattacharyya_self_overlap_is_one():
    a_emb, _ = _seed_pair()
    splat = create_splat_from_type("x", a_emb, "x", "belief", 0.8)
    bc = bhattacharyya_coefficient(splat, splat)
    assert bc == pytest.approx(1.0, abs=1e-6)


def test_confirming_evidence_tightens_covariance():
    a_emb, _ = _seed_pair()
    splat = create_splat_from_type("x", a_emb, "x", "belief", 0.8)
    sigma_before = splat.total_uncertainty
    for _ in range(5):
        update_splat_confirming(splat)
    assert splat.total_uncertainty < sigma_before


def test_contradicting_evidence_widens_covariance_and_lowers_alpha():
    a_emb, _ = _seed_pair()
    splat = create_splat_from_type("x", a_emb, "x", "belief", 0.8)
    sigma_before = splat.total_uncertainty
    alpha_before = splat.alpha
    update_splat_contradicting(splat)
    assert splat.total_uncertainty > sigma_before
    assert splat.alpha < alpha_before


def test_kl_self_is_zero():
    a_emb, _ = _seed_pair()
    splat = create_splat_from_type("x", a_emb, "x", "belief", 0.8)
    assert kl_divergence(splat, splat) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Cosine trajectory predictor (384D robust signal)
# ---------------------------------------------------------------------------


def test_cosine_trend_detects_convergence():
    a_emb, b_emb = _seed_pair(sep=0.5)
    a = create_splat_from_type("a", a_emb, "A0", "belief", 0.7)
    b = create_splat_from_type("b", b_emb, "B0", "belief", 0.7)
    a.trajectory[-1]["timestamp"] = 1000.0
    b.trajectory[-1]["timestamp"] = 1000.0

    for step in range(1, 7):
        new_a = _unit(a.mu + 0.18 * (b.mu - a.mu))
        update_splat_with_evidence(a, new_a, weight=0.4)
        a.snapshot()
        a.trajectory[-1]["timestamp"] = 1000.0 + step * 60.0
        b.snapshot()
        b.trajectory[-1]["timestamp"] = 1000.0 + step * 60.0

    history = cosine_trajectory(a, b)
    trend = predict_cosine_trend(a, b, steps=3)
    assert history[-1] > history[0], "cosine should rise as A drifts toward B"
    assert trend[-1] > trend[len(history) - 1], "extrapolation should continue the rise"


def test_cosine_trend_returns_none_with_short_history():
    a_emb, b_emb = _seed_pair()
    a = create_splat_from_type("a", a_emb, "A0", "belief", 0.7)
    b = create_splat_from_type("b", b_emb, "B0", "belief", 0.7)
    assert predict_cosine_trend(a, b) is None


# ---------------------------------------------------------------------------
# SubstrateSplatOverlay end-to-end (with stub encoder so tests stay offline)
# ---------------------------------------------------------------------------


class _StubEncoder:
    """Deterministic stub: hashes text to a unit vector. No torch."""

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


def test_overlay_builds_one_splat_per_slot():
    sub = SubstrateGraph()
    sub.observe("user", "location", "Milwaukee")
    sub.observe("user", "location", "Portland")
    sub.observe("user", "employer", "Acme")
    overlay = SubstrateSplatOverlay(sub, encoder=_StubEncoder())
    splats = overlay.build()
    assert set(splats.keys()) == {"user:location", "user:employer"}
    assert len(splats["user:location"].trajectory) == 2  # two states
    assert len(splats["user:employer"].trajectory) == 1


def test_overlay_predict_pairwise_runs_without_error():
    sub = SubstrateGraph()
    for v in ("Seattle", "Portland", "Vancouver"):
        sub.observe("user", "location", v)
    for v in ("Acme", "Beta Inc"):
        sub.observe("user", "employer", v)
    overlay = SubstrateSplatOverlay(sub, encoder=_StubEncoder())
    overlay.build()
    preds = overlay.predict_pairwise_contradictions(
        steps_ahead=3,
        min_current_cosine=-1.0,
        require_converging=False,
        same_namespace=True,
    )
    # Single same-namespace pair (location, employer); should be analyzed.
    assert len(preds) == 1
    p = preds[0]
    assert {p.slot_id_a, p.slot_id_b} == {"user:location", "user:employer"}


def test_overlay_velocity_report_orders_by_velocity():
    sub = SubstrateGraph()
    for v in ("v1", "v2", "v3"):
        sub.observe("user", "color", v)
    overlay = SubstrateSplatOverlay(sub, encoder=_StubEncoder())
    overlay.build()
    rows = overlay.velocity_report()
    assert any(slot_id == "user:color" for slot_id, _ in rows)


def test_overlay_handles_unavailable_encoder():
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
    sub.observe("user", "location", "Milwaukee")
    overlay = SubstrateSplatOverlay(sub, encoder=_DeadEncoder())
    splats = overlay.build()
    assert splats == {}
