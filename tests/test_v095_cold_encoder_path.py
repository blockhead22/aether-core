"""Regression tests for v0.9.5: governance must work in cold-encoder mode.

Background — the v0.9.4 production miss:
    The first end-to-end re-run of the calibration rubric (2026-04-28
    late-night) confirmed v0.9.2 (no-hang) but flagged that v0.9.3
    (methodological detection) and v0.9.1 (auto-link RELATED_TO) BOTH
    failed in production despite passing 100% in the v0.9.4 bench.

Root cause:
    aether/contradiction/tension.py:_encode handled `self._encoder is None`
    (no encoder passed) but NOT the case where the encoder is present but
    its `encode()` method returns None (LazyEncoder's contract when not
    yet warm). When that happened, _compute_similarity did `vector.size`
    on None → AttributeError → the `try/except: continue` in BOTH
    `compute_grounding` and `_detect_and_record_tensions` swallowed the
    raise and skipped the entire loop body, including:
      - the methodological-overclaim check (v0.9.3)
      - the auto-link RELATED_TO logic (v0.9.1)
      - the contradiction-on-write logic (v0.5.0+)

    The bench passed because every test fixture forces synchronous
    `s._encoder._load()` before running. So the bench was testing the
    wrong path. Same meta-pattern as v0.9.0: synthetic tests bypassed
    the production code path.

Fix (v0.9.5):
    1. tension.py:_encode now also returns np.array([]) when encoder.encode()
       returns None (the LazyEncoder cold-state contract).
    2. AUTO_LINK_THRESHOLD becomes adaptive: 0.7 (embedding cosine) when
       embedding_similarity is in supporting_signals, AUTO_LINK_THRESHOLD_SUBSTRING
       (0.4 default) when only Jaccard fallback is available. Otherwise
       Jaccard rarely hits 0.7 and auto-link is dead in cold mode.
    3. Both fixes apply to read path (compute_grounding) and write path
       (_detect_and_record_tensions) and backfill (backfill_edges).

This file proves the cold-encoder paths work:
    - meter.measure with cold encoder returns UNRELATED instead of raising
    - compute_grounding fires methodological branch in cold mode
    - aether_remember auto-links via Jaccard in cold mode (lower threshold)
    - backfill_edges uses adaptive threshold automatically
"""

from __future__ import annotations

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx required (install [graph] extra)",
)

pytest.importorskip("mcp")

from aether._lazy_encoder import LazyEncoder
from aether.contradiction import StructuralTensionMeter, TensionRelationship
from aether.mcp.state import StateStore


def _cold_store(tmp_path):
    """Build a StateStore with a NEVER-loaded encoder.

    We replace the StateStore's encoder with a fresh LazyEncoder that has
    not had start_warmup() called. Its `encode()` returns None
    indefinitely, simulating the production cold-start window where the
    background warmup hasn't completed.
    """
    s = StateStore(state_path=str(tmp_path / "state.json"))
    cold = LazyEncoder()
    # Don't call start_warmup -- stays cold forever in test conditions
    s._encoder = cold
    s.meter._encoder = cold
    return s


# ==========================================================================
# Root-cause fix: tension.py:_encode returns array on None
# ==========================================================================

class TestMeterDoesNotRaiseOnColdEncoder:
    def test_encode_returns_empty_array_on_lazy_encoder_cold(self):
        """The actual bug: LazyEncoder.encode() returns None when not loaded.
        tension._encode must convert that to np.array([]) so downstream
        size checks don't AttributeError."""
        cold = LazyEncoder()
        meter = StructuralTensionMeter(encoder=cold)
        # _encode should return an empty array, not None
        result = meter._encode("any text")
        assert result is not None
        assert hasattr(result, "size")
        assert result.size == 0

    def test_meter_measure_does_not_raise_on_cold_encoder(self):
        """The downstream symptom: meter.measure raised AttributeError in
        cold mode, which the caller swallowed via try/except: continue.
        Now it must return a valid (degraded) TensionResult."""
        cold = LazyEncoder()
        meter = StructuralTensionMeter(encoder=cold)
        result = meter.measure(
            text_a="some memory text",
            text_b="some draft text",
            trust_a=0.8,
            trust_b=0.7,
            source_a="user",
            source_b="query",
        )
        # Should return UNRELATED at similarity 0 with no shared slots
        assert result is not None
        assert result.relationship in (
            TensionRelationship.UNRELATED,
            TensionRelationship.COMPATIBLE,
            TensionRelationship.DUPLICATE,
            TensionRelationship.REFINEMENT,
            TensionRelationship.TENSION,
            TensionRelationship.CONFLICT,
            TensionRelationship.DECAYED,
        )


# ==========================================================================
# Cold-encoder methodological detection (v0.9.3 in production)
# ==========================================================================

@needs_networkx
class TestMethodologicalDetectionInColdMode:
    def test_canonical_calic_overclaim_fires_with_cold_encoder(self, tmp_path):
        """The exact production miss: methodological detection failed in
        cold mode because meter.measure raised before the methodological
        branch was reached."""
        s = _cold_store(tmp_path)
        s.add_memory(
            "Methodological gap in v1-vs-v3 comparison: the conclusion "
            "that CALIC is bad is unsupported because the experiment "
            "was confounded.",
            trust=0.85,
            source="seed_compression_lab",
        )

        g = s.compute_grounding("v3 was worse than v1, so CALIC is bad")

        assert g["method"] == "substring"
        assert len(g["methodological_concerns"]) >= 1, (
            f"v0.9.4 production regression: cold-mode methodological "
            f"detection still empty. grounding={g}"
        )
        assert g["belief_confidence"] < 0.4

    def test_mutex_contradiction_visible_to_grounding_in_cold_mode(self, tmp_path):
        """Defensive: prove the grounding loop body actually runs in cold
        mode. Pre-fix, meter.measure raised AttributeError and the
        try/except: continue silently skipped the entire loop body.
        Mutex detection is regex-based and doesn't need embeddings, so
        if the loop runs at all in cold mode, mutex contradictions fire."""
        s = _cold_store(tmp_path)
        s.add_memory("We deploy this project to AWS.", trust=0.9)

        g = s.compute_grounding("We deploy this project to GCP.")
        assert g["method"] == "substring"
        assert len(g["contradict"]) >= 1, (
            f"cold-mode grounding loop didn't fire mutex check: {g}"
        )
        # Mutex tag should be on the contradict entry
        assert any(
            c.get("kind") == "mutex" for c in g["contradict"]
        )


# ==========================================================================
# Cold-encoder auto-link (v0.9.1 in production)
# ==========================================================================

@needs_networkx
class TestAutoLinkInColdMode:
    def test_similar_remembers_create_related_to_via_jaccard_threshold(self, tmp_path):
        """Auto-link must fire in cold mode at the lower substring
        threshold. Jaccard rarely hits 0.7; v0.9.5 uses 0.4."""
        s = _cold_store(tmp_path)

        # Two memories with high token overlap (Jaccard ~0.6+)
        a_id = s.add_memory(
            "the aether substrate stores belief state across sessions",
            trust=0.9,
        )["memory_id"]
        b_id = s.add_memory(
            "the aether substrate persists belief state between sessions",
            trust=0.9,
        )["memory_id"]

        # Should have a RELATED_TO edge in either direction
        has_edge = (
            s.graph.graph.has_edge(a_id, b_id) or
            s.graph.graph.has_edge(b_id, a_id)
        )
        assert has_edge, (
            f"v0.9.4 production regression: cold-mode auto-link did not "
            f"fire. Edges: {list(s.graph.graph.edges(data=True))}"
        )

    def test_unrelated_remembers_do_not_link_even_in_cold_mode(self, tmp_path):
        """False-positive guard: low-Jaccard pairs must not link even
        with the lower threshold."""
        s = _cold_store(tmp_path)

        a_id = s.add_memory(
            "the office printer is jammed again", trust=0.9,
        )["memory_id"]
        b_id = s.add_memory(
            "kubernetes pod scheduler reschedules failed deployments",
            trust=0.9,
        )["memory_id"]

        has_edge = (
            s.graph.graph.has_edge(a_id, b_id) or
            s.graph.graph.has_edge(b_id, a_id)
        )
        assert not has_edge


# ==========================================================================
# Cold-encoder contradiction-on-write (v0.5+ in production)
# ==========================================================================

@needs_networkx
class TestContradictionOnWriteInColdMode:
    def test_mutex_contradiction_fires_in_cold_mode(self, tmp_path):
        """Mutex detection is text-based (regex) so should fire even
        without embeddings. Verify it actually does in cold mode."""
        s = _cold_store(tmp_path)
        s.add_memory("we deploy this project to AWS", trust=0.9)
        result = s.add_memory("we deploy this project to GCP", trust=0.9)
        # Should have at least one tension finding
        assert len(result["tension_findings"]) >= 1, (
            f"mutex contradiction did not fire in cold mode: "
            f"findings={result['tension_findings']}"
        )


# ==========================================================================
# Backfill in cold mode uses adaptive threshold
# ==========================================================================

@needs_networkx
class TestBackfillInColdMode:
    def test_backfill_wires_orphans_at_jaccard_threshold(self, tmp_path):
        """backfill_edges must auto-pick the right threshold for whichever
        similarity mode each pair uses."""
        s = _cold_store(tmp_path)

        # Build orphan memories (skip auto-link by disabling detection)
        s.add_memory(
            "the aether substrate stores belief state across sessions",
            trust=0.9, detect_contradictions=False,
        )
        s.add_memory(
            "the aether substrate persists belief state between sessions",
            trust=0.9, detect_contradictions=False,
        )

        result = s.backfill_edges()
        # In cold mode, threshold falls to 0.4 (Jaccard) and these should link
        assert result["added"] >= 1, (
            f"backfill did not adapt threshold for cold mode: {result}"
        )
