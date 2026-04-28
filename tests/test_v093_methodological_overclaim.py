"""Regression tests for v0.9.3 (Layer 2): methodological overclaim detection.

Background — the v0.9.1 fidelity miss:
    The first end-to-end substrate-assisted dev test (2026-04-28) seeded a
    methodological-gap memory: "the v1-vs-v3 CogniMap comparison is
    unsupported because two variables changed simultaneously (predictor
    and strategy map) — the conclusion that CALIC is bad is unsupported."
    Trust 0.85.

    The agent then drafted a comment "v3 was worse than v1, so CALIC is bad"
    and called aether_fidelity to grade it.

    Fidelity returned: gap_score 0.0, action: PASS,
                       supporting_memories: [],
                       contradicting_memories: [].

    The miss: fidelity's contradiction detection is wired around the
    StructuralTensionMeter, which compares slot extractions
    (Seattle vs Portland). The methodological-gap memory has different
    slots than the draft (no slot conflict), so the meter classified them
    as unrelated. The memory dropped out of the grounding entirely.

The fix:
    Add a separate methodological-overclaim channel to compute_grounding.
    Fires when:
      1. The draft makes an inference (`so X`, `therefore Y`,
         `means that`, etc.)
      2. A topically-similar memory contains methodological-warning
         language ("unsupported", "doesn't follow", "missing cell",
         "confounded") OR carries source:methodological_gap.

    Surfaces in compute_grounding's output as `methodological_concerns`,
    a separate list from `contradict`. Reduces belief_confidence so
    gap_score / severity reflect the concern. aether_sanction also
    surfaces it; high-trust methodological concerns force HOLD.

This file proves the fix:
    - The exact draft + memory pair from the v0.9.1 fidelity miss now
      surfaces in `methodological_concerns`.
    - belief_confidence drops, gap_score rises, severity moves off SAFE.
    - aether_fidelity and aether_sanction expose the new field.
    - False-positive guards: dissimilar text doesn't surface concerns;
      drafts without inference markers don't trigger the channel.
"""

from __future__ import annotations

import asyncio
import json

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

from aether.mcp.state import (
    StateStore,
    _has_inference_marker,
    _has_methodological_signal,
)


@pytest.fixture
def store(tmp_path):
    s = StateStore(state_path=str(tmp_path / "state.json"))
    if s._encoder is not None:
        s._encoder._load()  # synchronous so similarity is real
    return s


def _extract(call_result):
    return json.loads(call_result[0].text)


def _run(coro):
    return asyncio.run(coro)


# ==========================================================================
# Helper-level unit tests
# ==========================================================================

class TestInferenceMarkers:
    def test_so_marker_detected(self):
        assert _has_inference_marker("v3 was worse than v1, so CALIC is bad")

    def test_therefore_marker_detected(self):
        assert _has_inference_marker("the test failed; therefore the design is wrong")

    def test_proves_marker_detected(self):
        assert _has_inference_marker("this proves the hypothesis")

    def test_implies_marker_detected(self):
        assert _has_inference_marker("the data implies a clear pattern")

    def test_no_marker_in_plain_statement(self):
        assert not _has_inference_marker("the temperature is 72 degrees")

    def test_no_marker_in_imperative(self):
        assert not _has_inference_marker("rm test_image.jpg")

    def test_substring_does_not_falsely_match(self):
        # 'son' contains 'so' as a substring; must not match
        assert not _has_inference_marker("my son went to school")


class TestMethodologicalSignal:
    def test_unsupported_signal(self):
        assert _has_methodological_signal(
            "this conclusion is unsupported by the data", source="user",
        )

    def test_confounded_signal(self):
        assert _has_methodological_signal(
            "the experiment is confounded — two variables changed", source="research",
        )

    def test_missing_cell_signal(self):
        assert _has_methodological_signal(
            "missing cell in the experimental matrix", source="research",
        )

    def test_doesnt_follow_signal(self):
        assert _has_methodological_signal(
            "the conclusion doesn't follow from the premises", source="user",
        )

    def test_source_tag_pathway(self):
        # Even without signal text, the source tag surfaces it
        assert _has_methodological_signal(
            "some neutral memory text",
            source="methodological_gap",
        )

    def test_no_signal_in_factual_memory(self):
        assert not _has_methodological_signal(
            "we deploy this project to AWS", source="user",
        )

    def test_no_signal_when_source_none(self):
        # Defensive: source=None shouldn't crash
        assert not _has_methodological_signal(
            "some neutral text", source=None,
        )


# ==========================================================================
# THE CANONICAL TEST — the v0.9.1 fidelity miss
# ==========================================================================

@needs_networkx
class TestCanonicalCalicOverclaim:
    """Reproduces the exact scenario from the v0.9.1 end-to-end test report."""

    def _seed_methodological_gap(self, store):
        """Seed the methodological-gap memory exactly as the test agent did."""
        result = store.add_memory(
            "The v1-vs-v3 CogniMap comparison is confounded because two "
            "variables changed simultaneously (predictor type and strategy "
            "map). The conclusion that CALIC is bad is unsupported — the "
            "missing cell is `fancy predictor + map`, which was never tested.",
            trust=0.85,
            source="methodological_gap",
        )
        return result["memory_id"]

    def test_grounding_surfaces_methodological_concern(self, store):
        """The exact pair: methodological-gap memory + overclaim draft.
        Must surface in `methodological_concerns`."""
        self._seed_methodological_gap(store)

        draft = "v3 was worse than v1, so CALIC is bad"
        grounding = store.compute_grounding(draft)

        concerns = grounding["methodological_concerns"]
        assert len(concerns) >= 1, (
            f"v0.9.1 miss not fixed: methodological_concerns is empty. "
            f"grounding={grounding}"
        )
        # Concern entry should be the seeded memory
        assert any(
            "unsupported" in c["text"].lower() or
            "confounded" in c["text"].lower() or
            c.get("source") == "methodological_gap"
            for c in concerns
        )
        # Should be tagged as methodological
        assert all(c.get("kind") == "methodological" for c in concerns)

    def test_belief_confidence_drops_due_to_methodological_concern(self, store):
        """The penalty math: methodological concerns reduce belief_confidence
        the same way factual contradictions do."""
        self._seed_methodological_gap(store)

        draft = "v3 was worse than v1, so CALIC is bad"
        grounding = store.compute_grounding(draft)

        # Without the methodological-concerns channel, belief_confidence
        # stayed near 0.4 (neutral). With it, the high-trust concern drops
        # belief substantially.
        assert grounding["belief_confidence"] < 0.4, (
            f"belief_confidence={grounding['belief_confidence']} — "
            f"methodological concern did not penalize belief"
        )

    def test_dissimilar_topic_does_not_falsely_surface(self, store):
        """False-positive guard: methodological memory about CALIC must not
        trigger on an unrelated draft about, e.g., authentication."""
        self._seed_methodological_gap(store)

        draft = "the auth token expires after one hour, so we cache it"
        grounding = store.compute_grounding(draft)

        # The methodological memory is on a different topic — must not
        # surface in concerns. It might appear in support/contradict via
        # the existing meter, but `methodological_concerns` should be empty
        # (the topical similarity doesn't pass the GROUNDING_MIN_SCORE
        # threshold for this dissimilar pair).
        assert len(grounding["methodological_concerns"]) == 0

    def test_draft_without_inference_marker_does_not_trigger(self, store):
        """False-positive guard: a plain factual claim, even if topically
        similar to the methodological memory, must not trigger the
        methodological channel — that channel is specifically for
        inference overclaims."""
        self._seed_methodological_gap(store)

        # Plain factual claim about CALIC — no inference marker
        draft = "CALIC is one type of predictor used in lossless compression"
        grounding = store.compute_grounding(draft)

        # No inference, no methodological concern even though topically
        # close to the seeded memory
        assert len(grounding["methodological_concerns"]) == 0


# ==========================================================================
# MCP tool surface — fidelity and sanction expose the new field
# ==========================================================================

@needs_networkx
class TestFidelityToolSurface:
    def test_aether_fidelity_includes_methodological_concerns_field(self, store):
        from aether.mcp.server import build_server
        server = build_server(store=store)

        store.add_memory(
            "The v1-vs-v3 conclusion that CALIC is bad is unsupported — "
            "the comparison was confounded.",
            trust=0.85,
            source="methodological_gap",
        )

        result = _extract(_run(server.call_tool(
            "aether_fidelity",
            {"response": "v3 was worse than v1, so CALIC is bad"},
        )))
        assert "methodological_concerns" in result
        assert len(result["methodological_concerns"]) >= 1

    def test_aether_fidelity_severity_off_safe_with_concern(self, store):
        from aether.mcp.server import build_server
        server = build_server(store=store)

        store.add_memory(
            "This conclusion is unsupported; the comparison is confounded.",
            trust=0.9,
            source="methodological_gap",
        )

        result = _extract(_run(server.call_tool(
            "aether_fidelity",
            {"response": "the data is bad, therefore the model is wrong"},
        )))
        # gap_score should be elevated relative to a no-concern baseline
        # (we just assert it's not exactly 0 — that was the canonical miss)
        assert result["gap_score"] > 0.0


@needs_networkx
class TestSanctionToolSurface:
    def test_aether_sanction_includes_methodological_concerns_field(self, store):
        from aether.mcp.server import build_server
        server = build_server(store=store)

        store.add_memory(
            "Skipping tests is unsupported as a CI strategy — it confounds "
            "regression detection with deploy speed.",
            trust=0.9,
            source="methodological_gap",
        )

        result = _extract(_run(server.call_tool(
            "aether_sanction",
            {"action": "tests are slow, so we should skip them"},
        )))
        assert "methodological_concerns" in result

    def test_high_trust_methodological_concern_forces_hold(self, store):
        """High-trust methodological concern + APPROVE-baseline action
        should be downgraded to HOLD."""
        from aether.mcp.server import build_server
        server = build_server(store=store)

        store.add_memory(
            "This conclusion is unsupported because the experiment was "
            "confounded; the missing cell was never tested.",
            trust=0.95,
            source="methodological_gap",
        )

        result = _extract(_run(server.call_tool(
            "aether_sanction",
            {"action": "the test passed, therefore we should ship"},
        )))
        # APPROVE would be the baseline. HOLD or REJECT means the
        # methodological concern was acted upon.
        assert result["verdict"] in ("HOLD", "REJECT")


# ==========================================================================
# Empty-substrate / no-concern paths still work
# ==========================================================================

@needs_networkx
class TestNoConcernPath:
    def test_empty_substrate_returns_empty_concerns_list(self, store):
        grounding = store.compute_grounding("anything at all")
        assert grounding["methodological_concerns"] == []

    def test_grounding_with_only_factual_support_works(self, store):
        store.add_memory("we use Python 3.10", trust=0.9)
        grounding = store.compute_grounding("we use Python 3.10")
        # Methodological concerns is empty; existing support/contradict logic
        # works unchanged.
        assert grounding["methodological_concerns"] == []

    def test_grounding_with_only_factual_contradiction_works(self, store):
        store.add_memory("we deploy to AWS", trust=0.9)
        grounding = store.compute_grounding("we deploy to GCP")
        assert grounding["methodological_concerns"] == []
