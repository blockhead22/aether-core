"""Regression tests for v0.12.0: slot-equality detector + shape local-context gate.

Two parallel fixes shipped in v0.12. Both came directly out of Lab A v2's
production substrate finding (bench/lab_a_v2_production_substrate_findings.md):

    Fix A — slot-equality detector
        Production substrate had 42 real value-change contradictions on
        slots like user.name (Nick<>Aether), user.favorite_color
        (blue<>orange), user.location, etc. v0.11 detection caught 0/42
        because:
          - shape() doesn't fire on categorical strings
          - mutex registry only knows ~10 technical-domain classes
        v0.12 adds patterns.slot_equality(): if two memories tag the same
        slot with different categorical values, that's a conflict.

    Fix B — shape local-context gate
        v0.11 shape() compared every typed value pairwise. False-fired on
        the action text vs Lab A v2 memory because both contained "v0.12"
        and "v0.11" tokens — same shape, different topics. The substrate
        sanction REJECTED v0.12's first design pass; the rejection itself
        was the substrate-assisted dev loop catching a v0.11 production
        bug. Fix: require the immediate surrounding tokens (3 before, 3
        after) to overlap above LOCAL_CONTEXT_MIN_OVERLAP (0.30) before
        treating differing values as a conflict.

This file tests:
    - slot_equality primitive in isolation
    - slot-equality integration via add_memory (write path)
    - slot-equality integration via compute_grounding (read path)
    - low-textual-similarity slot conflict still fires (production case)
    - local-context gate suppresses the sanction-self-reject false positive
    - real conflicts (Python 3.10 vs 3.8) still fire after gate
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

from aether.patterns import (
    slot_equality,
    shape,
    MatchResult,
    LOCAL_CONTEXT_MIN_OVERLAP,
)


# ==========================================================================
# Primitive: slot_equality in isolation
# ==========================================================================

class TestSlotEqualityPrimitive:
    def test_disagreement_scores_one(self):
        r = slot_equality(
            ["source:user", "slot:user.name=Nick"],
            ["source:llm", "slot:user.name=Aether"],
        )
        assert r.score == 1.0
        assert r.primitive == "slot_equality"
        conflicts = r.evidence["conflicts"]
        assert len(conflicts) == 1
        assert conflicts[0]["slot"] == "user.name"
        assert conflicts[0]["a"] == "Nick"
        assert conflicts[0]["b"] == "Aether"

    def test_agreement_scores_half(self):
        r = slot_equality(
            ["slot:user.name=Nick"],
            ["slot:user.name=Nick"],
        )
        assert r.score == 0.5
        assert r.evidence["agreements"] == [{"slot": "user.name", "value": "Nick"}]

    def test_no_shared_slots_scores_zero(self):
        r = slot_equality(
            ["slot:user.color=blue"],
            ["slot:user.name=Aether"],
        )
        assert r.score == 0.0
        assert "no_shared_slots" in r.evidence["reason"]

    def test_case_insensitive_equality(self):
        # "Nick" and "nick" should NOT be a conflict
        r = slot_equality(
            ["slot:user.name=Nick"],
            ["slot:user.name=nick"],
        )
        assert r.score == 0.5

    def test_whitespace_stripped(self):
        r = slot_equality(
            ["slot:user.name= Nick"],
            ["slot:user.name=Nick "],
        )
        assert r.score == 0.5

    def test_multiple_shared_slots_one_conflict(self):
        r = slot_equality(
            ["slot:user.name=Nick", "slot:user.color=blue"],
            ["slot:user.name=Aether", "slot:user.color=blue"],
        )
        assert r.score == 1.0
        conflicts = r.evidence["conflicts"]
        agreements = r.evidence["agreements"]
        assert len(conflicts) == 1
        assert conflicts[0]["slot"] == "user.name"
        assert len(agreements) == 1
        assert agreements[0]["slot"] == "user.color"

    def test_non_slot_tags_ignored(self):
        # source:user, type:fact, etc. are ignored
        r = slot_equality(
            ["source:user", "type:fact", "slot:user.name=Nick"],
            ["source:llm", "type:fact", "slot:user.name=Aether"],
        )
        assert r.score == 1.0

    def test_malformed_slot_tag_ignored(self):
        # tag without "=" is ignored
        r = slot_equality(
            ["slot:user.name", "slot:user.color=blue"],
            ["slot:user.color=red"],
        )
        assert r.score == 1.0
        conflicts = r.evidence["conflicts"]
        assert len(conflicts) == 1
        assert conflicts[0]["slot"] == "user.color"

    def test_empty_tag_lists(self):
        r = slot_equality([], [])
        assert r.score == 0.0


# ==========================================================================
# Integration: slot-equality fires on write path (add_memory)
# ==========================================================================

@needs_networkx
class TestSlotEqualityWritePath:
    """The Lab A v2 production case: 17 instances of Nick→Aether."""

    def test_nick_aether_high_similarity_text(self, tmp_path):
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"),
                       enable_embeddings=False)
        s.add_memory("The user is named Nick.", trust=0.8, source="user",
                     slots={"user.name": "Nick"})
        result = s.add_memory("Yes, the user is named Aether.", trust=0.5,
                              source="llm_stated",
                              slots={"user.name": "Aether"})

        findings = result["tension_findings"]
        assert len(findings) >= 1
        kinds = [f.get("kind") for f in findings]
        assert "slot_value_conflict" in kinds
        # tension_score should be 0.9 (the slot-equality boost)
        scores = [f["tension_score"] for f in findings if f["kind"] == "slot_value_conflict"]
        assert max(scores) >= 0.9

    def test_nick_aether_low_similarity_text(self, tmp_path):
        """Production-style case: low textual overlap, slot-keyed conflict.

        Without the v0.12 slot pre-screen + sim-gate bypass, this fails
        because Jaccard ~0.14 falls below the 0.2 candidate gate.
        """
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"),
                       enable_embeddings=False)
        s.add_memory("User asked me to remember their name is Nick.",
                     trust=0.8, source="user",
                     slots={"user.name": "Nick"})
        result = s.add_memory("I noted that your name is Aether.",
                              trust=0.5, source="llm_stated",
                              slots={"user.name": "Aether"})

        findings = result["tension_findings"]
        assert len(findings) >= 1, (
            f"low-similarity slot conflict not caught: {findings}"
        )
        assert any(f.get("kind") == "slot_value_conflict" for f in findings)

    def test_completely_unrelated_texts_slot_conflict(self, tmp_path):
        """Even with zero textual overlap, slot conflict fires."""
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"),
                       enable_embeddings=False)
        s.add_memory("Pizza was delivered yesterday.", trust=0.8,
                     slots={"user.favorite_color": "blue"})
        result = s.add_memory("Tomorrow is going to be cold.",
                              trust=0.5, source="llm_stated",
                              slots={"user.favorite_color": "orange"})

        findings = result["tension_findings"]
        assert len(findings) >= 1
        assert any(f.get("kind") == "slot_value_conflict" for f in findings)

    def test_same_slot_value_no_conflict(self, tmp_path):
        """Memories agreeing on slot value should not produce a contradiction."""
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"),
                       enable_embeddings=False)
        s.add_memory("Nick is the name.", trust=0.8,
                     slots={"user.name": "Nick"})
        result = s.add_memory("User is Nick.", trust=0.5, source="llm_stated",
                              slots={"user.name": "Nick"})

        # No slot_value_conflict finding
        for f in result["tension_findings"]:
            assert f.get("kind") != "slot_value_conflict"

    def test_disjoint_slot_keys_no_conflict(self, tmp_path):
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"),
                       enable_embeddings=False)
        s.add_memory("Pizza was delivered.", trust=0.8,
                     slots={"user.color": "blue"})
        result = s.add_memory("Tomorrow is cold.", trust=0.5,
                              slots={"user.name": "Aether"})

        for f in result["tension_findings"]:
            assert f.get("kind") != "slot_value_conflict"


# ==========================================================================
# Integration: slot-equality surfaces in compute_grounding (read path)
# ==========================================================================

@needs_networkx
class TestSlotEqualityReadPath:
    """compute_grounding extracts slots from the draft text on the fly.

    Note: read-path slot detection requires extract_fact_slots() to
    recognize the slot type. The OSS extractor handles `employer`,
    `location`, etc. — production's broader extractor recognizes
    `user.name`, `user.favorite_color` etc. See ROADMAP for the slot
    extractor backport to OSS.
    """

    def test_grounding_surfaces_employer_slot_conflict(self, tmp_path):
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"),
                       enable_embeddings=False)
        # The OSS slot extractor recognizes "work at X" patterns
        s.add_memory("Nick works at Microsoft.", trust=0.9, source="user")
        # Draft asserts conflicting employer
        grounding = s.compute_grounding("Nick works at Google.")

        contradict = grounding.get("contradict", [])
        assert len(contradict) >= 1
        kinds = [c.get("kind") for c in contradict]
        # Either shape, slot_value_conflict, or any contradiction kind
        # is fine — the point is the conflict is surfaced.
        assert any(k for k in kinds), f"no contradiction kind: {contradict}"


# ==========================================================================
# Fix B — shape local-context gate
# ==========================================================================

class TestShapeLocalContextGate:
    """The exact false-positive that caused the v0.12 sanction REJECT."""

    def test_real_python_version_conflict_still_fires(self):
        # Both texts say "Python 3.x" — local context overlaps on
        # "python" and "3" tokens.
        r = shape(
            "This project requires Python 3.10 or higher.",
            "This project requires Python 3.8.",
        )
        assert r.score == 1.0
        assert len(r.evidence["conflicts"]) >= 1

    def test_real_test_count_conflict_still_fires(self):
        r = shape(
            "The aether-core test suite has 222 tests.",
            "The aether-core test suite has 99 tests.",
        )
        assert r.score == 1.0

    def test_real_iso_date_conflict_still_fires(self):
        r = shape(
            "aether-core v0.9.0 was released on 2026-04-27.",
            "aether-core v0.9.0 was released on 2025-01-15.",
        )
        assert r.score == 1.0

    def test_unrelated_co_topical_versions_suppressed(self):
        """The exact pattern that caused the sanction REJECT: action text
        mentions v0.12 and a Lab A v2 memory mentions v0.11. Same shape,
        unrelated discussions.
        """
        action_text = (
            "Action: ship v0.12 with slot-equality detector. Tests: add 30 lines."
        )
        memory_text = (
            "Lab A v2 found v0.11 detection layer was 0/42 on production "
            "substrate's real contradictions across 324 facts."
        )
        r = shape(action_text, memory_text)
        # Without the gate, shape would false-fire.
        # With the gate: low local-context overlap → no conflict.
        assert r.score < 1.0, (
            f"local-context gate should suppress this false positive: {r.evidence}"
        )

    def test_suppressed_entries_visible_in_evidence(self):
        """Debug visibility: rejected conflicts should still be reported
        in evidence under 'suppressed'."""
        action_text = "ship v0.12 today"
        memory_text = "v0.11 was last week's release"
        r = shape(action_text, memory_text)
        # Either suppressed entries or no conflicts — but 'suppressed'
        # field should always be present on the evidence dict
        suppressed = r.evidence.get("suppressed", None)
        assert suppressed is not None

    def test_local_context_tokens_overlap_threshold_constant(self):
        # The constant should be 0.30; future tuning should be deliberate
        assert LOCAL_CONTEXT_MIN_OVERLAP == 0.30


@needs_networkx
class TestShapeLocalContextWritePath:
    """Verify the gate prevents false-positive contradiction edges in
    add_memory."""

    def test_unrelated_co_topical_no_false_contradiction(self, tmp_path):
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"),
                       enable_embeddings=False)
        s.add_memory(
            "Lab A v2 found v0.11 detection layer was 0/42.",
            trust=0.9,
        )
        result = s.add_memory(
            "Action: ship v0.12 with slot-equality detector.",
            trust=0.85,
        )

        # No quantitative contradiction should be created here
        for f in result["tension_findings"]:
            assert f.get("kind") != "quantitative"


# ==========================================================================
# Cross-cutting: both fixes together
# ==========================================================================

@needs_networkx
class TestV012BothFixesCompose:
    def test_both_signals_fire_when_text_has_slots_and_shapes(self, tmp_path):
        """When the same memory pair has both a slot conflict AND a
        shape conflict, both should appear in the rule_trace.
        """
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"),
                       enable_embeddings=False)
        s.add_memory("User name is Nick. Project version 0.11.",
                     trust=0.9, slots={"user.name": "Nick"})
        result = s.add_memory("User name is Aether. Project version 0.12.",
                              trust=0.5, source="llm_stated",
                              slots={"user.name": "Aether"})

        findings = result["tension_findings"]
        assert len(findings) >= 1
        # The trace must mention slot_value_conflict (and may also
        # include shape:version since the two versions share local
        # context "project version").
        traces = [str(f.get("trace", "")) for f in findings]
        joined = " ".join(traces)
        assert "slot_value_conflict" in joined
