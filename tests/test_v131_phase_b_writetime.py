"""v0.13.1 Phase B — slot canonicalization at the write-time contradiction cascade.

The 2026-05-01 verify probe (`bench/phase_b_verify.py`) confirmed the
handoff hypothesis from `HANDOFF_2026-05-01_to_windows.md`:

  > Phase B might be smaller than originally scoped. The slot tags
  > already flow through to write time naturally because `add_memory`
  > already calls `extract_fact_slots`.

Translation: every slot Phase A added at the read side automatically
flows to the write side because `_slot_equality_match` was already in
the cascade. Test #3 Cases A and C from the reflexive bench passed
without any code change — the existing detector picked up Phase A's
new `project_embedding_dim` slot at write time and fired
`slot_value_conflict` edges with disposition=resolvable.

The one case that didn't work was Test #3 Case B — paraphrased
decision drift across "Option A" / "Option B" prose. Neither memory
extracted any slots, so the cascade had nothing to fire on. Phase B
adds a single extractor (`project_chosen_option`) that captures the
option letter from prose like "we picked Option A" / "pivoted to
Option B" / "the team went with Option C". With the new extractor in
place, the existing write-time cascade flags the conflict as
`slot_value_conflict:project_chosen_option:A<>B`.

These tests pin both contracts:

  1. The new extractor produces a `project_chosen_option` slot on the
     prose patterns that show up in real decision logs.
  2. Cases A, B, and C from the reflexive bench all produce CONTRADICTS
     edges at *write* time (not just read time) when the contradicting
     memories pass through `add_memory` in sequence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("networkx")

from aether.memory import extract_fact_slots
from aether.mcp.state import StateStore


# --------------------------------------------------------------------------
# project_chosen_option extractor
# --------------------------------------------------------------------------

class TestProjectChosenOptionExtractor:
    def test_picked_option_a(self):
        facts = extract_fact_slots("CRT picked Option A")
        assert "project_chosen_option" in facts
        assert facts["project_chosen_option"].normalized == "A"

    def test_pivoted_to_option_b(self):
        facts = extract_fact_slots("we pivoted to Option B")
        assert "project_chosen_option" in facts
        assert facts["project_chosen_option"].normalized == "B"

    def test_team_went_with_option_c(self):
        facts = extract_fact_slots("the team went with Option C")
        assert "project_chosen_option" in facts
        assert facts["project_chosen_option"].normalized == "C"

    def test_decision_verbs_all_fire(self):
        for verb in ("picked", "chose", "selected", "decided on", "opted for"):
            facts = extract_fact_slots(f"the project {verb} Option D")
            assert "project_chosen_option" in facts, (
                f"verb {verb!r} did not match the extractor"
            )
            assert facts["project_chosen_option"].normalized == "D"

    def test_normalizes_to_uppercase(self):
        facts = extract_fact_slots("we chose option a")
        assert "project_chosen_option" in facts
        assert facts["project_chosen_option"].normalized == "A"

    def test_does_not_fire_on_non_decision_prose(self):
        # `Option` mentioned without a decision verb should not match.
        facts = extract_fact_slots("Option A is faster than Option B")
        assert "project_chosen_option" not in facts

    def test_full_case_b_prose(self):
        # The exact text from Test #3 Case B.
        facts = extract_fact_slots(
            "CRT picked Option A: Remember Me as the primary path."
        )
        assert "project_chosen_option" in facts
        assert facts["project_chosen_option"].normalized == "A"


# --------------------------------------------------------------------------
# Write-time cascade: slots flow through add_memory automatically
# --------------------------------------------------------------------------

def _contradicts_edges(store, mem_a, mem_b):
    """Walk store.graph for any contradicts edge between A and B."""
    edges = []
    for src, tgt, data in store.graph.graph.edges(data=True):
        if {src, tgt} != {mem_a, mem_b}:
            continue
        if data.get("edge_type") == "contradicts":
            edges.append(data)
    return edges


def _fresh_store(tmp_path: Path) -> StateStore:
    s = StateStore(state_path=str(tmp_path / "state.json"))
    if s._encoder is not None and hasattr(s._encoder, "_load"):
        # Sync-load the encoder so the meter classifies in warm mode.
        s._encoder._load()
    return s


class TestPhaseBWriteCascade:
    """Test #3 from the 2026-05-01 reflexive bench, run through
    add_memory rather than compute_grounding. The Mac session
    confirmed read-side; this confirms write-side."""

    def test_case_a_paraphrased_numeric_dim(self, tmp_path):
        store = _fresh_store(tmp_path)
        a = store.add_memory(
            text="CRT's vector dimension is 768.",
            trust=0.9, source="case_test",
        )
        b = store.add_memory(
            text="CRT uses 384-dim embeddings now.",
            trust=0.9, source="case_test",
        )
        edges = _contradicts_edges(store, a["memory_id"], b["memory_id"])
        assert len(edges) >= 1, (
            "Case A: expected slot_value_conflict on project_embedding_dim, "
            "got no contradicts edge"
        )
        traces = [str(e.get("rule_trace", [])) for e in edges]
        assert any("project_embedding_dim" in t for t in traces), (
            f"Edge fired but rule_trace doesn't reference the embedding-dim slot: "
            f"{traces}"
        )

    def test_case_b_paraphrased_decision_drift(self, tmp_path):
        store = _fresh_store(tmp_path)
        a = store.add_memory(
            text="CRT picked Option A: Remember Me as the primary path.",
            trust=0.9, source="case_test",
        )
        b = store.add_memory(
            text="CRT pivoted to Option B: structured dev tools.",
            trust=0.9, source="case_test",
        )
        edges = _contradicts_edges(store, a["memory_id"], b["memory_id"])
        assert len(edges) >= 1, (
            "Case B: expected slot_value_conflict on project_chosen_option, "
            "got no contradicts edge — Phase B extractor isn't firing or "
            "the write-time cascade isn't picking it up"
        )
        traces = [str(e.get("rule_trace", [])) for e in edges]
        assert any("project_chosen_option" in t for t in traces), (
            f"Edge fired but didn't trace through the chosen-option slot: "
            f"{traces}"
        )

    def test_case_c_template_identical_numeric(self, tmp_path):
        """Already passed pre-Phase-B via the shape detector. This pins
        that we didn't regress it."""
        store = _fresh_store(tmp_path)
        a = store.add_memory(
            text="CRT's launch timeline is 10 weeks.",
            trust=0.9, source="case_test",
        )
        b = store.add_memory(
            text="CRT's launch timeline is 6 weeks.",
            trust=0.9, source="case_test",
        )
        edges = _contradicts_edges(store, a["memory_id"], b["memory_id"])
        assert len(edges) >= 1, "Case C regression: shape detector stopped firing"


class TestNoFalsePositiveOnAlignedDecisions:
    """Two memories that both say 'Option B' should NOT contradict —
    same value on the same slot is alignment, not conflict."""

    def test_same_option_letter_no_contradiction(self, tmp_path):
        store = _fresh_store(tmp_path)
        a = store.add_memory(
            text="CRT picked Option B as the path forward.",
            trust=0.9, source="case_test",
        )
        b = store.add_memory(
            text="The team went with Option B for the rebuild.",
            trust=0.9, source="case_test",
        )
        edges = _contradicts_edges(store, a["memory_id"], b["memory_id"])
        assert edges == [], (
            "Same option letter on the same slot is agreement, not conflict. "
            f"Got {len(edges)} false-positive edge(s)."
        )
