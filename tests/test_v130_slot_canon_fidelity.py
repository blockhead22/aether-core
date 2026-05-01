"""v0.13.0 Phase A — slot-canonical fidelity grading.

Regression tests for the bench cases that exposed `aether_fidelity` as
polarity-blind on non-prohibition memories (2026-05-01 validation pass):

  * Draft "Nick is a chef in Paris" returned belief_conf=0.95 against a
    substrate where no memory grounded "chef" or "Paris" — the structural
    tension meter classified an unrelated "Nick is the maintainer of
    aether-core" memory as supporting (compatible — both could be true,
    so the meter said yes).
  * Draft "Nick uses emacs primarily" returned belief_conf=0.90 against
    a substrate with "Nick prefers vim over emacs for editing" — same
    polarity-blindness on a non-prohibition memory.

Phase A's fix was not a single new check in `compute_grounding`. It was
adding extractors for slots that the existing slot_equality_match code
needs to fire on — third-person occupation/location/employer/editor and
project-level vector_store/framework/embedding_dim. Once both substrate
memories and drafts produce the same slot:k=v tags, the existing slot
conflict detection fires on its own.

Phase B will wire slot canonicalization into the write-time contradiction
cascade. Phase A is read-side only (compute_grounding).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aether.memory import extract_fact_slots
from aether.mcp.state import StateStore


# --- Slot extractor coverage (the new third-person + project slots) ---------


def test_editor_slot_first_person():
    facts = extract_fact_slots("I prefer vim over emacs for editing.")
    assert "editor" in facts
    assert facts["editor"].normalized == "vim"


def test_editor_slot_third_person_uses():
    facts = extract_fact_slots("Nick uses emacs primarily.")
    assert "editor" in facts
    assert facts["editor"].normalized == "emacs"


def test_editor_slot_third_person_prefers():
    """The 'prefers' verb form must extract too — the bench case used it."""
    facts = extract_fact_slots("Nick prefers vim over emacs for editing.")
    assert "editor" in facts
    assert facts["editor"].normalized == "vim"


def test_third_person_occupation_simple():
    facts = extract_fact_slots("Nick is a chef in Paris.")
    assert "occupation" in facts
    assert facts["occupation"].normalized == "chef"
    assert "location" in facts
    assert facts["location"].normalized == "paris"


def test_third_person_occupation_with_complement():
    facts = extract_fact_slots("Nick is the maintainer of aether-core.")
    assert "occupation" in facts
    assert facts["occupation"].normalized == "maintainer"


def test_third_person_employer_does_not_pollute_location():
    """Pre-fix the third-person location regex matched 'works at X' and
    captured the workplace as a location. After narrowing the regex to
    'lives|is|is based|resides in', employer-only sentences emit only
    the employer slot."""
    facts = extract_fact_slots("Nick works at Anthropic on Claude.")
    assert "employer" in facts
    assert "anthropic" in facts["employer"].normalized
    assert "location" not in facts


def test_project_vector_store():
    facts = extract_fact_slots("CRT uses FAISS as the primary vector store.")
    assert "project_vector_store" in facts
    assert facts["project_vector_store"].normalized == "faiss"


def test_project_framework():
    facts = extract_fact_slots("CRT uses Flask with CORS enabled for the web API.")
    assert "project_framework" in facts
    assert facts["project_framework"].normalized == "flask"


def test_project_embedding_dim_predicate():
    facts = extract_fact_slots("The vector dimension is 768.")
    assert "project_embedding_dim" in facts
    assert facts["project_embedding_dim"].normalized == "768"


def test_project_embedding_dim_compound():
    facts = extract_fact_slots("CRT uses 384-dim embeddings.")
    assert "project_embedding_dim" in facts
    assert facts["project_embedding_dim"].normalized == "384"


def test_negative_controls_extract_nothing():
    """Statements without slot signal must not produce spurious slots."""
    for s in [
        "This is great.",
        "It works fine.",
        "Nick is happy today.",  # 'happy' is in the occupation blocklist
        "The weather is nice.",
    ]:
        facts = extract_fact_slots(s)
        assert "occupation" not in facts, f"unexpected occupation from {s!r}"
        assert "location" not in facts, f"unexpected location from {s!r}"
        assert "editor" not in facts, f"unexpected editor from {s!r}"


# --- End-to-end fidelity grading on the bench cases -------------------------


@pytest.fixture(scope="module")
def store_with_bench_seeds(tmp_path_factory):
    """A scratch StateStore seeded with the validation-bench memories.
    Uses tmp_path_factory so the fixture is module-scoped (encoder warmup
    runs once)."""
    state_path = tmp_path_factory.mktemp("v130") / "state.json"
    state_path.write_text(json.dumps({"nodes": [], "edges": []}))
    store = StateStore(state_path=str(state_path))
    if hasattr(store, "_encoder") and store._encoder is not None:
        if hasattr(store._encoder, "wait_until_ready"):
            store._encoder.wait_until_ready(timeout=120)
    store.add_memory(
        "Nick prefers vim over emacs for editing.",
        trust=0.85, source="user_preference",
    )
    store.add_memory(
        "The CRT codebase uses FAISS for vector search.",
        trust=0.85, source="project_fact",
    )
    store.add_memory(
        "Nick is the maintainer of aether-core.",
        trust=0.95, source="user_identity",
    )
    return store


def test_fidelity_supported_draft_high_confidence(store_with_bench_seeds):
    """A draft that paraphrases an existing memory should ground high."""
    g = store_with_bench_seeds.compute_grounding("Nick maintains aether-core.")
    assert g["belief_confidence"] >= 0.7, g
    assert len(g["support"]) >= 1


def test_fidelity_chef_in_paris_low_confidence(store_with_bench_seeds):
    """The headline pre-Phase-A failure: a hallucinated draft about Nick
    being a chef in Paris used to return belief_conf=0.95. After Phase A
    the slot:occupation=chef vs slot:occupation=maintainer conflict fires."""
    g = store_with_bench_seeds.compute_grounding("Nick is a chef in Paris.")
    assert g["belief_confidence"] < 0.4, (
        f"chef-in-Paris should not ground high; got {g['belief_confidence']:.2f}\n"
        f"support={g['support']}\ncontradict={g['contradict']}"
    )


def test_fidelity_emacs_contradicts_substrate_vim(store_with_bench_seeds):
    """Substrate has slot:editor=vim. A draft with slot:editor=emacs must
    surface as a contradiction — not as support."""
    g = store_with_bench_seeds.compute_grounding("Nick uses emacs primarily.")
    assert len(g["contradict"]) >= 1, (
        f"emacs draft should contradict the vim memory; got {g}\n"
    )


def test_fidelity_pinecone_contradicts_substrate_faiss(store_with_bench_seeds):
    """Substrate has slot:project_vector_store=faiss. A draft saying CRT
    uses Pinecone for vector search must contradict."""
    g = store_with_bench_seeds.compute_grounding(
        "CRT uses Pinecone for vector search."
    )
    assert len(g["contradict"]) >= 1


def test_fidelity_unrelated_draft_neutral(store_with_bench_seeds):
    """A draft about a topic the substrate has no information on (and no
    slot conflict against) should land at the empty/neutral baseline,
    not high confidence."""
    g = store_with_bench_seeds.compute_grounding("The weather is nice today.")
    assert g["belief_confidence"] < 0.5, g
