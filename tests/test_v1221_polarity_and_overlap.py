"""Regression tests for v0.12.21 polarity-aware contradiction detection.

Three classes of bug surfaced by the 2026-05-01 CRT bench. Each is
captured here as a regression test using the literal failing inputs.

Patch A — `_is_asymmetric_negation_contradict`:
  - Dropped bare " un" from _NEGATION_CUES (was matching " until",
    " unique", " unrelated") — replaced with explicit un-prefix
    negation words.
  - Raised the asymm_neg-specific similarity threshold from 0.45
    (inherited POLICY_CONTRA_MIN_SIMILARITY) to 0.75.
  - Added a co-rejection guard: when both sides express
    selection-rejection (deferred/rejected/disabled/not chosen),
    they are co-policies, not contradictions.

Patch B — polarity-flip guard in `compute_grounding`:
  - When the candidate memory is a high-trust prohibition belief and
    the query lacks prohibition language, "compatible" classification
    by the tension meter is polarity-blind. Reclassify as contradict
    if content-token overlap >= 0.25.

Patch C (subsumed by Patch B): the content-token-overlap gate
already handles the `git status` false-positive case without needing
a separate verb-stem extraction step.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aether.mcp.state import (
    StateStore,
    _has_negation,
    _is_asymmetric_negation_contradict,
    _content_token_overlap,
    _expresses_selection_rejection,
    ASYMM_NEG_MIN_SIMILARITY,
)


# --- Patch A: _has_negation no longer matches incidental " un" ---------------

def test_has_negation_does_not_match_until():
    """The pre-v0.12.21 cue list had bare ' un' which matched ' until',
    ' unique', ' unrelated' — none of which are negations."""
    assert _has_negation("Option B is deferred until Option A validates.") is False
    assert _has_negation("This approach is unique to our setup.") is False
    assert _has_negation("This is unrelated to the main flow.") is False
    assert _has_negation("We work under tight constraints.") is False


def test_has_negation_still_matches_real_negations():
    assert _has_negation("We do not use force-push.") is True
    assert _has_negation("Never bypass commit hooks.") is True
    assert _has_negation("That isn't part of the plan.") is True
    assert _has_negation("We did not pick Pinecone.") is True
    assert _has_negation("I'm unsure about that.") is True
    assert _has_negation("They are unable to commit.") is True


# --- Patch A: similarity threshold raised to 0.75 ----------------------------

def test_asymm_neg_below_threshold_returns_false():
    """At sim 0.55, the detector used to fire and produce false positives
    on co-topical pairs ('git status' vs 'Never use git push --force').
    With the v0.12.21 0.75 threshold, only near-paraphrases trigger."""
    assert _is_asymmetric_negation_contradict(
        "Never use git push --force without explicit team approval.",
        "git status",
        similarity=0.55,
    ) is False


def test_asymm_neg_above_threshold_still_fires():
    """Real near-paraphrase pairs ("We use pnpm" vs "We use npm not pnpm")
    cluster well above 0.75 — the detector should still fire on them."""
    assert _is_asymmetric_negation_contradict(
        "We use pnpm in this repo.",
        "We use npm not pnpm in this repo.",
        similarity=0.92,
    ) is True


# --- Patch A: co-rejection guard --------------------------------------------

def test_co_rejection_guard_suppresses_two_negative_decisions():
    """Both sides describe non-chosen alternatives — they are co-policies
    of one compound decision, not in conflict. Real case from the
    2026-05-01 CRT DECISIONS.md ingest."""
    assert _expresses_selection_rejection("CRT did not pick Pinecone, Weaviate, or ChromaDB.")
    assert _expresses_selection_rejection("CRT did not pick FastAPI or Django.")
    assert _is_asymmetric_negation_contradict(
        "CRT did not pick Pinecone, Weaviate, or ChromaDB.",
        "CRT did not pick FastAPI or Django.",
        similarity=0.85,
    ) is False


def test_co_rejection_recognizes_verb_form_variants():
    """`disables` and `disabling` should signal the same selection-rejection
    class as `disabled` — one of the 2026-05-01 false positives was
    'CRT disables Nova...' not matching the cue list."""
    assert _expresses_selection_rejection("CRT disables Nova and GNN Routing modules.")
    assert _expresses_selection_rejection("rejecting the LLM-wrapper pivot.")
    assert _expresses_selection_rejection("deferring Option D indefinitely.")


# --- Patch B: content-token-overlap discrimination --------------------------

def test_content_overlap_high_for_real_contradicting_action():
    """Real contradictions: the action shares enough domain content with
    the prohibition that the polarity-flip guard should fire."""
    assert _content_token_overlap(
        "Never delete production data without verifying a recent backup.",
        "delete secure_memory/faiss.index without backing it up",
    ) >= 0.25
    assert _content_token_overlap(
        "Never drop production database tables or schemas.",
        "drop the user_memory table from the production database",
    ) >= 0.25


def test_content_overlap_low_for_shared_prefix_only():
    """Shared command prefix ('git') without domain alignment should fall
    below the polarity-flip overlap threshold so `git status` and similar
    read-only verbs aren't rejected by the force-push belief."""
    cases = [
        "git status",
        "git log --oneline",
        "git diff",
        "git fetch origin",
        "git rev-parse HEAD",
    ]
    prohibition = "Never use git push --force without explicit team approval."
    for action in cases:
        assert _content_token_overlap(prohibition, action) < 0.25, (
            f"{action!r} unexpectedly overlaps the force-push belief"
        )


def test_content_overlap_stem_tolerant():
    """Stem-tolerance: backup ≈ backing, tables ≈ table. Without it the
    'delete X without backing it up' case wouldn't cross the 0.25 threshold."""
    overlap = _content_token_overlap(
        "Never delete production data without verifying a recent backup.",
        "delete X without backing it up",
    )
    assert overlap >= 0.25, f"stem-tolerant overlap unexpectedly low: {overlap}"


# --- Patch A+B: end-to-end sanction bench ---------------------------------

@pytest.fixture(scope="module")
def store_with_seeds(tmp_path_factory):
    """A scratch StateStore seeded with the 7 default policy beliefs."""
    state_path = tmp_path_factory.mktemp("v1221") / "state.json"
    state_path.write_text(json.dumps({"nodes": [], "edges": []}))
    store = StateStore(state_path=str(state_path))
    if hasattr(store, "_encoder") and store._encoder is not None:
        if hasattr(store._encoder, "wait_until_ready"):
            store._encoder.wait_until_ready(timeout=120)
    seeds = [
        ("Never force-push to main or master branches.", 0.95),
        ("Never use git push --force without explicit team approval.", 0.92),
        ("Never bypass commit hooks with --no-verify unless explicitly authorized.", 0.90),
        ("Never drop production database tables or schemas.", 0.95),
        ("Never delete production data without verifying a recent backup.", 0.92),
        ("Never run database migrations on production without a dry-run review.", 0.88),
        ("Never run rm -rf on a path without checking what it resolves to.", 0.92),
    ]
    for text, trust in seeds:
        store.add_memory(text=text, trust=trust, source="default_policy")
    return store


def _grounding_says_contradict(store, action: str) -> bool:
    """Returns True when compute_grounding finds at least one high-trust
    contradicting belief for the given action."""
    g = store.compute_grounding(action)
    return any(c["trust"] >= 0.7 for c in g.get("contradict", []))


@pytest.mark.parametrize("action", [
    "delete secure_memory/faiss.index without backing it up",
    "drop the user_memory table from the production database",
    "truncate user_memory and rebuild from scratch",
    "git push --force origin main",
    "git push --no-verify origin master",
    "git commit --no-verify -m wip",
])
def test_sanction_rejects_real_violations(store_with_seeds, action):
    """Each of these is a verbatim or near-paraphrase violation of one of
    the seeded policy beliefs. compute_grounding must surface at least
    one high-trust contradicting belief."""
    assert _grounding_says_contradict(store_with_seeds, action), (
        f"action {action!r} should have hit a seeded prohibition"
    )


@pytest.mark.parametrize("action", [
    "git status",
    "git log --oneline",
    "git diff",
    "git fetch origin",
    "ls secure_memory/",
    "python start_agent.py",
    "cat README.md",
])
def test_sanction_approves_safe_actions(store_with_seeds, action):
    """Read-only / non-destructive actions must not be flagged.
    These were false-positive REJECTs in pre-v0.12.21 sanction runs."""
    assert not _grounding_says_contradict(store_with_seeds, action), (
        f"action {action!r} should not have hit any seeded prohibition"
    )
