"""v0.12.2 — extended policy contradiction detection.

F#4 (caught by the e2e harness): aether_sanction APPROVE'd an action
that contradicted a high-trust prohibition belief because the cue
cross-check missed real CLI forms (`--force`, `-f origin`) and the
sim-gate (0.45) was unreachable in cold-encoder mode.

Fix:
- IMPERATIVE_CUES extended to cover real CLI forms.
- POLICY_CONTRA_STRONG_TRUST = 0.85 — when belief trust crosses this,
  the sim-gate is bypassed and cue alignment is sufficient on its own.
- Two inlined call sites (write-time + read-time) collapsed into a
  single `_is_policy_contradiction()` helper so future tweaks land in
  one place.
"""

from __future__ import annotations

from aether.mcp.state import (
    POLICY_CONTRA_MIN_SIMILARITY,
    POLICY_CONTRA_MIN_TRUST,
    POLICY_CONTRA_STRONG_TRUST,
    _is_policy_contradiction,
    _looks_like_imperative,
    _looks_like_prohibition,
)


class TestImperativeCues:
    """Cue list now covers real CLI forms, not just human-text phrasing."""

    def test_dash_dash_force_caught(self):
        assert _looks_like_imperative("git push --force origin main")

    def test_short_f_flag_caught(self):
        assert _looks_like_imperative("git push -f origin main")

    def test_push_dash_dash_force_caught(self):
        assert _looks_like_imperative("push --force origin main")

    def test_existing_human_text_still_caught(self):
        # Regression: pre-v0.12.2 cues still fire.
        assert _looks_like_imperative("force push to main")
        assert _looks_like_imperative("rm -rf /")
        assert _looks_like_imperative("drop database users")

    def test_benign_text_does_not_match(self):
        assert not _looks_like_imperative("the weather is nice today")
        assert not _looks_like_imperative("I prefer python over ruby")


class TestPolicyContradictionHelper:
    """The `_is_policy_contradiction` helper.

    Both call sites (write-time + read-time) route through this. Cases
    below cover the F#4 fix (strong-trust path) plus the standard
    high-overlap path that already worked.
    """

    BELIEF = "Never force-push to the main branch."
    ACTION = "git push --force origin main"

    def test_strong_trust_bypasses_sim_gate(self):
        """F#4: high trust + cue alignment fires even at low similarity."""
        assert _is_policy_contradiction(
            mem_text=self.BELIEF,
            new_text=self.ACTION,
            sim=0.10,  # cold-encoder Jaccard, below the 0.45 standard gate
            mem_trust=0.95,
        )

    def test_standard_path_still_fires(self):
        """Pre-v0.12.2 high-overlap path is unchanged."""
        assert _is_policy_contradiction(
            mem_text=self.BELIEF,
            new_text=self.ACTION,
            sim=POLICY_CONTRA_MIN_SIMILARITY + 0.01,
            mem_trust=POLICY_CONTRA_MIN_TRUST + 0.01,
        )

    def test_low_trust_does_not_fire_even_with_cues(self):
        """Trust gate prevents a stale or low-confidence belief from
        blocking an imperative just because cues happen to align."""
        assert not _is_policy_contradiction(
            mem_text=self.BELIEF,
            new_text=self.ACTION,
            sim=0.10,
            mem_trust=POLICY_CONTRA_MIN_TRUST - 0.01,
        )

    def test_strong_trust_threshold_is_a_hard_edge(self):
        """At STRONG_TRUST - epsilon, sim gate still applies."""
        # Cue-aligned + sim below standard gate + trust just below STRONG.
        # Should NOT fire — the strong-trust bypass is gated on >= 0.85.
        assert not _is_policy_contradiction(
            mem_text=self.BELIEF,
            new_text=self.ACTION,
            sim=POLICY_CONTRA_MIN_SIMILARITY - 0.05,
            mem_trust=POLICY_CONTRA_STRONG_TRUST - 0.01,
        )

    def test_misaligned_cues_do_not_fire(self):
        """Both texts being prohibitions is not a contradiction."""
        assert not _is_policy_contradiction(
            mem_text="Never force-push to main.",
            new_text="Don't force-push to any branch.",
            sim=0.9,
            mem_trust=0.95,
        )

    def test_two_imperatives_do_not_fire(self):
        """Two imperatives don't contradict each other on this channel
        (might be redundant or independent commands)."""
        assert not _is_policy_contradiction(
            mem_text="rm -rf /tmp/cache",
            new_text="git push --force origin main",
            sim=0.5,
            mem_trust=0.95,
        )

    def test_unrelated_texts_do_not_fire(self):
        """No cues on either side -> no contradiction."""
        assert not _is_policy_contradiction(
            mem_text="The user's name is Nick.",
            new_text="Run the test suite once.",
            sim=0.5,
            mem_trust=0.95,
        )
