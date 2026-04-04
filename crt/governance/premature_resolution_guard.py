"""
PrematureResolutionGuard — Immune Agent for CRT Law 3

CRT Law 3:
    "Contradiction must be preserved before resolution."

Watches the contradiction handling pipeline and blocks resolution attempts
on contradictions that haven't been classified as RESOLVABLE. Contradictions
in HELD, EVOLVING, CONTEXTUAL, or UNKNOWN states carry information — resolving
them prematurely destroys that signal.

Threat model:
  - A contradiction exists between two claims in memory
  - A downstream process (merge, dedup, correction) tries to resolve it
  - Without this gate, the system collapses a genuine tension into a
    single claim, losing the information encoded in the disagreement
  - HELD contradictions become invisible; CONTEXTUAL distinctions vanish
  - The system becomes more "consistent" but less truthful

This agent breaks the loop by requiring a RESOLVABLE disposition with
sufficient confidence before any resolution action can proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Contradiction dispositions
# ---------------------------------------------------------------------------

class Disposition(str, Enum):
    RESOLVABLE = "RESOLVABLE"    # Evidence clearly favors one side
    HELD = "HELD"                # Genuine tension, both sides have support
    EVOLVING = "EVOLVING"        # Trajectory is changing, wait for more data
    CONTEXTUAL = "CONTEXTUAL"    # Both true in different contexts
    UNKNOWN = "UNKNOWN"          # Not yet classified


# ---------------------------------------------------------------------------
# Resolution actions
# ---------------------------------------------------------------------------

class ResolutionAction(str, Enum):
    RESOLVE_A = "RESOLVE_A"      # Keep claim A, demote claim B
    RESOLVE_B = "RESOLVE_B"      # Keep claim B, demote claim A
    MERGE = "MERGE"              # Combine into single statement
    DELETE_BOTH = "DELETE_BOTH"  # Remove both claims


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------

class VerdictAction(str, Enum):
    ALLOW = "ALLOW"   # Disposition is RESOLVABLE, resolution can proceed
    BLOCK = "BLOCK"   # Resolution would destroy information
    WARN = "WARN"     # Proceed with caution


@dataclass
class Verdict:
    """Result of a resolution guard check."""
    action: VerdictAction
    reason: str
    law: str = "Law 3: Contradiction must be preserved before resolution"
    disposition: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Contradiction representation
# ---------------------------------------------------------------------------

@dataclass
class Contradiction:
    """A pair of claims in tension with classification metadata."""
    id: str
    claim_a: str
    claim_b: str
    disposition: str = Disposition.UNKNOWN.value
    disposition_confidence: float = 0.0
    trust_a: float = 0.5
    trust_b: float = 0.5
    evidence_count_a: int = 0
    evidence_count_b: int = 0
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Disposition confidence below this is treated as UNKNOWN regardless of label
CONFIDENCE_FLOOR = 0.5

# Trust difference within this range triggers a WARN on RESOLVABLE
TRUST_PROXIMITY_THRESHOLD = 0.15

# Both claims must have trust below this for DELETE_BOTH to be allowed
DELETE_TRUST_CEILING = 0.1


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

class PrematureResolutionGuard:
    """
    Watches the contradiction resolution pipeline and blocks premature
    resolution attempts.

    Usage:
        guard = PrematureResolutionGuard()
        verdict = guard.check_resolution(contradiction, ResolutionAction.RESOLVE_A)
        if verdict.action == VerdictAction.BLOCK:
            # do not resolve — contradiction must be preserved
        elif verdict.action == VerdictAction.WARN:
            # log warning, proceed with caution
    """

    def __init__(
        self,
        confidence_floor: float = CONFIDENCE_FLOOR,
        trust_proximity: float = TRUST_PROXIMITY_THRESHOLD,
        delete_trust_ceiling: float = DELETE_TRUST_CEILING,
    ):
        self.confidence_floor = confidence_floor
        self.trust_proximity = trust_proximity
        self.delete_trust_ceiling = delete_trust_ceiling

    # -------------------------------------------------------------------
    # Core check
    # -------------------------------------------------------------------

    def check_resolution(
        self,
        contradiction: Contradiction,
        proposed_action: ResolutionAction,
    ) -> Verdict:
        """
        Check whether a proposed resolution action is safe to execute.

        Args:
            contradiction:    The contradiction being resolved.
            proposed_action:  The resolution action to check.

        Returns:
            Verdict with action (ALLOW/BLOCK/WARN) and reason.
        """
        disp = contradiction.disposition
        conf = contradiction.disposition_confidence

        # --- Override: low confidence forces UNKNOWN treatment ---
        effective_disp = disp
        if conf < self.confidence_floor:
            effective_disp = Disposition.UNKNOWN.value

        # --- DELETE_BOTH: always blocked unless both trust < ceiling ---
        if proposed_action == ResolutionAction.DELETE_BOTH:
            if (contradiction.trust_a < self.delete_trust_ceiling
                    and contradiction.trust_b < self.delete_trust_ceiling):
                return Verdict(
                    action=VerdictAction.ALLOW,
                    reason=(
                        f"DELETE_BOTH allowed: both claims have trust below "
                        f"{self.delete_trust_ceiling} "
                        f"(A={contradiction.trust_a:.2f}, B={contradiction.trust_b:.2f})."
                    ),
                    disposition=disp,
                    confidence=conf,
                )
            else:
                return Verdict(
                    action=VerdictAction.BLOCK,
                    reason=(
                        f"DELETE_BOTH blocked: at least one claim has trust >= "
                        f"{self.delete_trust_ceiling} "
                        f"(A={contradiction.trust_a:.2f}, B={contradiction.trust_b:.2f}). "
                        f"Deleting both would destroy information."
                    ),
                    disposition=disp,
                    confidence=conf,
                )

        # --- MERGE: blocked for HELD and CONTEXTUAL ---
        if proposed_action == ResolutionAction.MERGE:
            if effective_disp in (Disposition.HELD.value, Disposition.CONTEXTUAL.value):
                return Verdict(
                    action=VerdictAction.BLOCK,
                    reason=(
                        f"MERGE blocked for {effective_disp} contradiction. "
                        f"Merging would destroy the distinction between contexts "
                        f"or collapse a genuine tension."
                    ),
                    disposition=disp,
                    confidence=conf,
                )

        # --- Non-RESOLVABLE dispositions: BLOCK ---
        if effective_disp != Disposition.RESOLVABLE.value:
            return Verdict(
                action=VerdictAction.BLOCK,
                reason=(
                    f"Resolution blocked: disposition is {effective_disp} "
                    f"(confidence={conf:.2f}). "
                    f"Only RESOLVABLE contradictions may be resolved. "
                    f"This contradiction must be preserved."
                ),
                disposition=disp,
                confidence=conf,
            )

        # --- RESOLVABLE: check additional safety conditions ---

        # Check for zero evidence on either side
        if (contradiction.evidence_count_a == 0
                or contradiction.evidence_count_b == 0):
            return Verdict(
                action=VerdictAction.WARN,
                reason=(
                    f"RESOLVABLE but one side has zero evidence "
                    f"(A={contradiction.evidence_count_a}, "
                    f"B={contradiction.evidence_count_b}). "
                    f"Resolution may be based on recency, not evidence."
                ),
                disposition=disp,
                confidence=conf,
            )

        # Check for close trust scores
        trust_diff = abs(contradiction.trust_a - contradiction.trust_b)
        if trust_diff < self.trust_proximity:
            return Verdict(
                action=VerdictAction.WARN,
                reason=(
                    f"RESOLVABLE but trust scores are close "
                    f"(A={contradiction.trust_a:.2f}, B={contradiction.trust_b:.2f}, "
                    f"diff={trust_diff:.2f} < {self.trust_proximity}). "
                    f"Resolution might be premature."
                ),
                disposition=disp,
                confidence=conf,
            )

        # All checks pass — ALLOW
        return Verdict(
            action=VerdictAction.ALLOW,
            reason=(
                f"Resolution allowed: disposition is RESOLVABLE "
                f"(confidence={conf:.2f}), trust clearly favors one side "
                f"(A={contradiction.trust_a:.2f}, B={contradiction.trust_b:.2f})."
            ),
            disposition=disp,
            confidence=conf,
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PrematureResolutionGuard — Law 3 enforcement tests")
    print("=" * 70)

    guard = PrematureResolutionGuard()

    passed = 0
    total = 7

    # --- Test 1: HELD contradiction, attempt RESOLVE_A → BLOCK ---
    print("\nTest 1: HELD contradiction — RESOLVE_A should be BLOCKED")
    c1 = Contradiction(
        id="c001",
        claim_a="Free will exists as a fundamental property of consciousness.",
        claim_b="Free will is an illusion created by deterministic neural processes.",
        disposition=Disposition.HELD.value,
        disposition_confidence=0.85,
        trust_a=0.70,
        trust_b=0.65,
        evidence_count_a=5,
        evidence_count_b=4,
    )
    v1 = guard.check_resolution(c1, ResolutionAction.RESOLVE_A)
    print(f"  Verdict: {v1.action.value}")
    print(f"  Reason:  {v1.reason}")
    print(f"  Law:     {v1.law}")
    assert v1.action == VerdictAction.BLOCK, \
        f"Expected BLOCK, got {v1.action}"
    print("  PASSED")
    passed += 1

    # --- Test 2: RESOLVABLE with high confidence → ALLOW ---
    print("\nTest 2: RESOLVABLE with high confidence — should ALLOW")
    c2 = Contradiction(
        id="c002",
        claim_a="The Earth orbits the Sun.",
        claim_b="The Sun orbits the Earth.",
        disposition=Disposition.RESOLVABLE.value,
        disposition_confidence=0.95,
        trust_a=0.95,
        trust_b=0.10,
        evidence_count_a=100,
        evidence_count_b=1,
    )
    v2 = guard.check_resolution(c2, ResolutionAction.RESOLVE_A)
    print(f"  Verdict: {v2.action.value}")
    print(f"  Reason:  {v2.reason}")
    assert v2.action == VerdictAction.ALLOW, \
        f"Expected ALLOW, got {v2.action}"
    print("  PASSED")
    passed += 1

    # --- Test 3: RESOLVABLE but trust scores are close → WARN ---
    print("\nTest 3: RESOLVABLE but close trust scores — should WARN")
    c3 = Contradiction(
        id="c003",
        claim_a="Python is the best language for data science.",
        claim_b="R is the best language for data science.",
        disposition=Disposition.RESOLVABLE.value,
        disposition_confidence=0.80,
        trust_a=0.72,
        trust_b=0.68,
        evidence_count_a=10,
        evidence_count_b=8,
    )
    v3 = guard.check_resolution(c3, ResolutionAction.RESOLVE_A)
    print(f"  Verdict: {v3.action.value}")
    print(f"  Reason:  {v3.reason}")
    assert v3.action == VerdictAction.WARN, \
        f"Expected WARN, got {v3.action}"
    print("  PASSED")
    passed += 1

    # --- Test 4: DELETE_BOTH — BLOCK unless both trust < 0.1 ---
    print("\nTest 4a: DELETE_BOTH with normal trust — should BLOCK")
    c4a = Contradiction(
        id="c004a",
        claim_a="Outdated claim A.",
        claim_b="Outdated claim B.",
        disposition=Disposition.RESOLVABLE.value,
        disposition_confidence=0.90,
        trust_a=0.50,
        trust_b=0.40,
        evidence_count_a=3,
        evidence_count_b=2,
    )
    v4a = guard.check_resolution(c4a, ResolutionAction.DELETE_BOTH)
    print(f"  Verdict: {v4a.action.value}")
    print(f"  Reason:  {v4a.reason}")
    assert v4a.action == VerdictAction.BLOCK, \
        f"Expected BLOCK, got {v4a.action}"
    print("  PASSED")

    print("\nTest 4b: DELETE_BOTH with both trust < 0.1 — should ALLOW")
    c4b = Contradiction(
        id="c004b",
        claim_a="Discredited claim X.",
        claim_b="Discredited claim Y.",
        disposition=Disposition.RESOLVABLE.value,
        disposition_confidence=0.90,
        trust_a=0.05,
        trust_b=0.03,
        evidence_count_a=1,
        evidence_count_b=1,
    )
    v4b = guard.check_resolution(c4b, ResolutionAction.DELETE_BOTH)
    print(f"  Verdict: {v4b.action.value}")
    print(f"  Reason:  {v4b.reason}")
    assert v4b.action == VerdictAction.ALLOW, \
        f"Expected ALLOW, got {v4b.action}"
    print("  PASSED")
    passed += 1

    # --- Test 5: UNKNOWN disposition → BLOCK ---
    print("\nTest 5: UNKNOWN disposition — should BLOCK")
    c5 = Contradiction(
        id="c005",
        claim_a="Claim with no analysis yet.",
        claim_b="Contradicting claim with no analysis.",
        disposition=Disposition.UNKNOWN.value,
        disposition_confidence=0.0,
        trust_a=0.60,
        trust_b=0.55,
        evidence_count_a=3,
        evidence_count_b=2,
    )
    v5 = guard.check_resolution(c5, ResolutionAction.RESOLVE_B)
    print(f"  Verdict: {v5.action.value}")
    print(f"  Reason:  {v5.reason}")
    assert v5.action == VerdictAction.BLOCK, \
        f"Expected BLOCK, got {v5.action}"
    print("  PASSED")
    passed += 1

    # --- Test 6: CONTEXTUAL, attempt MERGE → BLOCK ---
    print("\nTest 6: CONTEXTUAL contradiction — MERGE should be BLOCKED")
    c6 = Contradiction(
        id="c006",
        claim_a="Async is better for I/O-bound workloads.",
        claim_b="Threads are better for CPU-bound workloads.",
        disposition=Disposition.CONTEXTUAL.value,
        disposition_confidence=0.90,
        trust_a=0.85,
        trust_b=0.85,
        evidence_count_a=20,
        evidence_count_b=20,
    )
    v6 = guard.check_resolution(c6, ResolutionAction.MERGE)
    print(f"  Verdict: {v6.action.value}")
    print(f"  Reason:  {v6.reason}")
    assert v6.action == VerdictAction.BLOCK, \
        f"Expected BLOCK, got {v6.action}"
    print("  PASSED")
    passed += 1

    # --- Test 7: Low disposition_confidence overrides RESOLVABLE → BLOCK ---
    print("\nTest 7: RESOLVABLE but low confidence — treated as UNKNOWN, BLOCK")
    c7 = Contradiction(
        id="c007",
        claim_a="The meeting is on Tuesday.",
        claim_b="The meeting is on Wednesday.",
        disposition=Disposition.RESOLVABLE.value,
        disposition_confidence=0.30,  # below 0.5 floor
        trust_a=0.80,
        trust_b=0.40,
        evidence_count_a=5,
        evidence_count_b=2,
    )
    v7 = guard.check_resolution(c7, ResolutionAction.RESOLVE_A)
    print(f"  Verdict: {v7.action.value}")
    print(f"  Reason:  {v7.reason}")
    assert v7.action == VerdictAction.BLOCK, \
        f"Expected BLOCK, got {v7.action}"
    print("  PASSED")
    passed += 1

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("Law 3 enforcement is operational.")
    else:
        print("FAILURES DETECTED — review above.")
    print("=" * 70)
