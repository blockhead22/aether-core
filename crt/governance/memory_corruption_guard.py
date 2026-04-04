"""
MemoryCorruptionGuard — Immune Agent for CRT Law 4

CRT Law 4:
    "Degraded reconstruction cannot silently overwrite trusted memory."

Watches the memory update/overwrite path and prevents reconstructions with
lower fidelity from replacing memories with higher trust. This is critical
during compression/decompression cycles, consolidation passes, and any
pipeline that rewrites memory content.

Threat model:
  - A memory is stored at trust 0.9 from user input
  - System runs a compression pass (encode -> quantize -> decode)
  - Reconstructed memory has fidelity 0.6 due to lossy compression
  - Without this gate, the degraded version silently overwrites the original
  - Future retrievals return the corrupted version as if it were trusted
  - Epistemic rot: the system's beliefs degrade invisibly over time

This agent breaks the rot by requiring fidelity >= trust for overwrites,
and by enforcing source-hierarchy rules on all memory mutations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------

class OverwriteAction(Enum):
    ALLOW = "ALLOW"         # Replacement is safe
    BLOCK = "BLOCK"         # Replacement would degrade the memory
    DOWNGRADE = "DOWNGRADE" # Allow the write but reduce trust proportionally


class OverwriteReason(Enum):
    RECONSTRUCTION = "RECONSTRUCTION"   # Model reconstructed (compression/decompression)
    USER_CORRECTION = "USER_CORRECTION" # User explicitly corrected
    NEW_EVIDENCE = "NEW_EVIDENCE"       # New information updates the memory
    CONSOLIDATION = "CONSOLIDATION"     # System merging/summarizing memories
    DECAY = "DECAY"                     # Time-based memory refresh


@dataclass
class OverwriteVerdict:
    """Result of a memory corruption guard check."""
    action: OverwriteAction
    reason: str
    law: str = "Law 4: Degraded reconstruction cannot silently overwrite trusted memory"
    trust_delta: float = 0.0          # Change in trust that would result
    fidelity_gap: float = 0.0         # existing_trust - proposed_fidelity
    content_similarity: Optional[float] = None  # Cosine sim if embeddings available


# ---------------------------------------------------------------------------
# Memory representation
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    """A stored memory with trust, fidelity, and source metadata."""
    id: str
    content: str
    trust: float                          # 0.0 - 1.0, current trust score
    source: str                           # "user", "retrieved", "generated", "reconstructed"
    fidelity: float                       # 0.0 - 1.0, how accurately this represents original
    version: int = 1                      # How many times this memory has been updated
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    embedding: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

class MemoryCorruptionGuard:
    """
    Watches memory overwrites and prevents fidelity degradation.

    Usage:
        guard = MemoryCorruptionGuard()
        verdict = guard.check_overwrite(
            existing_memory=mem_old,
            proposed_replacement=mem_new,
            reason=OverwriteReason.RECONSTRUCTION,
        )
        if verdict.action == OverwriteAction.BLOCK:
            # reject the overwrite
        elif verdict.action == OverwriteAction.DOWNGRADE:
            # write with reduced trust
    """

    def __init__(self):
        # Track how many times each memory has been blocked
        self._block_counts: dict[str, int] = {}

    # -------------------------------------------------------------------
    # Core check
    # -------------------------------------------------------------------

    def check_overwrite(
        self,
        existing_memory: Memory,
        proposed_replacement: Memory,
        reason: OverwriteReason,
    ) -> OverwriteVerdict:
        """
        Check whether a proposed memory overwrite violates Law 4.

        Args:
            existing_memory:       The current memory in storage.
            proposed_replacement:  The proposed new version.
            reason:                Why this overwrite is happening.

        Returns:
            OverwriteVerdict with action, explanation, and metrics.
        """
        e_trust = existing_memory.trust
        p_fidelity = proposed_replacement.fidelity
        p_trust = proposed_replacement.trust
        fidelity_gap = e_trust - p_fidelity

        # --- Compute content similarity if embeddings available ---
        similarity = self._compute_similarity(existing_memory, proposed_replacement)

        # --- Content drift guard (checked before reason-specific rules) ---
        if similarity is not None:
            if similarity < 0.5:
                verdict = OverwriteVerdict(
                    action=OverwriteAction.BLOCK,
                    reason=(
                        f"Content drift too high (similarity={similarity:.3f} < 0.5). "
                        f"This is a replacement, not an update. Blocked."
                    ),
                    fidelity_gap=fidelity_gap,
                    content_similarity=similarity,
                )
                self._record_block(existing_memory.id)
                return verdict

        # --- User-corrected memory protection ---
        if existing_memory.source == "user" and reason != OverwriteReason.USER_CORRECTION:
            if p_fidelity < 0.95:
                verdict = OverwriteVerdict(
                    action=OverwriteAction.BLOCK,
                    reason=(
                        f"Existing memory is user-sourced. Non-user overwrites require "
                        f"fidelity >= 0.95 but proposed is {p_fidelity:.3f}. Blocked."
                    ),
                    fidelity_gap=fidelity_gap,
                    content_similarity=similarity,
                )
                self._record_block(existing_memory.id)
                return verdict

        # --- Reason-specific rules ---
        if reason == OverwriteReason.USER_CORRECTION:
            return OverwriteVerdict(
                action=OverwriteAction.ALLOW,
                reason="User correction: user is the ultimate authority.",
                trust_delta=p_trust - e_trust,
                fidelity_gap=fidelity_gap,
                content_similarity=similarity,
            )

        if reason == OverwriteReason.NEW_EVIDENCE:
            threshold = e_trust * 0.8
            if p_fidelity >= threshold:
                return OverwriteVerdict(
                    action=OverwriteAction.ALLOW,
                    reason=(
                        f"New evidence with adequate fidelity "
                        f"({p_fidelity:.3f} >= {threshold:.3f}). Allowed."
                    ),
                    trust_delta=p_trust - e_trust,
                    fidelity_gap=fidelity_gap,
                    content_similarity=similarity,
                )
            else:
                verdict = OverwriteVerdict(
                    action=OverwriteAction.BLOCK,
                    reason=(
                        f"New evidence fidelity too low "
                        f"({p_fidelity:.3f} < {threshold:.3f}). Blocked."
                    ),
                    fidelity_gap=fidelity_gap,
                    content_similarity=similarity,
                )
                self._record_block(existing_memory.id)
                return verdict

        if reason == OverwriteReason.RECONSTRUCTION:
            # Core Law 4 case: reconstruction must not degrade
            if p_fidelity < e_trust:
                # Check if it's close enough for a downgrade
                if abs(e_trust - p_fidelity) <= 0.1:
                    # Close but slightly degraded — downgrade trust proportionally
                    adjusted_trust = p_fidelity  # trust capped at fidelity
                    return OverwriteVerdict(
                        action=OverwriteAction.DOWNGRADE,
                        reason=(
                            f"Reconstruction fidelity ({p_fidelity:.3f}) within 0.1 "
                            f"of existing trust ({e_trust:.3f}). Allowed with trust "
                            f"downgraded to {adjusted_trust:.3f}."
                        ),
                        trust_delta=adjusted_trust - e_trust,
                        fidelity_gap=fidelity_gap,
                        content_similarity=similarity,
                    )
                else:
                    # Too much degradation — block
                    verdict = OverwriteVerdict(
                        action=OverwriteAction.BLOCK,
                        reason=(
                            f"Reconstruction fidelity ({p_fidelity:.3f}) below existing "
                            f"trust ({e_trust:.3f}) by {fidelity_gap:.3f}. "
                            f"Law 4 violation: degraded reconstruction cannot overwrite."
                        ),
                        fidelity_gap=fidelity_gap,
                        content_similarity=similarity,
                    )
                    self._record_block(existing_memory.id)
                    return verdict
            else:
                # Fidelity >= trust: safe reconstruction
                # Trust cannot increase through reconstruction alone
                capped_trust = min(p_trust, e_trust)
                return OverwriteVerdict(
                    action=OverwriteAction.ALLOW,
                    reason=(
                        f"Reconstruction fidelity ({p_fidelity:.3f}) >= existing trust "
                        f"({e_trust:.3f}). Trust capped at {capped_trust:.3f} "
                        f"(reconstruction cannot raise trust)."
                    ),
                    trust_delta=capped_trust - e_trust,
                    fidelity_gap=fidelity_gap,
                    content_similarity=similarity,
                )

        if reason == OverwriteReason.CONSOLIDATION:
            threshold = e_trust * 0.7
            if p_trust < threshold:
                verdict = OverwriteVerdict(
                    action=OverwriteAction.BLOCK,
                    reason=(
                        f"Consolidation would drop trust from {e_trust:.3f} to "
                        f"{p_trust:.3f} (below 70% threshold of {threshold:.3f}). "
                        f"Blocked."
                    ),
                    fidelity_gap=fidelity_gap,
                    content_similarity=similarity,
                )
                self._record_block(existing_memory.id)
                return verdict
            else:
                return OverwriteVerdict(
                    action=OverwriteAction.ALLOW,
                    reason=(
                        f"Consolidation preserves adequate trust "
                        f"({p_trust:.3f} >= {threshold:.3f}). Allowed."
                    ),
                    trust_delta=p_trust - e_trust,
                    fidelity_gap=fidelity_gap,
                    content_similarity=similarity,
                )

        if reason == OverwriteReason.DECAY:
            # Decay can never increase trust
            capped_trust = min(p_trust, e_trust)
            return OverwriteVerdict(
                action=OverwriteAction.ALLOW,
                reason=(
                    f"Decay refresh allowed. Trust capped at existing level: "
                    f"{capped_trust:.3f} (was {p_trust:.3f}, existing {e_trust:.3f})."
                ),
                trust_delta=capped_trust - e_trust,
                fidelity_gap=fidelity_gap,
                content_similarity=similarity,
            )

        # Unknown reason — block by default (fail closed)
        verdict = OverwriteVerdict(
            action=OverwriteAction.BLOCK,
            reason=f"Unknown overwrite reason '{reason}'. Blocked (fail closed).",
            fidelity_gap=fidelity_gap,
            content_similarity=similarity,
        )
        self._record_block(existing_memory.id)
        return verdict

    # -------------------------------------------------------------------
    # Block tracking
    # -------------------------------------------------------------------

    def get_block_count(self, memory_id: str) -> int:
        """Return how many times a memory has been blocked from overwrite."""
        return self._block_counts.get(memory_id, 0)

    def needs_manual_review(self, memory_id: str) -> bool:
        """True if a memory has been blocked 3+ times (flag for review)."""
        return self.get_block_count(memory_id) >= 3

    def _record_block(self, memory_id: str):
        """Increment the block counter for a memory."""
        self._block_counts[memory_id] = self._block_counts.get(memory_id, 0) + 1

    # -------------------------------------------------------------------
    # Similarity
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_similarity(
        a: Memory, b: Memory
    ) -> Optional[float]:
        """Compute cosine similarity if both memories have embeddings."""
        if a.embedding is None or b.embedding is None:
            return None

        va = a.embedding
        vb = b.embedding

        dot = sum(x * y for x, y in zip(va, vb))
        norm_a = math.sqrt(sum(x * x for x in va))
        norm_b = math.sqrt(sum(x * x for x in vb))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


# ======================================================================
# Unit tests
# ======================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MemoryCorruptionGuard — Law 4 enforcement tests")
    print("=" * 70)

    guard = MemoryCorruptionGuard()
    passed = 0
    total = 8

    # --- Test 1: Reconstruction with lower fidelity than trust -> BLOCK ---
    print("\n[Test 1] Reconstruction with lower fidelity than trust -> BLOCK")
    existing = Memory(
        id="mem_001", content="CRT uses append-only architecture",
        trust=0.9, source="retrieved", fidelity=0.95, version=1,
    )
    proposed = Memory(
        id="mem_001", content="CRT uses append-only architecture",
        trust=0.9, source="reconstructed", fidelity=0.6, version=2,
    )
    v = guard.check_overwrite(existing, proposed, OverwriteReason.RECONSTRUCTION)
    print(f"  Action: {v.action.value} | Reason: {v.reason}")
    assert v.action == OverwriteAction.BLOCK, f"Expected BLOCK, got {v.action}"
    print("  PASSED")
    passed += 1

    # --- Test 2: Reconstruction with equal fidelity -> ALLOW ---
    print("\n[Test 2] Reconstruction with equal fidelity -> ALLOW")
    proposed2 = Memory(
        id="mem_001", content="CRT uses append-only architecture",
        trust=0.9, source="reconstructed", fidelity=0.9, version=2,
    )
    v = guard.check_overwrite(existing, proposed2, OverwriteReason.RECONSTRUCTION)
    print(f"  Action: {v.action.value} | Reason: {v.reason}")
    assert v.action == OverwriteAction.ALLOW, f"Expected ALLOW, got {v.action}"
    print("  PASSED")
    passed += 1

    # --- Test 3: User correction always passes -> ALLOW ---
    print("\n[Test 3] User correction always passes -> ALLOW")
    proposed3 = Memory(
        id="mem_001", content="CRT uses event-sourced append-only architecture",
        trust=0.95, source="user", fidelity=1.0, version=2,
    )
    v = guard.check_overwrite(existing, proposed3, OverwriteReason.USER_CORRECTION)
    print(f"  Action: {v.action.value} | Reason: {v.reason}")
    assert v.action == OverwriteAction.ALLOW, f"Expected ALLOW, got {v.action}"
    print("  PASSED")
    passed += 1

    # --- Test 4: New evidence with adequate fidelity -> ALLOW ---
    print("\n[Test 4] New evidence with adequate fidelity -> ALLOW")
    proposed4 = Memory(
        id="mem_001", content="CRT uses append-only + event-sourced architecture",
        trust=0.85, source="retrieved", fidelity=0.8, version=2,
    )
    v = guard.check_overwrite(existing, proposed4, OverwriteReason.NEW_EVIDENCE)
    print(f"  Action: {v.action.value} | Reason: {v.reason}")
    assert v.action == OverwriteAction.ALLOW, f"Expected ALLOW, got {v.action}"
    print("  PASSED")
    passed += 1

    # --- Test 5: Consolidation that loses too much trust -> BLOCK ---
    print("\n[Test 5] Consolidation that loses too much trust -> BLOCK")
    proposed5 = Memory(
        id="mem_001", content="CRT architecture summary",
        trust=0.5, source="generated", fidelity=0.7, version=2,
    )
    v = guard.check_overwrite(existing, proposed5, OverwriteReason.CONSOLIDATION)
    print(f"  Action: {v.action.value} | Reason: {v.reason}")
    assert v.action == OverwriteAction.BLOCK, f"Expected BLOCK, got {v.action}"
    print("  PASSED")
    passed += 1

    # --- Test 6: Content drift too high (similarity < 0.5) -> BLOCK ---
    print("\n[Test 6] Content drift too high (similarity < 0.5) -> BLOCK")
    # Use orthogonal-ish embeddings to simulate content drift
    existing6 = Memory(
        id="mem_002", content="Python is great for ML",
        trust=0.85, source="user", fidelity=0.95, version=1,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    proposed6 = Memory(
        id="mem_002", content="The weather is sunny today",
        trust=0.85, source="retrieved", fidelity=0.95, version=2,
        embedding=[0.0, 1.0, 0.0, 0.0],
    )
    v = guard.check_overwrite(existing6, proposed6, OverwriteReason.NEW_EVIDENCE)
    print(f"  Action: {v.action.value} | Similarity: {v.content_similarity}")
    print(f"  Reason: {v.reason}")
    assert v.action == OverwriteAction.BLOCK, f"Expected BLOCK, got {v.action}"
    assert v.content_similarity is not None and v.content_similarity < 0.5
    print("  PASSED")
    passed += 1

    # --- Test 7: Overwrite of user-corrected memory requires high fidelity -> BLOCK ---
    print("\n[Test 7] Overwrite user-corrected memory needs fidelity >= 0.95 -> BLOCK")
    existing7 = Memory(
        id="mem_003", content="Nick prefers Python for backend",
        trust=0.95, source="user", fidelity=1.0, version=1,
    )
    proposed7 = Memory(
        id="mem_003", content="Nick prefers Python for backend work",
        trust=0.9, source="reconstructed", fidelity=0.85, version=2,
    )
    v = guard.check_overwrite(existing7, proposed7, OverwriteReason.RECONSTRUCTION)
    print(f"  Action: {v.action.value} | Reason: {v.reason}")
    assert v.action == OverwriteAction.BLOCK, f"Expected BLOCK, got {v.action}"
    print("  PASSED")
    passed += 1

    # --- Test 8: Decay cannot increase trust -> ALLOW but capped ---
    print("\n[Test 8] Decay cannot increase trust -> ALLOW but capped")
    existing8 = Memory(
        id="mem_004", content="CRT launched in March 2026",
        trust=0.7, source="retrieved", fidelity=0.8, version=3,
    )
    proposed8 = Memory(
        id="mem_004", content="CRT launched in March 2026",
        trust=0.95, source="retrieved", fidelity=0.9, version=4,
    )
    v = guard.check_overwrite(existing8, proposed8, OverwriteReason.DECAY)
    print(f"  Action: {v.action.value} | Reason: {v.reason}")
    assert v.action == OverwriteAction.ALLOW, f"Expected ALLOW, got {v.action}"
    # Trust delta should be <= 0 (cannot increase)
    assert v.trust_delta <= 0.0, f"Decay should not increase trust, delta={v.trust_delta}"
    print("  PASSED")
    passed += 1

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("Law 4 enforcement is operational.")
    else:
        print("FAILURES DETECTED — review above.")
    print("=" * 70)
