"""
SpeechLeakDetector — Immune Agent for CRT Law 1

Law 1 of the Mirus/Holden constitution: "Speech cannot upgrade belief."

A model's generated output must never be stored as memory with higher trust
than the evidence that produced it. This agent watches the output -> memory
write path and fires when a generated response tries to promote itself into
trusted memory without grounding.

Threat model:
  - LLM hallucinates a fact in its response
  - Downstream pipeline tries to store that response as a memory
  - Without this gate, the hallucination becomes "trusted memory"
  - Next retrieval cycle treats it as grounded evidence
  - Feedback loop: speech -> belief -> stronger speech (runaway)

This agent breaks the loop by requiring grounding evidence before any
generated text can be written at non-zero trust.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------

class VerdictType(Enum):
    ALLOW = "ALLOW"           # Write permitted, grounding found
    DOWNGRADE = "DOWNGRADE"   # Write permitted but trust capped/zeroed
    BLOCK = "BLOCK"           # Write rejected entirely


@dataclass
class Verdict:
    """Result of a speech-leak detection check."""
    action: VerdictType
    original_trust: float
    final_trust: float
    reason: str
    best_grounding_similarity: float = 0.0
    grounding_source: Optional[str] = None


# ---------------------------------------------------------------------------
# Memory representation (lightweight, for the detector's interface)
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """A stored memory with trust and source metadata."""
    text: str
    trust: float                         # 0.0 - 1.0
    source: str                          # "user", "retrieved", "generated", etc.
    embedding: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Cosine similarity above this means "grounded" — the claim has support
GROUNDING_THRESHOLD = 0.65

# Below this, even partial matches don't count
PARTIAL_THRESHOLD = 0.45

# Generated sources that cannot self-promote
GENERATED_SOURCES = {"generated", "model_output", "llm_response", "synthesis"}

# Trusted sources that don't need grounding checks
GROUNDED_SOURCES = {"user", "user_input", "retrieved", "external_api", "sensor"}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SpeechLeakDetector:
    """
    Watches the output -> memory write path.
    Fires when generated text tries to store itself as trusted belief.

    Usage:
        detector = SpeechLeakDetector(embedding_fn=my_encoder)
        verdict = detector.detect(
            candidate_text="The capital of France is Paris",
            proposed_trust=0.9,
            source="generated",
            existing_memories=[...list of MemoryRecord...]
        )
        if verdict.action == VerdictType.BLOCK:
            # reject the write
        elif verdict.action == VerdictType.DOWNGRADE:
            # write with verdict.final_trust instead
    """

    def __init__(
        self,
        embedding_fn=None,
        grounding_threshold: float = GROUNDING_THRESHOLD,
        partial_threshold: float = PARTIAL_THRESHOLD,
    ):
        """
        Args:
            embedding_fn: Callable that takes a string and returns a numpy
                          array (normalized). If None, uses the project's
                          EmbeddingEngine.
            grounding_threshold: Cosine similarity above which a memory
                                 is considered full grounding.
            partial_threshold: Cosine similarity above which a memory is
                               considered partial grounding (trust capped).
        """
        self.grounding_threshold = grounding_threshold
        self.partial_threshold = partial_threshold

        if embedding_fn is not None:
            self._encode = embedding_fn
        else:
            self._encoder = None  # lazy-load

    def _get_encoder(self):
        """Lazy-load an embedding engine. Tries sentence-transformers, falls back to error."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError(
                    "SpeechLeakDetector requires an embedding function. "
                    "Either pass embedding_fn= to the constructor, or "
                    "install sentence-transformers: pip install sentence-transformers"
                )
        return self._encoder

    def _encode(self, text: str) -> np.ndarray:
        """Default encode using project embeddings."""
        return self._get_encoder().encode(text)

    # -------------------------------------------------------------------
    # Core detection
    # -------------------------------------------------------------------

    def detect(
        self,
        candidate_text: str,
        proposed_trust: float,
        source: str,
        existing_memories: List[MemoryRecord],
    ) -> Verdict:
        """
        Check whether a candidate memory write violates Law 1.

        Args:
            candidate_text: The text about to be written to memory.
            proposed_trust: The trust score the writer wants to assign.
            source: Where this text came from ("generated", "user", etc.).
            existing_memories: Current grounded memories to check against.

        Returns:
            Verdict with action, adjusted trust, and reason.
        """
        # --- Pass-through for grounded sources ---
        if source.lower() in GROUNDED_SOURCES:
            return Verdict(
                action=VerdictType.ALLOW,
                original_trust=proposed_trust,
                final_trust=proposed_trust,
                reason=f"Source '{source}' is grounded; no check needed.",
            )

        # --- Generated source: needs grounding evidence ---
        if not existing_memories:
            return Verdict(
                action=VerdictType.DOWNGRADE,
                original_trust=proposed_trust,
                final_trust=0.0,
                reason="No existing memories to ground against. "
                       "Trust zeroed; tagged as ungrounded generation.",
            )

        # Encode candidate
        candidate_emb = self._encode(candidate_text)

        # Find best grounding match among non-generated memories
        best_sim = 0.0
        best_record: Optional[MemoryRecord] = None

        for mem in existing_memories:
            # Only ground against trusted (non-generated) memories
            if mem.source.lower() in GENERATED_SOURCES:
                continue

            # Get or compute embedding
            if mem.embedding is not None:
                mem_emb = mem.embedding
            else:
                mem_emb = self._encode(mem.text)

            sim = float(np.dot(candidate_emb, mem_emb))
            if sim > best_sim:
                best_sim = sim
                best_record = mem

        # --- Decision logic ---

        # Full grounding: high similarity to a trusted memory
        if best_sim >= self.grounding_threshold and best_record is not None:
            # Cap trust at grounding memory's trust level
            capped_trust = min(proposed_trust, best_record.trust)
            return Verdict(
                action=VerdictType.ALLOW,
                original_trust=proposed_trust,
                final_trust=capped_trust,
                reason=f"Grounded by existing memory (sim={best_sim:.3f}). "
                       f"Trust capped at grounding source's {best_record.trust:.2f}.",
                best_grounding_similarity=best_sim,
                grounding_source=best_record.text[:80],
            )

        # Partial grounding: moderate similarity
        if best_sim >= self.partial_threshold and best_record is not None:
            # Downgrade: scale trust by similarity ratio and cap
            scaled_trust = proposed_trust * (best_sim / self.grounding_threshold)
            capped_trust = min(scaled_trust, best_record.trust * 0.5)
            return Verdict(
                action=VerdictType.DOWNGRADE,
                original_trust=proposed_trust,
                final_trust=round(capped_trust, 3),
                reason=f"Partial grounding (sim={best_sim:.3f} < threshold "
                       f"{self.grounding_threshold}). Trust reduced.",
                best_grounding_similarity=best_sim,
                grounding_source=best_record.text[:80],
            )

        # No grounding: block or zero trust
        if proposed_trust > 0.3:
            return Verdict(
                action=VerdictType.BLOCK,
                original_trust=proposed_trust,
                final_trust=0.0,
                reason=f"No grounding found (best sim={best_sim:.3f}). "
                       f"Generated claim at trust {proposed_trust:.2f} blocked. "
                       f"Law 1 violation: speech cannot upgrade belief.",
                best_grounding_similarity=best_sim,
            )
        else:
            # Low-trust writes get through but zeroed
            return Verdict(
                action=VerdictType.DOWNGRADE,
                original_trust=proposed_trust,
                final_trust=0.0,
                reason=f"No grounding found (best sim={best_sim:.3f}). "
                       f"Low-trust write allowed but zeroed.",
                best_grounding_similarity=best_sim,
            )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print("=" * 70)
    print("SpeechLeakDetector — Law 1 enforcement tests")
    print("=" * 70)

    # Load a real embedding model for proper semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(text: str) -> np.ndarray:
        emb = model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    detector = SpeechLeakDetector(embedding_fn=encode)

    # --- Build some grounded memories ---
    grounded_memories = [
        MemoryRecord(
            text="The capital of France is Paris.",
            trust=0.95,
            source="user",
            embedding=encode("The capital of France is Paris."),
        ),
        MemoryRecord(
            text="Nick prefers Python over JavaScript for backend work.",
            trust=0.90,
            source="user",
            embedding=encode("Nick prefers Python over JavaScript for backend work."),
        ),
        MemoryRecord(
            text="The CRT system uses append-only ledger architecture.",
            trust=0.85,
            source="retrieved",
            embedding=encode("The CRT system uses append-only ledger architecture."),
        ),
    ]

    passed = 0
    total = 4

    # --- Test 1: Hallucinated claim (no grounding) ---
    print("\nTest 1: Hallucinated claim — no grounding exists")
    v1 = detector.detect(
        candidate_text="The population of Mars is 12 million people.",
        proposed_trust=0.85,
        source="generated",
        existing_memories=grounded_memories,
    )
    print(f"  Verdict: {v1.action.value} | Trust: {v1.original_trust} -> {v1.final_trust}")
    print(f"  Reason: {v1.reason}")
    assert v1.action in (VerdictType.BLOCK, VerdictType.DOWNGRADE), \
        f"Expected BLOCK or DOWNGRADE, got {v1.action}"
    assert v1.final_trust == 0.0, f"Expected trust 0.0, got {v1.final_trust}"
    print("  PASSED")
    passed += 1

    # --- Test 2: Grounded claim ---
    print("\nTest 2: Grounded claim — matches existing memory")
    v2 = detector.detect(
        candidate_text="Paris is the capital city of France.",
        proposed_trust=0.90,
        source="generated",
        existing_memories=grounded_memories,
    )
    print(f"  Verdict: {v2.action.value} | Trust: {v2.original_trust} -> {v2.final_trust}")
    print(f"  Reason: {v2.reason}")
    print(f"  Grounding sim: {v2.best_grounding_similarity:.3f}")
    assert v2.action == VerdictType.ALLOW, f"Expected ALLOW, got {v2.action}"
    assert v2.final_trust <= 0.95, f"Trust should be capped at grounding's 0.95"
    assert v2.final_trust > 0.0, f"Trust should be non-zero for grounded claim"
    print("  PASSED")
    passed += 1

    # --- Test 3: User input (grounded source, pass-through) ---
    print("\nTest 3: User input — grounded source, no check needed")
    v3 = detector.detect(
        candidate_text="I like cats more than dogs.",
        proposed_trust=0.95,
        source="user",
        existing_memories=grounded_memories,
    )
    print(f"  Verdict: {v3.action.value} | Trust: {v3.original_trust} -> {v3.final_trust}")
    print(f"  Reason: {v3.reason}")
    assert v3.action == VerdictType.ALLOW, f"Expected ALLOW, got {v3.action}"
    assert v3.final_trust == 0.95, f"User input trust should pass through unchanged"
    print("  PASSED")
    passed += 1

    # --- Test 4: Partial match — generated claim that's related but not exact ---
    print("\nTest 4: Partial match — generated claim tangentially related")
    v4 = detector.detect(
        candidate_text="Nick's software architecture follows event-driven patterns.",
        proposed_trust=0.80,
        source="generated",
        existing_memories=grounded_memories,
    )
    print(f"  Verdict: {v4.action.value} | Trust: {v4.original_trust} -> {v4.final_trust}")
    print(f"  Reason: {v4.reason}")
    print(f"  Grounding sim: {v4.best_grounding_similarity:.3f}")
    # Should be DOWNGRADE: related to the "Nick prefers Python" memory but
    # not semantically identical. Trust should be capped below proposed.
    assert v4.action in (VerdictType.DOWNGRADE, VerdictType.BLOCK), \
        f"Expected DOWNGRADE or BLOCK for partial match, got {v4.action}"
    assert v4.final_trust < v4.original_trust, \
        f"Trust should be reduced: {v4.final_trust} < {v4.original_trust}"
    print("  PASSED")
    passed += 1

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("Law 1 enforcement is operational.")
    else:
        print("FAILURES DETECTED — review above.")
    print("=" * 70)
