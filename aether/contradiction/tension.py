"""Structural Tension Meter: Zero-LLM belief verification for Aether.

Measures the tension between two memories using only structural signals:
- Slot extraction (via aether.memory.slots)
- Embedding similarity (via injected encoder)
- Metadata comparison (trust, recency, source type)
- Temporal status (past/active/future/potential)

No LLM calls. Runs in ~0.2s per pair. Designed for continuous breathing loop.

Discovered through BDG audit experiments (2026-04-10):
- LLM-based contradiction detection: 38-40% accuracy
- Structural slot+embedding approach: 75% accuracy, zero model calls
- Key insight: tension between beliefs is measurable, not judgable

Integration points:
1. Memory write path — pre-storage tension check
2. Memory consolidation — NLI pre-filter
3. Heartbeat breathing loop — continuous monitoring
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class TensionRelationship(str, Enum):
    """Classification of the relationship between two memories."""
    DUPLICATE = "duplicate"         # Same slot, same value, high embedding similarity
    REFINEMENT = "refinement"       # Same slot, one value contains the other
    COMPATIBLE = "compatible"       # Same topic, no conflicting slots (or temporal evolution)
    TENSION = "tension"             # Same slot, different value, but weak/ambiguous signals
    CONFLICT = "conflict"           # Same slot, different value, strong structural signals
    UNRELATED = "unrelated"         # Low embedding similarity, no shared slots
    DECAYED = "decayed"             # One side has trust below threshold with no recent support


class TensionAction(str, Enum):
    """Recommended action based on tension analysis."""
    KEEP_BOTH = "keep_both"
    MERGE = "merge"
    KEEP_MORE_SPECIFIC = "keep_more_specific"
    FLAG_FOR_REVIEW = "flag_for_review"
    ESCALATE_TO_NLI = "escalate_to_nli"
    BUMP_EXISTING = "bump_existing"
    DEPRECATE_WEAKER = "deprecate_weaker"


@dataclass
class SlotOverlap:
    """A single overlapping slot between two memories."""
    slot: str
    value_a: str
    value_b: str
    match_type: str  # "same", "a_subset_of_b", "b_subset_of_a", "different"
    temporal_a: str = "active"
    temporal_b: str = "active"


@dataclass
class TensionResult:
    """Result of a structural tension measurement."""
    tension_score: float            # 0.0 (no tension) to 1.0 (direct conflict)
    relationship: TensionRelationship
    action: TensionAction
    confidence: float               # How confident the meter is (0.0-1.0)
    supporting_signals: Dict[str, Any] = field(default_factory=dict)
    slot_overlaps: List[SlotOverlap] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tension_score": round(self.tension_score, 4),
            "relationship": self.relationship.value,
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "supporting_signals": self.supporting_signals,
            "slot_overlaps": [
                {"slot": s.slot, "value_a": s.value_a, "value_b": s.value_b,
                 "match_type": s.match_type}
                for s in self.slot_overlaps
            ],
        }


# ============================================================================
# Thresholds
# ============================================================================

EMBEDDING_DUPLICATE_THRESHOLD = 0.90
EMBEDDING_SAME_TOPIC_THRESHOLD = 0.40
EMBEDDING_UNRELATED_THRESHOLD = 0.25
TRUST_DECAY_THRESHOLD = 0.5
TRUST_CONFLICT_DELTA = 0.15  # Minimum trust difference to pick a winner

# Oscillation thresholds (validated by exp5b_oscillation.py 2026-04-10)
# osc=0: 2.9% corrected, osc>=2: 11.5% (3.9x), osc>=4: 31.1% (10.7x)
OSCILLATION_FLAG_THRESHOLD = 4   # Flag memories with 4+ direction changes
OSCILLATION_TENSION_BOOST = 0.20  # Boost tension_score for high oscillators


# ============================================================================
# Oscillation Detection
# ============================================================================

def compute_oscillation_count(trust_history: List[Dict]) -> int:
    """Count trust direction changes from a memory's trust history.

    An oscillation is a direction change: trust goes up, then down, or vice versa.

    Args:
        trust_history: List of {old_trust, new_trust, ...} dicts, ordered by time

    Returns:
        Number of direction changes (oscillations). 0 if monotonic or <2 entries.

    Validated by exp5b_oscillation.py (2026-04-10):
    - osc=0: 2.9% correction rate (stable memories)
    - osc>=4: 31.1% correction rate (10.7x baseline, highly unstable)
    """
    if len(trust_history) < 2:
        return 0

    # Sort by timestamp ascending (oldest first) if not already
    # trust_history may come in DESC order from DB
    sorted_hist = sorted(trust_history, key=lambda x: x.get('timestamp', 0))

    oscillations = 0
    last_direction = None  # +1 for up, -1 for down, None for no change

    for entry in sorted_hist:
        old = entry.get('old_trust', 0.5)
        new = entry.get('new_trust', 0.5)
        delta = new - old

        if abs(delta) < 0.01:  # Skip negligible changes
            continue

        direction = 1 if delta > 0 else -1

        if last_direction is not None and direction != last_direction:
            oscillations += 1

        last_direction = direction

    return oscillations


# ============================================================================
# Core: StructuralTensionMeter
# ============================================================================

class StructuralTensionMeter:
    """Measures tension between two memories using structural signals only.

    No LLM calls. Uses slot extraction, embedding similarity, and metadata.
    Designed to run in the hot path (memory writes, heartbeat loop).
    """

    def __init__(self, encoder=None, slot_extractor=None):
        """
        Args:
            encoder: Optional embedding encoder instance with an .encode(text)
                     method. If None, _encode() returns np.array([]) as a
                     graceful fallback (no lazy import).
            slot_extractor: Optional callable(text) -> Dict of extracted slots.
                           If None, uses aether.memory.slots extractors.
        """
        self._encoder = encoder
        self._slot_extractor = slot_extractor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(
        self,
        text_a: str,
        text_b: str,
        trust_a: float = 0.5,
        trust_b: float = 0.5,
        source_a: str = "user",
        source_b: str = "user",
        timestamp_a: float = 0.0,
        timestamp_b: float = 0.0,
        slots_a: Optional[Dict] = None,
        slots_b: Optional[Dict] = None,
        vector_a: Optional[np.ndarray] = None,
        vector_b: Optional[np.ndarray] = None,
        oscillation_a: int = 0,
        oscillation_b: int = 0,
    ) -> TensionResult:
        """Measure structural tension between two memories.

        Args:
            text_a, text_b: Memory text content
            trust_a, trust_b: Trust scores (0.0-1.0)
            source_a, source_b: Source type (user_fact, inferred, correction, etc.)
            timestamp_a, timestamp_b: Unix timestamps
            slots_a, slots_b: Pre-extracted slots (Dict[str, ExtractedFact]). Extracted if None.
            vector_a, vector_b: Pre-computed embedding vectors. Computed if None.

        Returns:
            TensionResult with relationship, action, score, and evidence.
        """
        signals: Dict[str, Any] = {}

        # Step 1: Slot extraction
        if slots_a is None:
            slots_a = self._extract_slots(text_a)
        if slots_b is None:
            slots_b = self._extract_slots(text_b)
        signals["slots_a_count"] = len(slots_a)
        signals["slots_b_count"] = len(slots_b)

        # Step 2: Slot comparison
        overlaps = self._compare_slots(slots_a, slots_b)
        signals["shared_slots"] = len(overlaps)

        has_conflict = any(o.match_type == "different" for o in overlaps)
        has_refinement = any(o.match_type in ("a_subset_of_b", "b_subset_of_a") for o in overlaps)
        has_agreement = any(o.match_type == "same" for o in overlaps)
        all_agree = overlaps and all(o.match_type in ("same", "a_subset_of_b", "b_subset_of_a") for o in overlaps)
        no_shared_slots = len(overlaps) == 0

        signals["has_conflict"] = has_conflict
        signals["has_refinement"] = has_refinement
        signals["has_agreement"] = has_agreement

        # Step 3: Embedding similarity
        similarity = self._compute_similarity(text_a, text_b, vector_a, vector_b)
        signals["embedding_similarity"] = round(similarity, 4)

        same_topic = similarity > EMBEDDING_SAME_TOPIC_THRESHOLD
        near_dup = similarity > EMBEDDING_DUPLICATE_THRESHOLD
        is_unrelated = similarity < EMBEDDING_UNRELATED_THRESHOLD

        # Step 4: Metadata comparison
        trust_delta = abs(trust_a - trust_b)
        signals["trust_a"] = trust_a
        signals["trust_b"] = trust_b
        signals["trust_delta"] = round(trust_delta, 4)

        is_correction = "correction" in source_b or "correction" in source_a
        is_observation = "observation" in source_b or "observation" in source_a
        signals["is_correction"] = is_correction

        b_newer = timestamp_b > timestamp_a if (timestamp_a > 0 and timestamp_b > 0) else None
        signals["b_newer"] = b_newer

        trust_a_low = trust_a < TRUST_DECAY_THRESHOLD
        trust_b_low = trust_b < TRUST_DECAY_THRESHOLD
        signals["trust_a_low"] = trust_a_low

        # Step 5: Temporal status
        temporal_evolution = self._check_temporal_evolution(overlaps)
        signals["temporal_evolution"] = temporal_evolution

        # Step 5b: Oscillation (trust direction changes — validated predictor of correction)
        max_oscillation = max(oscillation_a, oscillation_b)
        high_oscillator = max_oscillation >= OSCILLATION_FLAG_THRESHOLD
        signals["oscillation_a"] = oscillation_a
        signals["oscillation_b"] = oscillation_b
        signals["high_oscillator"] = high_oscillator

        # Step 6: Scoring / Propagation
        return self._score(
            overlaps=overlaps,
            signals=signals,
            has_conflict=has_conflict,
            has_refinement=has_refinement,
            has_agreement=has_agreement,
            all_agree=all_agree,
            no_shared_slots=no_shared_slots,
            same_topic=same_topic,
            near_dup=near_dup,
            is_unrelated=is_unrelated,
            similarity=similarity,
            trust_a=trust_a,
            trust_b=trust_b,
            trust_delta=trust_delta,
            is_correction=is_correction,
            is_observation=is_observation,
            b_newer=b_newer,
            trust_a_low=trust_a_low,
            trust_b_low=trust_b_low,
            temporal_evolution=temporal_evolution,
            high_oscillator=high_oscillator,
            max_oscillation=max_oscillation,
        )

    def measure_pair(self, mem_a, mem_b) -> TensionResult:
        """Convenience: measure from MemoryItem-like objects.

        Works with any object that has .text, .trust, .source, .timestamp attributes.
        Reads pre-cached slots from .fact_slots if available.
        Reads oscillation_count from .oscillation if available.
        """
        return self.measure(
            text_a=getattr(mem_a, "text", str(mem_a)),
            text_b=getattr(mem_b, "text", str(mem_b)),
            trust_a=getattr(mem_a, "trust", 0.5),
            trust_b=getattr(mem_b, "trust", 0.5),
            source_a=getattr(mem_a, "source", "user"),
            source_b=getattr(mem_b, "source", "user"),
            timestamp_a=getattr(mem_a, "timestamp", 0.0),
            timestamp_b=getattr(mem_b, "timestamp", 0.0),
            vector_a=getattr(mem_a, "vector", None),
            vector_b=getattr(mem_b, "vector", None),
            oscillation_a=getattr(mem_a, "oscillation", 0),
            oscillation_b=getattr(mem_b, "oscillation", 0),
        )

    def measure_against_cluster(
        self,
        text: str,
        cluster,
        trust: float = 0.5,
        source: str = "user",
        timestamp: float = 0.0,
        vector: Optional[np.ndarray] = None,
    ) -> List[Tuple[Any, TensionResult]]:
        """Measure one memory's text against a cluster of existing memories.

        Extracts the new text's slots once, reuses across all comparisons.

        Args:
            text: New memory text
            cluster: List of MemoryItem-like objects
            trust, source, timestamp: Metadata for the new memory
            vector: Pre-computed embedding vector for the new text

        Returns:
            List of (memory, TensionResult) tuples, sorted by tension_score descending.
        """
        if not cluster:
            return []

        # Extract new text's slots once
        new_slots = self._extract_slots(text)
        if vector is None:
            vector = self._encode(text)

        results = []
        for mem in cluster:
            result = self.measure(
                text_a=text,
                text_b=getattr(mem, "text", str(mem)),
                trust_a=trust,
                trust_b=getattr(mem, "trust", 0.5),
                source_a=source,
                source_b=getattr(mem, "source", "user"),
                timestamp_a=timestamp,
                timestamp_b=getattr(mem, "timestamp", 0.0),
                slots_a=new_slots,
                vector_a=vector,
                vector_b=getattr(mem, "vector", None),
            )
            results.append((mem, result))

        results.sort(key=lambda x: x[1].tension_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _extract_slots(self, text: str) -> Dict:
        """Extract fact slots from memory text.

        Tries the injected slot_extractor first, then aether.memory.slots extractors.
        Falls back to lightweight regex patterns for third-person memory format
        like "Nick lives in X".
        """
        slots: Dict = {}

        # Try injected extractor first
        if self._slot_extractor is not None:
            try:
                slots = self._slot_extractor(text) or {}
            except Exception:
                pass
            if slots:
                return slots

        # Try aether.memory.slots extractors
        try:
            from aether.memory.slots import extract_fact_slots_contextual
            slots = extract_fact_slots_contextual(text) or {}
        except Exception:
            try:
                from aether.memory.slots import extract_fact_slots
                slots = extract_fact_slots(text) or {}
            except Exception:
                pass

        # If production extractor returned nothing, try third-person patterns
        # (stored memories are often "Nick lives in X" not "I live in X")
        if not slots:
            slots = self._extract_slots_fallback(text)

        return slots

    def _extract_slots_fallback(self, text: str) -> Dict:
        """Lightweight slot extraction for third-person memory text.

        Covers common patterns that the first-person fact_slots extractor misses:
        - "X lives in Y" -> location
        - "X is a Y" -> occupation
        - "X's favorite Y is Z" -> preference/favorite_Y
        - "X works on/at/as Y" -> occupation/skill
        - "X's name is Y" / "X is called Y" -> name
        - "X is learning Y" -> skill
        - "X built/created Y" -> skill
        - "X prefers Y" -> preference
        """
        text_lower = text.lower().strip()
        # Strip common metadata suffixes for cleaner matching
        text_clean = re.sub(r'\.\s*(?:trust|source|age):.*$', '', text_lower, flags=re.IGNORECASE).strip()

        slots = {}

        def _make_fact(slot: str, value: str, temporal: str = "active"):
            f = object.__new__(type("Fact", (), {}))
            f.slot = slot
            f.value = value
            f.normalized = value.lower().strip()
            f.temporal_status = temporal
            f.period_text = None
            f.domains = ()
            f.confidence = 0.9
            return f

        PATTERNS = [
            # Location
            (r"(?:\w+\s+)?lives?\s+in\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "location"),
            (r"(?:\w+\s+)?is\s+from\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "location"),
            (r"(?:\w+\s+)?(?:is\s+)?based\s+in\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "location"),
            (r"(?:\w+\s+)?(?:is\s+)?located\s+in\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "location"),
            # Occupation
            (r"(?:\w+\s+)?is\s+a\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "occupation"),
            (r"(?:\w+\s+)?works?\s+as\s+a?\s*(.+?)(?:\.|,\s*(?:trust|source)|$)", "occupation"),
            # Skill / work area
            (r"(?:\w+\s+)?works?\s+on\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "skill"),
            (r"(?:\w+\s+)?is\s+learning\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "skill"),
            (r"(?:\w+\s+)?built\s+(?:a\s+)?(?:project\s+in\s+)?(.+?)(?:\.|,\s*(?:trust|source)|$)", "skill"),
            # Preference / favorite
            (r"(?:\w+'s\s+)?favorite\s+(\w+)\s+is\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "favorite"),
            (r"(?:\w+'s\s+)?favourite\s+(\w+)\s+is\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "favorite"),
            (r"(?:\w+\s+)?prefers?\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "preference"),
            (r"(?:\w+\s+)?enjoys?\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "preference"),
            # Name
            (r"(?:\w+'s\s+)?name\s+is\s+(.+?)(?:\.|,\s*(?:trust|source)|$)", "name"),
        ]

        # Check temporal status
        temporal = "active"
        if re.search(r"\b(?:used to|formerly|no longer|don't|quit|left)\b", text_clean):
            temporal = "past"
        elif re.search(r"\b(?:will|plan to|going to|starting)\b", text_clean):
            temporal = "future"
        elif re.search(r"\b(?:has not mentioned|not mentioned|no longer)\b", text_clean):
            temporal = "past"

        for pattern, slot_type in PATTERNS:
            m = re.search(pattern, text_clean, re.IGNORECASE)
            if m:
                if slot_type == "favorite" and m.lastindex and m.lastindex >= 2:
                    # "favorite X is Y" -> slot = "favorite_X", value = "Y"
                    category = m.group(1).strip()
                    value = m.group(2).strip()
                    slot_name = f"favorite_{category}"
                    if value and len(value) > 1:
                        slots[slot_name] = _make_fact(slot_name, value, temporal)
                elif m.lastindex:
                    value = m.group(m.lastindex).strip()
                    if value and len(value) > 1:
                        slots[slot_type] = _make_fact(slot_type, value, temporal)

        return slots

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector.

        If no encoder was injected, returns an empty array (graceful fallback).
        """
        if self._encoder is None:
            return np.array([])
        try:
            return self._encoder.encode(text)
        except Exception:
            logger.debug("[TENSION] Embedding failed for: %s", text[:80])
            return np.array([])

    def _compute_similarity(
        self,
        text_a: str,
        text_b: str,
        vector_a: Optional[np.ndarray],
        vector_b: Optional[np.ndarray],
    ) -> float:
        """Compute cosine similarity between two memories."""
        if vector_a is None:
            vector_a = self._encode(text_a)
        if vector_b is None:
            vector_b = self._encode(text_b)

        if vector_a.size == 0 or vector_b.size == 0:
            return 0.0
        if vector_a.shape != vector_b.shape:
            return 0.0

        dot = float(np.dot(vector_a, vector_b))
        norm_a = float(np.linalg.norm(vector_a))
        norm_b = float(np.linalg.norm(vector_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _compare_slots(self, slots_a: Dict, slots_b: Dict) -> List[SlotOverlap]:
        """Compare overlapping slots between two memories."""
        overlaps = []
        shared_keys = set(slots_a.keys()) & set(slots_b.keys())

        for key in shared_keys:
            fact_a = slots_a[key]
            fact_b = slots_b[key]

            # Get normalized values
            val_a = self._normalize_value(fact_a)
            val_b = self._normalize_value(fact_b)

            # Get temporal status
            temp_a = getattr(fact_a, "temporal_status", "active")
            temp_b = getattr(fact_b, "temporal_status", "active")

            # Determine match type
            if val_a == val_b:
                match_type = "same"
            elif val_a in val_b:
                match_type = "a_subset_of_b"
            elif val_b in val_a:
                match_type = "b_subset_of_a"
            else:
                match_type = "different"

            overlaps.append(SlotOverlap(
                slot=key,
                value_a=val_a,
                value_b=val_b,
                match_type=match_type,
                temporal_a=temp_a,
                temporal_b=temp_b,
            ))

        return overlaps

    def _normalize_value(self, fact) -> str:
        """Extract and normalize a fact value for comparison."""
        if hasattr(fact, "normalized") and fact.normalized:
            return str(fact.normalized).strip().lower()
        if hasattr(fact, "value"):
            return str(fact.value).strip().lower()
        return str(fact).strip().lower()

    def _check_temporal_evolution(self, overlaps: List[SlotOverlap]) -> bool:
        """Check if slot differences are explained by temporal evolution.

        Example: slot 'employer' with temporal_a='past' and temporal_b='active'
        means the user changed jobs — this is evolution, not conflict.
        """
        for o in overlaps:
            if o.match_type == "different":
                # Different values but one is past and one is active = evolution
                if (o.temporal_a == "past" and o.temporal_b == "active") or \
                   (o.temporal_a == "active" and o.temporal_b == "past"):
                    return True
                # Future/potential vs active = planned change
                if o.temporal_b in ("future", "potential") or o.temporal_a in ("future", "potential"):
                    return True
        return False

    def _score(self, overlaps, signals, **kwargs) -> TensionResult:
        """Deterministic scoring: map signals to relationship + action + score."""
        has_conflict = kwargs["has_conflict"]
        has_refinement = kwargs["has_refinement"]
        all_agree = kwargs["all_agree"]
        no_shared_slots = kwargs["no_shared_slots"]
        same_topic = kwargs["same_topic"]
        near_dup = kwargs["near_dup"]
        is_unrelated = kwargs["is_unrelated"]
        similarity = kwargs["similarity"]
        is_correction = kwargs["is_correction"]
        is_observation = kwargs["is_observation"]
        b_newer = kwargs["b_newer"]
        trust_a_low = kwargs["trust_a_low"]
        trust_b_low = kwargs["trust_b_low"]
        temporal_evolution = kwargs["temporal_evolution"]
        trust_delta = kwargs["trust_delta"]
        # Oscillation: 4+ direction changes = 31.1% correction rate (exp5b)
        high_oscillator = kwargs.get("high_oscillator", False)
        max_oscillation = kwargs.get("max_oscillation", 0)

        # --- Rule 1: Unrelated (low similarity, no shared slots) ---
        if is_unrelated and no_shared_slots:
            return TensionResult(
                tension_score=0.0,
                relationship=TensionRelationship.UNRELATED,
                action=TensionAction.KEEP_BOTH,
                confidence=0.9,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 2: Refinement (one value contains the other) — checked before duplicate ---
        if has_refinement and not has_conflict:
            return TensionResult(
                tension_score=0.15,
                relationship=TensionRelationship.REFINEMENT,
                action=TensionAction.KEEP_MORE_SPECIFIC,
                confidence=0.85,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 3: Near-duplicate (very high similarity, slots agree, no refinement) ---
        if near_dup and (all_agree or no_shared_slots):
            return TensionResult(
                tension_score=0.05,
                relationship=TensionRelationship.DUPLICATE,
                action=TensionAction.BUMP_EXISTING,
                confidence=0.95,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 4: Temporal evolution (same slot, different value, but explained by time) ---
        if has_conflict and temporal_evolution:
            return TensionResult(
                tension_score=0.25,
                relationship=TensionRelationship.COMPATIBLE,
                action=TensionAction.KEEP_BOTH,
                confidence=0.8,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 4b: High oscillation (validated 2026-04-10, exp5b) ---
        # osc>=4 = 31.1% correction rate vs 2.9% baseline (10.7x)
        # These memories have unstable trust history — flag for consolidation review
        if high_oscillator and same_topic:
            # Apply tension boost proportional to oscillation count
            osc_boost = min(OSCILLATION_TENSION_BOOST, max_oscillation * 0.04)
            return TensionResult(
                tension_score=0.45 + osc_boost,  # Elevated base + boost
                relationship=TensionRelationship.TENSION,
                action=TensionAction.FLAG_FOR_REVIEW,
                confidence=0.7,  # Lower confidence — oscillation is a signal, not proof
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 5: Conflict with correction source (superseded) ---
        if has_conflict and is_correction and b_newer:
            return TensionResult(
                tension_score=0.8,
                relationship=TensionRelationship.CONFLICT,
                action=TensionAction.DEPRECATE_WEAKER,
                confidence=0.9,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 6: Conflict (same slot, different value, strong signal) ---
        if has_conflict and same_topic:
            # High confidence conflict — structural evidence is clear
            score = 0.7 + (similarity - 0.4) * 0.5  # Higher similarity = more tension
            score = max(0.5, min(1.0, score))
            return TensionResult(
                tension_score=score,
                relationship=TensionRelationship.CONFLICT,
                action=(TensionAction.DEPRECATE_WEAKER
                        if trust_delta > TRUST_CONFLICT_DELTA
                        else TensionAction.FLAG_FOR_REVIEW),
                confidence=0.75,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 7: Decayed (low trust + observation or no activity) ---
        if trust_a_low and (is_observation or not same_topic):
            return TensionResult(
                tension_score=0.3,
                relationship=TensionRelationship.DECAYED,
                action=TensionAction.FLAG_FOR_REVIEW,
                confidence=0.7,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 8: Compatible (same topic, no conflicts) ---
        if same_topic and not has_conflict:
            return TensionResult(
                tension_score=0.1,
                relationship=TensionRelationship.COMPATIBLE,
                action=TensionAction.KEEP_BOTH,
                confidence=0.8,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 9: No shared slots but different topic ---
        if no_shared_slots and not same_topic:
            return TensionResult(
                tension_score=0.0,
                relationship=TensionRelationship.UNRELATED,
                action=TensionAction.KEEP_BOTH,
                confidence=0.85,
                supporting_signals=signals,
                slot_overlaps=overlaps,
            )

        # --- Rule 10: Ambiguous — escalate ---
        return TensionResult(
            tension_score=0.4,
            relationship=TensionRelationship.TENSION,
            action=TensionAction.ESCALATE_TO_NLI,
            confidence=0.4,
            supporting_signals=signals,
            slot_overlaps=overlaps,
        )
