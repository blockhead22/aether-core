"""CRT Contradiction -- Structural tension detection and contradiction lifecycle.

Measures tension between beliefs using structural signals (slot extraction,
embedding similarity, metadata comparison) without LLM calls.

Usage:
    from crt.contradiction import StructuralTensionMeter, TensionRelationship

    meter = StructuralTensionMeter(encoder=my_encoder)
    result = meter.measure("I live in Seattle", "I live in Portland")
    print(result.relationship)  # TensionRelationship.CONFLICT
    print(result.tension_score)  # 0.7+
"""

from .tension import (
    StructuralTensionMeter,
    TensionResult,
    TensionRelationship,
    TensionAction,
    SlotOverlap,
    compute_oscillation_count,
    EMBEDDING_DUPLICATE_THRESHOLD,
    EMBEDDING_SAME_TOPIC_THRESHOLD,
    EMBEDDING_UNRELATED_THRESHOLD,
    TRUST_DECAY_THRESHOLD,
    TRUST_CONFLICT_DELTA,
    OSCILLATION_FLAG_THRESHOLD,
    OSCILLATION_TENSION_BOOST,
)

__all__ = [
    "StructuralTensionMeter",
    "TensionResult",
    "TensionRelationship",
    "TensionAction",
    "SlotOverlap",
    "compute_oscillation_count",
    "EMBEDDING_DUPLICATE_THRESHOLD",
    "EMBEDDING_SAME_TOPIC_THRESHOLD",
    "EMBEDDING_UNRELATED_THRESHOLD",
    "TRUST_DECAY_THRESHOLD",
    "TRUST_CONFLICT_DELTA",
    "OSCILLATION_FLAG_THRESHOLD",
    "OSCILLATION_TENSION_BOOST",
]
