"""CRT Epistemics -- Belief evolution and trust dynamics.

Error-driven trust evolution through belief backpropagation.
When a belief is corrected, gradients flow backward through the
dependency graph, adjusting trust scores proportionally.

Usage:
    from crt.epistemics import EpistemicLoss, CorrectionEvent, DomainVolatility

    loss_fn = EpistemicLoss()
    event = CorrectionEvent(
        corrected_node_id="mem_123",
        trust_at_assertion=0.9,
        times_corrected=2,
        correction_source="user",
        time_since_assertion=3600,
        domain="employer",
    )
    loss = loss_fn.compute(event)
"""

from .backprop import (
    EpistemicLoss,
    CorrectionEvent,
    BackpropResult,
    DomainVolatility,
    compute_backward_gradients,
    apply_trust_adjustments,
    flat_demotion,
)

__all__ = [
    "EpistemicLoss",
    "CorrectionEvent",
    "BackpropResult",
    "DomainVolatility",
    "compute_backward_gradients",
    "apply_trust_adjustments",
    "flat_demotion",
]
