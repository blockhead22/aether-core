"""CRT (Cognitive-Reflective Transformer) math layer.

Ported from `personal_agent/crt_core.py` on 2026-05-01 as part of the
"full-wire CRT into aether-core OSS" decision (see HANDOFF for context).
This module is the mathematical substrate that powers volatility-aware
fact-checking, contradiction triage, and trust evolution.

Public surface (re-exported for `from aether.crt import ...` ergonomics):

    BetaTrust         — Beta(alpha, beta) trust distribution with
                        conjugate Bayesian updates.
    SSEMode           — LOSSLESS / COGNI / HYBRID compression modes.
    MemorySource      — taxonomy used for trust caps + safety gates.
    CRTConfig         — all thresholds + learnable parameters in one place.
    CRTMath           — the load-bearing class: similarity, drift_meaning,
                        compute_volatility, should_reflect, detect_contradiction,
                        evolve_trust_*, classify_fact_change, etc.
    encode_vector     — text → embedding (real if [ml] extra warm, hash
                        fallback otherwise — see core.py docstring).
    extract_emotion_intensity, extract_future_relevance — heuristic helpers
                        for SSE significance scoring.

Nothing here imports from `aether.mcp` or `aether.memory`, keeping `crt`
a leaf module that downstream tools (disclosure policy, governance) can
layer on top of.

Fact-checker (``aether.crt.fact_checker``):
    compute_finding_volatility — CRT volatility scoring for findings
    crt_filter_contradiction   — second-pass false-positive filter
    run_verification           — GroundCheck integration (optional dep)
    get_pending_fact_checks    — query stored findings by volatility
    resolve_fact_check         — mark a finding resolved
"""

from aether.crt.core import (
    BetaTrust,
    SSEMode,
    MemorySource,
    CRTConfig,
    CRTMath,
    encode_vector,
    extract_emotion_intensity,
    extract_future_relevance,
)

from aether.crt.fact_checker import (
    compute_finding_volatility,
    crt_filter_contradiction,
    run_verification,
    get_pending_fact_checks,
    resolve_fact_check,
)

__all__ = [
    "BetaTrust",
    "SSEMode",
    "MemorySource",
    "CRTConfig",
    "CRTMath",
    "encode_vector",
    "extract_emotion_intensity",
    "extract_future_relevance",
    "compute_finding_volatility",
    "crt_filter_contradiction",
    "run_verification",
    "get_pending_fact_checks",
    "resolve_fact_check",
]
