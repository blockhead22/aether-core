"""Regression net for the CRT math port (2026-05-01).

Locks in the public surface of `aether.crt` so future refactors don't
silently break the fact-checker / disclosure-policy ports that will
import from this module. Not aiming for behavioral coverage of every
threshold — that lives in the original personal_agent test suite. Just
verifying that:

    1. The public exports import cleanly.
    2. encode_vector falls back to a real (non-zero) hash vector when
       the [ml] extra is unavailable.
    3. Core math (similarity, drift_meaning, compute_volatility,
       should_reflect, detect_contradiction) returns plausible values
       on canonical inputs.
    4. BetaTrust conjugate updates produce the expected mean shift.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.crt import (
    BetaTrust,
    CRTConfig,
    CRTMath,
    MemorySource,
    SSEMode,
    encode_vector,
    extract_emotion_intensity,
    extract_future_relevance,
)


@pytest.fixture
def crt() -> CRTMath:
    return CRTMath(CRTConfig())


def test_public_surface_imports():
    """All names declared in __all__ are importable and callable."""
    from aether.crt import __all__ as exported
    for name in exported:
        assert name in dir(__import__("aether.crt", fromlist=[name])), (
            f"declared export {name} not present at runtime"
        )


def test_encode_vector_hash_fallback_is_deterministic_and_nonzero():
    """The hash fallback must produce a real unit vector, not the
    all-signed-zeros bug present in the original implementation.
    """
    v1 = encode_vector("the cat sat on the mat")
    v2 = encode_vector("the cat sat on the mat")
    v3 = encode_vector("totally different sentence")

    assert v1.dtype == np.float32
    assert v1.shape == v2.shape == v3.shape

    norm = float(np.linalg.norm(v1))
    assert norm == pytest.approx(1.0, abs=1e-5), (
        f"encode_vector should return a unit-norm vector; got norm={norm}"
    )

    assert np.allclose(v1, v2), "encode_vector must be deterministic"
    assert not np.allclose(v1, v3), (
        "different inputs must produce different vectors"
    )


def test_similarity_self_is_one(crt: CRTMath):
    v = encode_vector("anchor sentence")
    assert crt.similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_similarity_dimension_mismatch_returns_zero(crt: CRTMath):
    a = np.ones(8, dtype=np.float32)
    b = np.ones(16, dtype=np.float32)
    assert crt.similarity(a, b) == 0.0


def test_drift_meaning_inverse_of_similarity(crt: CRTMath):
    a = encode_vector("foo")
    b = encode_vector("bar")
    sim = crt.similarity(a, b)
    drift = crt.drift_meaning(a, b)
    assert drift == pytest.approx(1.0 - sim, abs=1e-6)


def test_compute_volatility_monotonic_in_drift(crt: CRTMath):
    """Higher drift, holding everything else equal, should not decrease
    volatility — the function should be monotonic in its drift argument.
    """
    low = crt.compute_volatility(
        drift=0.1, memory_alignment=0.5,
        is_contradiction=False, is_fallback=False,
    )
    high = crt.compute_volatility(
        drift=0.9, memory_alignment=0.5,
        is_contradiction=False, is_fallback=False,
    )
    assert high > low


def test_should_reflect_threshold(crt: CRTMath):
    cfg = crt.config
    assert crt.should_reflect(cfg.theta_reflect + 0.01) is True
    assert crt.should_reflect(cfg.theta_reflect - 0.01) is False


def test_beta_trust_aligned_increases_mean():
    bt = BetaTrust()
    initial = bt.mean
    bt.update_aligned()
    bt.update_aligned()
    assert bt.mean > initial


def test_beta_trust_contradicted_decreases_mean():
    bt = BetaTrust()
    initial = bt.mean
    bt.update_contradicted()
    bt.update_contradicted()
    assert bt.mean < initial


def test_detect_contradiction_entity_swap(crt: CRTMath):
    """The fast-path entity-swap heuristic fires on proper-noun values.
    `_looks_like_entity` requires at least one capitalized token, so this
    path is for things like employer / city / product / person, not for
    common-noun slots like color or food (those go through aether-core's
    slot-conflict layer instead).
    """
    is_contra, reason = crt.detect_contradiction(
        drift=0.1,
        confidence_new=0.9,
        confidence_prior=0.9,
        source=MemorySource.USER,
        text_new="The user works at Anthropic",
        text_prior="The user works at Google",
        slot="employer",
        value_new="Anthropic",
        value_prior="Google",
    )
    assert is_contra is True
    assert "swap" in reason.lower() or "entity" in reason.lower()


def test_detect_contradiction_paraphrase_tolerance(crt: CRTMath):
    """Same meaning expressed differently — without preference verbs that
    would trigger `_is_boolean_inversion` — should not be flagged at
    moderate drift. Verifies the paraphrase check actually gets invoked
    in the rule chain.
    """
    is_contra, reason = crt.detect_contradiction(
        drift=0.20,
        confidence_new=0.9,
        confidence_prior=0.9,
        source=MemorySource.USER,
        text_new="The capital of France is Paris.",
        text_prior="Paris is the capital of France.",
    )
    assert is_contra is False, f"paraphrase falsely flagged: {reason!r}"


def test_detect_contradiction_high_drift_fires(crt: CRTMath):
    """Above-threshold drift on truly different statements should fire
    the contradiction even when no slot/entity heuristic applies.
    """
    cfg = crt.config
    is_contra, reason = crt.detect_contradiction(
        drift=cfg.theta_contra + 0.1,
        confidence_new=0.9,
        confidence_prior=0.9,
        source=MemorySource.USER,
        text_new="The Earth orbits the Sun.",
        text_prior="The team meeting is on Tuesday.",
    )
    assert is_contra is True
    assert reason


def test_extract_emotion_intensity_bounds():
    assert 0.0 <= extract_emotion_intensity("hello world") <= 1.0
    high = extract_emotion_intensity("I LOVE THIS!!!")
    low = extract_emotion_intensity("the report is on the desk")
    assert high > low


def test_extract_future_relevance_bounds():
    assert 0.0 <= extract_future_relevance("hello") <= 1.0
    plan = extract_future_relevance("remember to call tomorrow")
    flat = extract_future_relevance("the report is on the desk")
    assert plan > flat


def test_sse_mode_enum_has_three_modes():
    assert {m.value for m in SSEMode} == {"L", "C", "H"}


def test_memory_source_enum_includes_user_and_fallback():
    values = {s.value for s in MemorySource}
    assert "user" in values
    assert "fallback" in values
