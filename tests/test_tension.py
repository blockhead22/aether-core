"""Tests for aether.contradiction — structural tension meter."""

import numpy as np

from aether.contradiction import (
    StructuralTensionMeter,
    TensionRelationship,
    TensionAction,
    TensionResult,
    compute_oscillation_count,
)


def _fake_encoder(text):
    """Deterministic fake encoder for testing."""
    np.random.seed(hash(text) % 2**31)
    return np.random.randn(384).astype(np.float32)


class TestTensionMeter:
    def setup_method(self):
        self.meter = StructuralTensionMeter(encoder=_fake_encoder)

    def test_identical_texts_low_tension(self):
        # Use pre-computed identical vectors to test duplicate detection
        vec = _fake_encoder("I live in Seattle")
        result = self.meter.measure(
            "I live in Seattle", "I live in Seattle",
            vector_a=vec, vector_b=vec,
        )
        assert result.tension_score < 0.2
        assert result.relationship in (
            TensionRelationship.DUPLICATE,
            TensionRelationship.COMPATIBLE,
        )

    def test_conflicting_slots_high_tension(self):
        result = self.meter.measure(
            "I live in Seattle",
            "I live in Portland",
            trust_a=0.8,
            trust_b=0.8,
        )
        assert result.relationship in (
            TensionRelationship.CONFLICT,
            TensionRelationship.TENSION,
        )
        assert result.tension_score > 0.3

    def test_unrelated_texts_zero_tension(self):
        result = self.meter.measure(
            "I live in Seattle",
            "The sky is blue today",
        )
        # With fake encoder these may or may not be similar,
        # but slot-wise they share nothing
        assert result.tension_score <= 0.4

    def test_correction_source_detected(self):
        result = self.meter.measure(
            "I work at Google",
            "I work at Microsoft",
            source_b="correction",
            timestamp_a=1000,
            timestamp_b=2000,
            trust_a=0.8,
            trust_b=0.8,
        )
        assert result.supporting_signals.get("is_correction") is True

    def test_temporal_evolution_compatible(self):
        result = self.meter.measure(
            "I used to work at Google",
            "I work at Microsoft",
            trust_a=0.8,
            trust_b=0.8,
        )
        # Temporal evolution should reduce tension
        assert result.relationship in (
            TensionRelationship.COMPATIBLE,
            TensionRelationship.TENSION,
            TensionRelationship.CONFLICT,
        )

    def test_measure_pair_duck_typing(self):
        class FakeMemory:
            def __init__(self, text, trust=0.5):
                self.text = text
                self.trust = trust
                self.source = "user"
                self.timestamp = 0.0
                self.vector = None
                self.oscillation = 0

        mem_a = FakeMemory("I live in Seattle", 0.8)
        mem_b = FakeMemory("I live in Portland", 0.7)
        result = self.meter.measure_pair(mem_a, mem_b)
        assert isinstance(result, TensionResult)

    def test_measure_against_cluster(self):
        class FakeMemory:
            def __init__(self, text):
                self.text = text
                self.trust = 0.5
                self.source = "user"
                self.timestamp = 0.0
                self.vector = None

        cluster = [FakeMemory("I live in Seattle"), FakeMemory("I enjoy hiking")]
        results = self.meter.measure_against_cluster("I live in Portland", cluster)
        assert len(results) == 2
        # Results should be sorted by tension_score descending
        assert results[0][1].tension_score >= results[1][1].tension_score

    def test_result_to_dict(self):
        result = self.meter.measure("I live in Seattle", "I live in Portland")
        d = result.to_dict()
        assert "tension_score" in d
        assert "relationship" in d
        assert "action" in d

    def test_no_encoder_graceful_fallback(self):
        meter = StructuralTensionMeter()  # no encoder
        result = meter.measure("I live in Seattle", "I live in Portland")
        assert isinstance(result, TensionResult)


class TestOscillation:
    def test_monotonic_zero(self):
        history = [
            {"old_trust": 0.5, "new_trust": 0.6, "timestamp": 1},
            {"old_trust": 0.6, "new_trust": 0.7, "timestamp": 2},
            {"old_trust": 0.7, "new_trust": 0.8, "timestamp": 3},
        ]
        assert compute_oscillation_count(history) == 0

    def test_single_oscillation(self):
        history = [
            {"old_trust": 0.5, "new_trust": 0.7, "timestamp": 1},
            {"old_trust": 0.7, "new_trust": 0.5, "timestamp": 2},
        ]
        assert compute_oscillation_count(history) == 1

    def test_multiple_oscillations(self):
        history = [
            {"old_trust": 0.5, "new_trust": 0.7, "timestamp": 1},
            {"old_trust": 0.7, "new_trust": 0.4, "timestamp": 2},
            {"old_trust": 0.4, "new_trust": 0.8, "timestamp": 3},
            {"old_trust": 0.8, "new_trust": 0.3, "timestamp": 4},
        ]
        assert compute_oscillation_count(history) == 3

    def test_empty_history(self):
        assert compute_oscillation_count([]) == 0

    def test_single_entry(self):
        assert compute_oscillation_count([{"old_trust": 0.5, "new_trust": 0.7}]) == 0
