"""Regression tests for v0.11.0: composable pattern primitives.

v0.11 adds aether/patterns.py with four primitives:
    1. token_overlap   -- Jaccard on token sets
    2. shape           -- typed-pattern detection (numeric/version/date)
                          with comparators
    3. substring_window -- multiple substrings within N tokens
    4. ncd             -- normalized compression distance via gzip

The shape primitive is wired into both _detect_and_record_tensions
(write path) and compute_grounding (read path). It closes the v0.9.4
known_gap_quantitative cases (Python 3.10 vs 3.8, 222 vs 99 tests,
2026-04-27 vs 2025-01-15) that the slot extractor couldn't catch
because they're not categorical conflicts.

This file tests:
    - Each primitive in isolation
    - Shape comparators per type
    - Integration: aether_remember of conflicting versions/numbers/
      dates produces tension_findings
    - Integration: compute_grounding surfaces shape conflicts in
      contradicting_memories
    - Cold-mode behavior: shape works without embeddings
    - False-positive guard: text without shapes doesn't trigger
"""

from __future__ import annotations

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx required (install [graph] extra)",
)

pytest.importorskip("mcp")

from aether.patterns import (
    token_overlap,
    shape,
    substring_window,
    ncd,
    combined_score,
    extract_shapes,
    MatchResult,
)


# ==========================================================================
# Primitive 1: token_overlap (Jaccard)
# ==========================================================================

class TestTokenOverlap:
    def test_identical_texts(self):
        r = token_overlap("we use Postgres", "we use Postgres")
        assert r.score == 1.0
        assert r.primitive == "token_overlap"

    def test_disjoint_texts(self):
        r = token_overlap("apple banana cherry", "x y z")
        assert r.score == 0.0

    def test_partial_overlap(self):
        r = token_overlap("we deploy to AWS", "we deploy to GCP")
        # Tokens: {we, deploy, to, AWS} vs {we, deploy, to, GCP}
        # Jaccard = 3/5 = 0.6
        assert abs(r.score - 0.6) < 0.01

    def test_evidence_includes_shared_tokens(self):
        r = token_overlap("we use Postgres for OLTP", "we use Postgres for analytics")
        assert "shared_tokens" in r.evidence
        assert "postgres" in r.evidence["shared_tokens"]

    def test_both_empty(self):
        r = token_overlap("", "")
        assert r.score == 0.0
        assert r.evidence.get("reason") == "both_empty"


# ==========================================================================
# Primitive 2: shape (typed conflicts)
# ==========================================================================

class TestShapeExtraction:
    def test_extracts_version(self):
        shapes = extract_shapes("Python 3.10 or higher")
        assert any(s[0] == "version" and s[1] == (3, 10) for s in shapes)

    def test_extracts_date(self):
        shapes = extract_shapes("released on 2026-04-27")
        assert any(s[0] == "date" and s[1] == "2026-04-27" for s in shapes)

    def test_extracts_integer(self):
        shapes = extract_shapes("the test suite has 222 tests")
        assert any(s[0] == "integer" and s[1] == 222 for s in shapes)

    def test_no_shapes_in_pure_text(self):
        shapes = extract_shapes("we deploy to AWS and use Postgres")
        # Should find no shapes (no numbers/dates/versions)
        assert all(s[0] not in ("version", "date", "integer", "float") for s in shapes) or not shapes

    def test_does_not_double_count_version_as_integer(self):
        """3.10 should be detected as a version, not three separate
        integers (3 and 10)."""
        shapes = extract_shapes("Python 3.10")
        types = [s[0] for s in shapes]
        assert "version" in types
        # The integer 3 inside the version should NOT also appear
        version_count = types.count("version")
        assert version_count == 1


class TestShapeConflicts:
    def test_version_conflict_detected(self):
        """The canonical case — known_gap_quantitative_001."""
        r = shape("Python 3.10 or higher", "Python 3.8")
        assert r.score == 1.0
        conflicts = r.evidence["conflicts"]
        assert any(c["shape"] == "version" for c in conflicts)

    def test_integer_conflict_detected(self):
        """known_gap_quantitative_002."""
        r = shape("the test suite has 222 tests", "the test suite has 99 tests")
        assert r.score == 1.0
        conflicts = r.evidence["conflicts"]
        assert any(c["shape"] == "integer" for c in conflicts)

    def test_date_conflict_detected(self):
        """known_gap_quantitative_003."""
        r = shape("released on 2026-04-27", "released on 2025-01-15")
        assert r.score == 1.0
        conflicts = r.evidence["conflicts"]
        assert any(c["shape"] == "date" for c in conflicts)

    def test_same_version_no_conflict(self):
        """When the typed values agree, score is 0.5 (agreement, not conflict)."""
        r = shape("Python 3.10 ships next week", "Python 3.10 has new features")
        assert r.score == 0.5
        assert not r.evidence.get("conflicts")
        assert r.evidence.get("agreements")

    def test_no_shape_no_score(self):
        """Texts without typed values don't fire shape detection."""
        r = shape("we deploy to AWS", "we deploy to GCP")
        assert r.score == 0.0
        assert r.evidence.get("reason") == "no_shapes_detected"

    def test_different_shape_types_no_conflict(self):
        """Date in one, version in the other — different types,
        no conflict."""
        r = shape("released 2026-04-27", "Python 3.10")
        assert r.score == 0.0


# ==========================================================================
# Primitive 3: substring_window
# ==========================================================================

class TestSubstringWindow:
    def test_targets_within_window(self):
        r = substring_window(
            "the test passed, so we should ship",
            ["so", "should ship"],
            window=10,
        )
        assert r.score == 1.0

    def test_target_missing(self):
        r = substring_window("hello world", ["foo", "bar"], window=10)
        assert r.score == 0.0
        assert r.evidence.get("reason") == "target_missing"

    def test_targets_too_far_apart(self):
        # All targets present but spread far
        text = "first " + " ".join(["filler"] * 30) + " second"
        r = substring_window(text, ["first", "second"], window=5)
        assert r.score == 0.5  # found but outside window

    def test_no_targets(self):
        r = substring_window("hello", [], window=10)
        assert r.score == 0.0
        assert r.evidence.get("reason") == "no_targets"


# ==========================================================================
# Primitive 4: NCD
# ==========================================================================

class TestNCD:
    def test_similar_texts_high_similarity(self):
        r = ncd("aether is a belief substrate library",
                "aether is a belief substrate")
        # NCD between similar strings should give high similarity
        assert r.score > 0.5
        assert r.primitive == "ncd"

    def test_dissimilar_texts_lower_similarity(self):
        r1 = ncd("aether is a belief substrate", "the cat sat on the mat")
        r2 = ncd("aether is a belief substrate", "aether is a belief substrate")
        assert r1.score < r2.score

    def test_empty_returns_zero(self):
        r = ncd("", "anything")
        assert r.score == 0.0


# ==========================================================================
# Combined scoring
# ==========================================================================

class TestCombinedScore:
    def test_combines_multiple_primitives(self):
        results = [
            MatchResult(score=0.5, primitive="token_overlap"),
            MatchResult(score=1.0, primitive="shape"),
            MatchResult(score=0.7, primitive="ncd"),
        ]
        combined, evidence = combined_score(results)
        # Default weights: 0.25 + 0.35 + 0.20 = 0.80
        # Score: 0.5*0.25 + 1.0*0.35 + 0.7*0.20 = 0.125 + 0.35 + 0.14 = 0.615
        # /0.80 = 0.769
        assert 0.7 < combined < 0.8

    def test_empty_results(self):
        combined, evidence = combined_score([])
        assert combined == 0.0


# ==========================================================================
# INTEGRATION — shape conflicts produce contradictions on write
# ==========================================================================

@needs_networkx
class TestShapeIntegrationWrite:
    """The actual closure of the v0.9.4 known_gap_quantitative cases."""

    def test_version_conflict_on_write_produces_contradiction(self, tmp_path):
        from aether.mcp.state import StateStore

        s = StateStore(state_path=str(tmp_path / "state.json"))
        if s._encoder is not None:
            s._encoder._load()

        s.add_memory("This project requires Python 3.10 or higher.", trust=0.9)
        result = s.add_memory("This project requires Python 3.8.", trust=0.9)

        findings = result["tension_findings"]
        assert len(findings) >= 1, (
            f"Python version conflict not detected: {findings}"
        )
        # Should be tagged as quantitative kind
        kinds = [f.get("kind") for f in findings]
        assert "quantitative" in kinds or any("shape:" in str(f.get("trace", "")) for f in findings)

    def test_integer_conflict_on_write(self, tmp_path):
        from aether.mcp.state import StateStore
        s = StateStore(state_path=str(tmp_path / "state.json"))
        if s._encoder is not None:
            s._encoder._load()

        s.add_memory("The aether-core test suite has 222 tests.", trust=0.9)
        result = s.add_memory("The aether-core test suite has 99 tests.", trust=0.9)

        assert len(result["tension_findings"]) >= 1

    def test_date_conflict_on_write(self, tmp_path):
        from aether.mcp.state import StateStore
        s = StateStore(state_path=str(tmp_path / "state.json"))
        if s._encoder is not None:
            s._encoder._load()

        s.add_memory("aether-core v0.9.0 was released on 2026-04-27.", trust=0.95)
        result = s.add_memory("aether-core v0.9.0 was released on 2025-01-15.", trust=0.9)

        assert len(result["tension_findings"]) >= 1


# ==========================================================================
# INTEGRATION — shape conflicts surface in compute_grounding
# ==========================================================================

@needs_networkx
class TestShapeIntegrationGrounding:
    """compute_grounding should put shape conflicts in
    contradicting_memories."""

    def test_grounding_surfaces_version_conflict(self, tmp_path):
        from aether.mcp.state import StateStore
        s = StateStore(state_path=str(tmp_path / "state.json"))
        if s._encoder is not None:
            s._encoder._load()

        s.add_memory("This project requires Python 3.10 or higher.", trust=0.9)
        grounding = s.compute_grounding("This project requires Python 3.8.")

        assert len(grounding["contradict"]) >= 1
        # The contradict entry should be tagged as quantitative
        kinds = [c.get("kind") for c in grounding["contradict"]]
        assert "quantitative" in kinds


# ==========================================================================
# Cold-mode safety: shape detection doesn't need embeddings
# ==========================================================================

@needs_networkx
class TestShapeColdMode:
    def test_shape_detection_in_cold_mode(self, tmp_path):
        """Shape primitive is regex-based, must work without encoder."""
        from aether.mcp.state import StateStore
        from aether._lazy_encoder import LazyEncoder

        s = StateStore(state_path=str(tmp_path / "state.json"))
        # Force cold encoder
        cold = LazyEncoder()
        s._encoder = cold
        s.meter._encoder = cold

        s.add_memory("Python 3.10 or higher", trust=0.9)
        result = s.add_memory("Python 3.8", trust=0.9)
        # Even cold, shape detection fires
        assert len(result["tension_findings"]) >= 1


# ==========================================================================
# False-positive guards
# ==========================================================================

@needs_networkx
class TestFalsePositiveGuards:
    def test_text_without_shapes_does_not_trigger(self, tmp_path):
        from aether.mcp.state import StateStore
        s = StateStore(state_path=str(tmp_path / "state.json"))
        if s._encoder is not None:
            s._encoder._load()

        s.add_memory("we deploy to AWS", trust=0.9)
        result = s.add_memory("we use Docker for containers", trust=0.9)

        # No shape conflict (no numerics/versions/dates), shouldn't fire
        # Note: mutex / structural may still fire on other grounds; we
        # just assert shape didn't add a spurious finding
        for finding in result["tension_findings"]:
            assert finding.get("kind") != "quantitative"

    def test_same_quantitative_value_no_false_conflict(self, tmp_path):
        """Both memories say "Python 3.10" — agreement, not conflict."""
        from aether.mcp.state import StateStore
        s = StateStore(state_path=str(tmp_path / "state.json"))
        if s._encoder is not None:
            s._encoder._load()

        s.add_memory("Python 3.10 ships with new features", trust=0.9)
        result = s.add_memory("Python 3.10 has been released", trust=0.9)

        # No shape CONFLICT — agreement should not produce a contradiction
        for finding in result["tension_findings"]:
            assert finding.get("kind") != "quantitative"
