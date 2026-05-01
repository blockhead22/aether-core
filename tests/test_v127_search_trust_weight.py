"""F#10 fix: search() weights combined score by trust.

Background — F#10 surfaced 2026-04-30 evening:
    After the AI_round2 production import landed (substrate grew 45 -> 127
    nodes), running `aether_search "user favorite color"` returned ALL 9
    distinct color memories — but ranked the demoted entry (red, trust=0.67)
    *above* the trust=0.95 truths. Cosine similarity preferred the bare
    "user favorite color: red" over the verbose "user favorite color: cyan
    (observed 13x in production)" because the suffix dilutes the embedding.
    The score function `0.7*sim + 0.3*substring` had no trust term, so a
    demoted-but-short memory could outrank a trusted-but-annotated truth.

The fix:
    A new constant `SEARCH_TRUST_WEIGHT` (default 0.7) controls how much
    trust shapes the rank. The score is multiplied by
    `(1 - w) + w * trust` so trust=1 leaves the score unchanged, trust=0
    drops it to (1-w) of the unweighted score. Relevant low-trust matches
    still surface; current-truth high-trust matches rise.

These tests prove the contract:
    - In substring-only (cold) mode, two entries that tie on substring
      score now rank by trust.
    - In cosine mode with a deterministic fake encoder, a high-trust entry
      with slightly-lower cosine similarity outranks a low-trust entry
      with slightly-higher similarity.
    - The constant is tunable: setting SEARCH_TRUST_WEIGHT=0 reverts to
      the old behavior (regression-resistance for downstream callers).
"""

from __future__ import annotations

import numpy as np
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

from aether.mcp import state as state_mod
from aether.mcp.state import StateStore


@needs_networkx
class TestSearchTrustWeightSubstring:
    """Substring/cold mode: trust differentiates entries that tie on text."""

    def test_high_trust_ranks_above_low_trust_when_substring_ties(self, tmp_path):
        s = StateStore(
            state_path=str(tmp_path / "state.json"),
            enable_embeddings=False,
        )
        s.add_memory(text="user favorite color: red", trust=0.67)
        s.add_memory(text="user favorite color: blue", trust=0.95)
        s.add_memory(text="user favorite color: green", trust=0.95)

        results = s.search("user favorite color", limit=5)
        assert len(results) == 3

        # The 0.67 entry must not rank first.
        assert results[0]["trust"] >= 0.95, (
            f"Expected trust>=0.95 first, got {results[0]['trust']:.2f}: "
            f"{results[0]['text']!r}"
        )
        assert results[-1]["trust"] == pytest.approx(0.67), (
            f"Expected trust=0.67 last, got {results[-1]['trust']:.2f}: "
            f"{results[-1]['text']!r}"
        )

    def test_demoted_trust_zero_still_appears_but_at_bottom(self, tmp_path):
        """trust=0 memories are not zeroed out (they may still be the
        only relevant match), but the multiplicative floor ensures they
        rank below anything with positive trust."""
        s = StateStore(
            state_path=str(tmp_path / "state.json"),
            enable_embeddings=False,
        )
        s.add_memory(text="user favorite color: red", trust=0.0)
        s.add_memory(text="user favorite color: blue", trust=0.5)

        results = s.search("user favorite color", limit=5)
        assert len(results) == 2
        assert results[0]["trust"] == pytest.approx(0.5)
        assert results[1]["trust"] == pytest.approx(0.0)
        # The trust=0 entry's score is non-zero (still surfaced).
        assert results[1]["score"] > 0


@needs_networkx
class TestSearchTrustWeightCosine:
    """Cosine mode: trust beats a small similarity advantage from a
    demoted entry whose bare text happens to align with the query."""

    def test_high_trust_overcomes_small_sim_advantage(self, tmp_path):
        # Build deterministic embeddings such that the low-trust entry
        # has a SLIGHTLY higher cosine similarity to the query than the
        # high-trust one (mimicking the production case where the verbose
        # truth's embedding is diluted by an annotation suffix).
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Low-trust: cos(query) ~= 0.95
        low_trust_vec = np.array([0.95, 0.31, 0.0], dtype=np.float32)
        # High-trust: cos(query) ~= 0.85
        high_trust_vec = np.array([0.85, 0.53, 0.0], dtype=np.float32)

        text_to_vec = {
            "user favorite color: red": low_trust_vec,
            "user favorite color: cyan (observed 13x in production)": high_trust_vec,
        }

        class StubEncoder:
            is_loaded = True
            is_warming = False
            is_unavailable = False

            def encode(self, text):
                if text == "user favorite color":
                    return query_vec
                return text_to_vec.get(text, np.zeros(3, dtype=np.float32))

            def start_warmup(self):
                pass

        s = StateStore(state_path=str(tmp_path / "state.json"))
        s._encoder = StubEncoder()

        # add_memory will call encode() on the text and store the embedding.
        s.add_memory(text="user favorite color: red", trust=0.67)
        s.add_memory(
            text="user favorite color: cyan (observed 13x in production)",
            trust=0.95,
        )

        results = s.search("user favorite color", limit=5)
        assert len(results) == 2
        assert results[0]["trust"] == pytest.approx(0.95), (
            f"Expected trust=0.95 first (high-trust truth), got "
            f"trust={results[0]['trust']:.2f} text={results[0]['text']!r}. "
            f"Sim values: {[r['similarity'] for r in results]}"
        )
        # Sanity-check that the low-trust entry's raw similarity was
        # actually higher — that's what makes this the F#10 scenario.
        sim_low = next(r["similarity"] for r in results if r["trust"] < 0.7)
        sim_high = next(r["similarity"] for r in results if r["trust"] > 0.9)
        assert sim_low > sim_high, (
            f"Test setup invariant: low-trust sim ({sim_low}) must exceed "
            f"high-trust sim ({sim_high}); otherwise the test isn't "
            f"actually testing the F#10 bug."
        )


@needs_networkx
class TestSearchTrustWeightTunable:
    """The trust weight is a module-level constant; setting it to 0
    reverts to the legacy ranking. Downstream callers that depend on
    the old behavior have an escape hatch."""

    def test_zero_weight_reverts_to_legacy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state_mod, "SEARCH_TRUST_WEIGHT", 0.0)

        s = StateStore(
            state_path=str(tmp_path / "state.json"),
            enable_embeddings=False,
        )
        # Two entries, identical substring score, very different trust.
        # With weight=0, ranking is whatever the underlying graph order
        # produces — NOT trust-sorted. We just check both come back
        # with the same score (the legacy invariant we're preserving).
        s.add_memory(text="alpha topic match", trust=0.95)
        s.add_memory(text="alpha topic match", trust=0.10)

        results = s.search("alpha topic match", limit=5)
        assert len(results) == 2
        assert results[0]["score"] == pytest.approx(results[1]["score"])
