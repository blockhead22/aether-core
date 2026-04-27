"""Regression tests for v0.8.2: encoder warmup must never block a tool call.

v0.8.0 had `aether_context` blocking on cold start.
v0.8.1 fixed `aether_context` but `aether_search` still blocked for
several minutes on the SentenceTransformer import + load.
v0.8.2 makes the encoder load run in a background thread; encode()
returns None during warmup; search() falls back to substring.
"""

from __future__ import annotations

import time

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX,
    reason="networkx required (install [graph] extra)",
)

from aether.mcp.state import StateStore, _LazyEncoder


# --------------------------------------------------------------------------
# _LazyEncoder direct tests (no networkx needed)
# --------------------------------------------------------------------------

class TestLazyEncoderNonBlocking:
    def test_encode_returns_none_before_warmup(self):
        """A fresh encoder hasn't loaded — encode() must NOT block."""
        enc = _LazyEncoder()
        t0 = time.monotonic()
        result = enc.encode("hello")
        elapsed = time.monotonic() - t0
        # Should be effectively instant — well under a second
        assert elapsed < 0.1, f"encode() took {elapsed:.2f}s, expected near-zero"
        assert result is None

    def test_start_warmup_returns_immediately(self):
        """start_warmup() must spawn the load on a background thread."""
        enc = _LazyEncoder()
        t0 = time.monotonic()
        enc.start_warmup()
        elapsed = time.monotonic() - t0
        # Should be effectively instant
        assert elapsed < 0.5, f"start_warmup() took {elapsed:.2f}s, expected near-zero"

    def test_warmup_idempotent(self):
        """Calling start_warmup() multiple times must be safe."""
        enc = _LazyEncoder()
        enc.start_warmup()
        enc.start_warmup()
        enc.start_warmup()
        # No crash, no error.

    def test_is_warming_true_immediately_after_warmup_start(self):
        """During the warmup window, is_warming reflects the in-flight load."""
        enc = _LazyEncoder()
        # Before warmup: not warming, not loaded, not unavailable
        assert enc.is_warming is False
        assert enc.is_loaded is False
        # After kicking off warmup it should report as warming OR have
        # already finished (very fast machine). Either is correct;
        # what's NOT correct is "not warming AND not loaded AND
        # not unavailable" — that means we lost track of it.
        enc.start_warmup()
        # Without sentence-transformers installed, warmup will mark
        # the encoder unavailable nearly instantly. We just need
        # this to not crash and the state to be consistent.
        time.sleep(0.05)
        # Either warming, loaded, or unavailable -- never just stuck
        # in "configured but doing nothing"
        states = (enc.is_warming, enc.is_loaded, enc.is_unavailable)
        assert any(states), f"encoder is in limbo: warming/loaded/unavailable = {states}"


# --------------------------------------------------------------------------
# StateStore behavior under warmup
# --------------------------------------------------------------------------

@needs_networkx
class TestStateStoreUnderWarmup:
    def test_init_kicks_off_warmup(self, tmp_path):
        """Building a StateStore must start the encoder warmup."""
        store = StateStore(state_path=str(tmp_path / "s.json"))
        # _warmup_started should be True (the thread may have already
        # finished on a fast machine, so we can't assert is_warming)
        assert store._encoder._warmup_started is True

    def test_search_does_not_block_during_warmup(self, tmp_path):
        """First search() call must NOT block on encoder load.

        Hard regression for v0.8.0/0.8.1 — the search was synchronous
        on cold-start and could take minutes.
        """
        store = StateStore(state_path=str(tmp_path / "s.json"))
        store.add_memory("a test memory", trust=0.7)

        t0 = time.monotonic()
        results = store.search("test")
        elapsed = time.monotonic() - t0

        # Even if the model is mid-warmup, search must return quickly.
        # 2 seconds is generous — substring fallback is sub-millisecond.
        assert elapsed < 2.0, (
            f"search() took {elapsed:.2f}s; encoder warmup is blocking"
        )
        assert len(results) >= 1

    def test_stats_reports_warmup_state(self, tmp_path):
        """stats() must surface embeddings_warming."""
        store = StateStore(state_path=str(tmp_path / "s.json"))
        result = store.stats()
        assert "embeddings_warming" in result
        assert "embeddings_loaded" in result
        assert "embeddings_available" in result
