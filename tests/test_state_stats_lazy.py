"""Regression test for v0.8.0 bug: aether_context (stats) eagerly loaded
the SentenceTransformer model on first access, blocking the cheapest
tool call for ~30-120s on cold start.

Fix: stats() reports `embeddings_available` (encoder configured) and
`embeddings_loaded` (model warm in memory) without triggering a load.
"""

from __future__ import annotations

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

try:
    import sentence_transformers  # noqa: F401
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX,
    reason="networkx required (install [graph] extra)",
)
needs_sentence_transformers = pytest.mark.skipif(
    not _HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers required (install [ml] extra)",
)

from aether.mcp.state import StateStore, _LazyEncoder


@needs_networkx
def test_stats_does_not_block(tmp_path):
    """stats() must return quickly even when the encoder is mid-warmup.

    Before v0.8.1, accessing `_encoder.available` from inside stats()
    eagerly imported torch and the MiniLM weights. The cheapest MCP
    tool (`aether_context`) blocked for tens of seconds on first call.

    This test asserts the actual user-facing property: stats() never
    blocks. We don't assert is_loaded state because v0.8.2's
    auto-warmup combined with the process-wide cache means the
    encoder may have already finished loading by the time we measure
    (especially on subsequent test runs that hit the cached model).
    """
    import time as _time
    store = StateStore(state_path=str(tmp_path / "s.json"))
    assert store._encoder is not None

    t0 = _time.monotonic()
    result = store.stats()
    elapsed = _time.monotonic() - t0

    # stats() must be effectively instantaneous regardless of whether
    # encoder warmup is mid-flight or already done.
    assert elapsed < 0.5, (
        f"stats() took {elapsed:.2f}s; v0.8.0 bug regression — "
        "stats() should never block on encoder loading"
    )
    assert "embeddings_available" in result
    assert "embeddings_loaded" in result
    assert "embeddings_warming" in result
    assert result["embeddings_available"] is True


@needs_networkx
@needs_sentence_transformers
def test_stats_reports_loaded_after_first_encode(tmp_path):
    """Once the encoder finishes loading, stats() reports it.

    Forces synchronous load to keep the test deterministic across
    suite vs isolation runs. Requires sentence-transformers — if [ml]
    isn't installed, _load() is a no-op and `embeddings_loaded` stays
    False.
    """
    store = StateStore(state_path=str(tmp_path / "s.json"))
    initial_loaded = store.stats()["embeddings_loaded"]
    # Force synchronous load -- bypasses background-thread machinery
    store._encoder._load()
    assert store.stats()["embeddings_loaded"] is True, (
        f"Expected loaded=True after _load(); was {initial_loaded} "
        f"initially and still False after sync load."
    )


def test_lazy_encoder_state_props_dont_load():
    """is_loaded and is_unavailable are observation-only — never load."""
    enc = _LazyEncoder()
    assert enc.is_loaded is False
    assert enc.is_unavailable is False
    # Read both several times — still no load.
    for _ in range(5):
        _ = enc.is_loaded
        _ = enc.is_unavailable
    assert enc._model is None  # never loaded
