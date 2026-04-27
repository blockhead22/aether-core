"""Regression test for v0.8.0 bug: aether_context (stats) eagerly loaded
the SentenceTransformer model on first access, blocking the cheapest
tool call for ~30-120s on cold start.

Fix: stats() reports `embeddings_available` (encoder configured) and
`embeddings_loaded` (model warm in memory) without triggering a load.
"""

from __future__ import annotations

import pytest

from aether.mcp.state import StateStore, _LazyEncoder


def test_stats_does_not_force_encoder_load(tmp_path):
    """stats() must NOT trigger SentenceTransformer load.

    Before the fix, accessing `_encoder.available` from inside stats()
    eagerly imported torch and the MiniLM weights. This caused the
    cheapest MCP tool (`aether_context`) to block for tens of seconds
    on first invocation.
    """
    store = StateStore(state_path=str(tmp_path / "s.json"))

    # Encoder is wired up but model is NOT loaded yet.
    assert store._encoder is not None
    assert store._encoder.is_loaded is False, (
        "encoder shouldn't be loaded before any tool needs it"
    )

    # The bug: calling stats() forced a load.
    result = store.stats()

    # After the fix: stats() reports configured/loaded without loading.
    assert store._encoder.is_loaded is False, (
        "stats() must not trigger encoder load"
    )
    assert "embeddings_available" in result
    assert "embeddings_loaded" in result
    # `embeddings_available` (configured) is True even when not loaded.
    assert result["embeddings_available"] is True
    # `embeddings_loaded` is False until something actually encodes.
    assert result["embeddings_loaded"] is False


def test_stats_reports_loaded_after_first_encode(tmp_path):
    """Once a tool that encodes runs, stats() should reflect the load."""
    store = StateStore(state_path=str(tmp_path / "s.json"))
    assert store.stats()["embeddings_loaded"] is False

    # Trigger an encode through search (only does work if there's data,
    # but the .available check inside _encode is what loads the model)
    store.add_memory("test fact", trust=0.7)
    store.search("test")  # this calls _encode under the hood

    assert store.stats()["embeddings_loaded"] is True


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
