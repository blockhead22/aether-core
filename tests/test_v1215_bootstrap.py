"""v0.12.15: aether_bootstrap MCP tool — Track 1 follow-up for Desktop.

Claude Desktop (and other MCP clients without a hooks system) can't
benefit from the SessionStart auto-everything that v0.12.14 shipped.
On Desktop, the user has to run `aether init` and `aether warmup` from
a terminal, then edit ~/Library/Application Support/Claude/
claude_desktop_config.json by hand. The terminal step is the friction
point — many users won't do it.

aether_bootstrap moves the bootstrap into the MCP surface itself:
the user can just say "set up aether" once after wiring the MCP
config, and Claude calls the tool. No terminal needed beyond
`pip install`.

Contract:
  - Idempotent. Calling twice does not double-seed.
  - Reports what it did (seeded count, encoder mode, memory count).
  - Triggers warmup but doesn't block on it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from aether.mcp.server import build_server
from aether.mcp.state import StateStore


@pytest.fixture
def store(tmp_path):
    """Fresh substrate per test. Encoder pre-loaded so seeded memories
    get real embeddings (mirrors the live SessionStart flow)."""
    s = StateStore(state_path=str(tmp_path / "state.json"))
    if s._encoder is not None and hasattr(s._encoder, "_load"):
        s._encoder._load()
    return s


def _run(coro):
    return asyncio.run(coro)


def _extract(call_result):
    return json.loads(call_result[0].text)


class TestBootstrapSeeding:
    def test_seeds_seven_defaults_on_empty_substrate(self, store):
        server = build_server(store=store)
        result = _extract(_run(server.call_tool("aether_bootstrap", {})))
        assert result["seeded_default_beliefs"] == 7
        assert result["default_beliefs_already_present"] is False
        assert result["memory_count"] == 7

    def test_idempotent_on_second_call(self, store):
        server = build_server(store=store)
        first = _extract(_run(server.call_tool("aether_bootstrap", {})))
        assert first["seeded_default_beliefs"] == 7

        second = _extract(_run(server.call_tool("aether_bootstrap", {})))
        # Second call sees the existing beliefs, doesn't re-seed.
        assert second["seeded_default_beliefs"] == 0
        assert second["default_beliefs_already_present"] is True
        # Memory count stays at 7, not 14.
        assert second["memory_count"] == 7

    def test_skips_if_any_default_policy_belief_present(self, store):
        # Simulate a partially-seeded substrate (e.g. user previously ran
        # `aether init`, then later wired Desktop and called bootstrap).
        store.add_memory(
            text="Never force-push to main or master branches.",
            trust=0.95,
            source="default_policy",
        )
        server = build_server(store=store)
        result = _extract(_run(server.call_tool("aether_bootstrap", {})))
        # Bootstrap sees the source:default_policy tag, doesn't re-seed.
        assert result["seeded_default_beliefs"] == 0
        assert result["default_beliefs_already_present"] is True
        assert result["memory_count"] == 1


class TestBootstrapReporting:
    def test_returns_aether_version(self, store):
        from aether import __version__
        server = build_server(store=store)
        result = _extract(_run(server.call_tool("aether_bootstrap", {})))
        assert result["aether_version"] == __version__

    def test_returns_state_path(self, store, tmp_path):
        server = build_server(store=store)
        result = _extract(_run(server.call_tool("aether_bootstrap", {})))
        assert result["state_path"] == str(tmp_path / "state.json")

    def test_returns_encoder_mode(self, store):
        server = build_server(store=store)
        result = _extract(_run(server.call_tool("aether_bootstrap", {})))
        # Encoder was pre-loaded in the fixture, so mode should be 'warm'.
        # (Falls through to 'cold' if [ml] extra isn't installed —
        # bootstrap still succeeds.)
        assert result["encoder_mode"] in ("warm", "warming", "cold", "unavailable")

    def test_returns_next_steps_guidance(self, store):
        server = build_server(store=store)
        result = _extract(_run(server.call_tool("aether_bootstrap", {})))
        assert "next_steps" in result
        # Mentions the key tool names so the model knows what to call.
        text = result["next_steps"]
        assert "aether_remember" in text
        assert "aether_sanction" in text


class TestBootstrapInteractsCorrectly:
    """After bootstrap runs, the rest of the substrate's contracts
    should hold — sanction blocks force-push, search finds beliefs,
    etc. Smoke tests that bootstrap leaves the substrate in a good
    state, not a half-seeded one.
    """

    def test_sanction_blocks_force_push_after_bootstrap(self, store):
        server = build_server(store=store)
        _run(server.call_tool("aether_bootstrap", {}))

        # F#11 contract: force-push action grounds against the seeded
        # "Never force-push" belief.
        grounding = store.compute_grounding("git push --force origin main")
        contradicting = grounding.get("contradict", [])
        assert len(contradicting) >= 1, (
            f"bootstrap left substrate in a state where sanction can't "
            f"block force-push. Grounding: {grounding}"
        )

    def test_search_finds_seeded_beliefs(self, store):
        server = build_server(store=store)
        _run(server.call_tool("aether_bootstrap", {}))

        results = store.search("force push", limit=5)
        assert len(results) >= 1
        assert any("force-push" in r["text"].lower() for r in results)
