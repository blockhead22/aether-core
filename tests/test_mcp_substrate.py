"""MCP tool surface for substrate v0.14.

Verifies the six aether_substrate_* tools registered in build_server work
end-to-end against an isolated SubstrateGraph (no pollution of the user's
~/.aether/substrate.json).
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from aether.mcp import server as server_mod
from aether.mcp.server import build_server
from aether.mcp.state import StateStore
from aether.substrate import SubstrateGraph


def _extract(call_result):
    return json.loads(call_result[0].text)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def isolated_substrate(tmp_path, monkeypatch):
    """Swap in a fresh SubstrateGraph that persists to tmp_path.

    Resets the module-level singleton both before and after the test so
    no state leaks between tests or into the user's real substrate.
    """
    sub = SubstrateGraph()
    sub.persist_path = str(tmp_path / "substrate.json")
    monkeypatch.setattr(server_mod, "_SUBSTRATE_SINGLETON", sub)
    yield sub
    monkeypatch.setattr(server_mod, "_SUBSTRATE_SINGLETON", None)


@pytest.fixture
def server(tmp_path):
    store = StateStore(state_path=str(tmp_path / "state.json"))
    return build_server(store=store)


class TestObserveAndCurrent:
    def test_observe_then_current(self, server, isolated_substrate):
        async def go():
            await server.call_tool(
                "aether_substrate_observe",
                {
                    "namespace": "user",
                    "slot_name": "location",
                    "value": "Milwaukee",
                    "trust": 0.9,
                },
            )
            return await server.call_tool(
                "aether_substrate_current",
                {"namespace": "user", "slot_name": "location"},
            )

        result = _extract(_run(go()))
        assert result["state"]["value"] == "Milwaukee"
        assert result["state"]["trust"] == 0.9
        assert result["state"]["superseded_by"] is None

    def test_current_unknown_slot_returns_null(self, server, isolated_substrate):
        async def go():
            return await server.call_tool(
                "aether_substrate_current",
                {"namespace": "user", "slot_name": "nope"},
            )

        result = _extract(_run(go()))
        assert result["state"] is None


class TestSupersession:
    def test_value_change_supersedes_prior(self, server, isolated_substrate):
        async def go():
            await server.call_tool(
                "aether_substrate_observe",
                {"namespace": "user", "slot_name": "location", "value": "Chicago"},
            )
            await server.call_tool(
                "aether_substrate_observe",
                {"namespace": "user", "slot_name": "location", "value": "Milwaukee"},
            )
            return await server.call_tool(
                "aether_substrate_history",
                {"namespace": "user", "slot_name": "location"},
            )

        result = _extract(_run(go()))
        assert result["count"] == 2
        # Oldest first, newest last
        assert result["states"][0]["value"] == "Chicago"
        assert result["states"][0]["superseded_by"] is not None
        assert result["states"][1]["value"] == "Milwaukee"
        assert result["states"][1]["superseded_by"] is None


class TestContradictions:
    def test_value_mismatch_surfaces_as_contradiction(
        self, server, isolated_substrate
    ):
        async def go():
            await server.call_tool(
                "aether_substrate_observe",
                {"namespace": "user", "slot_name": "employer", "value": "Anthropic"},
            )
            await server.call_tool(
                "aether_substrate_observe",
                {"namespace": "user", "slot_name": "employer", "value": "Microsoft"},
            )
            return await server.call_tool(
                "aether_substrate_contradictions",
                {"namespace": "user", "threshold": 0.0},
            )

        result = _extract(_run(go()))
        assert result["count"] >= 1
        pair = result["pairs"][0]
        assert pair["slot_id"] == "user:employer"
        assert {pair["a"]["value"], pair["b"]["value"]} == {"Anthropic", "Microsoft"}


class TestSlotsAndStats:
    def test_slots_lists_observed_slots(self, server, isolated_substrate):
        async def go():
            await server.call_tool(
                "aether_substrate_observe",
                {"namespace": "user", "slot_name": "location", "value": "Milwaukee"},
            )
            await server.call_tool(
                "aether_substrate_observe",
                {"namespace": "code", "slot_name": "primary_lang", "value": "python"},
            )
            return await server.call_tool(
                "aether_substrate_slots",
                {"namespace": "user"},
            )

        result = _extract(_run(go()))
        assert result["count"] == 1
        assert result["slots"][0]["slot_id"] == "user:location"
        assert result["slots"][0]["state_count"] == 1

    def test_stats_reflects_writes(self, server, isolated_substrate):
        async def go():
            await server.call_tool(
                "aether_substrate_observe",
                {"namespace": "user", "slot_name": "location", "value": "Milwaukee"},
            )
            return await server.call_tool("aether_substrate_stats", {})

        result = _extract(_run(go()))
        assert result["slots"] == 1
        assert result["states"] == 1
        assert result["observations"] == 1
        assert result["namespace_breakdown"] == {"user": 1}


class TestPersistence:
    def test_observe_persists_to_disk(self, server, isolated_substrate, tmp_path):
        async def go():
            return await server.call_tool(
                "aether_substrate_observe",
                {"namespace": "user", "slot_name": "location", "value": "Milwaukee"},
            )

        _run(go())
        # Reload from disk into a fresh graph and verify the write survived
        reloaded = SubstrateGraph()
        reloaded.load(str(tmp_path / "substrate.json"))
        cur = reloaded.current_state("user", "location")
        assert cur is not None
        assert cur.value == "Milwaukee"
