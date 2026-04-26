"""Smoke tests for the aether.mcp server.

Verifies the server builds, registers the expected tools, and that
each tool returns sensible output. Skipped automatically if the
optional `mcp` dependency isn't installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

# Skip the whole module if mcp isn't available
pytest.importorskip("mcp")

from aether.mcp.server import build_server
from aether.mcp.state import StateStore


def _extract(call_result):
    """FastMCP returns a list of TextContent. Pull the JSON out of the first."""
    return json.loads(call_result[0].text)


@pytest.fixture
def server(tmp_path):
    state_path = str(tmp_path / "mcp_state.json")
    store = StateStore(state_path=state_path)
    return build_server(store=store)


def _run(coro):
    return asyncio.run(coro)


class TestMcpServer:
    def test_registers_expected_tools(self, server):
        async def listem():
            tools = await server.list_tools()
            return [t.name for t in tools]

        names = _run(listem())
        for expected in (
            "aether_remember",
            "aether_search",
            "aether_sanction",
            "aether_fidelity",
            "aether_context",
        ):
            assert expected in names

    def test_remember_returns_id_and_slots(self, server):
        async def go():
            return await server.call_tool(
                "aether_remember",
                {"text": "I live in Seattle and work at Microsoft"},
            )

        result = _extract(_run(go()))
        assert result["memory_id"].startswith("m")
        assert result["trust"] == 0.7
        assert "extracted_slots" in result

    def test_search_finds_remembered_memory(self, server):
        async def go():
            await server.call_tool(
                "aether_remember", {"text": "User prefers Python over Rust"},
            )
            return await server.call_tool("aether_search", {"query": "Python"})

        result = _extract(_run(go()))
        assert result["result_count"] >= 1
        assert any("Python" in r["text"] for r in result["results"])

    def test_sanction_blocks_overconfident(self, server):
        async def go():
            return await server.call_tool(
                "aether_sanction",
                {
                    "action": "The answer is absolutely and definitively X. I am 100% certain.",
                    "belief_confidence": 0.2,
                },
            )

        result = _extract(_run(go()))
        # Severe overconfidence vs low belief should at minimum HOLD.
        assert result["verdict"] in {"HOLD", "REJECT"}
        assert len(result["annotations"]) >= 1

    def test_sanction_approves_calibrated(self, server):
        async def go():
            return await server.call_tool(
                "aether_sanction",
                {
                    "action": "I think the answer might be X, but I am not fully sure.",
                    "belief_confidence": 0.6,
                },
            )

        result = _extract(_run(go()))
        assert result["verdict"] == "APPROVE"
        assert result["tier"] == "safe"

    def test_fidelity_returns_gap_score(self, server):
        async def go():
            return await server.call_tool(
                "aether_fidelity",
                {
                    "response": "The answer is absolutely and definitively X. I am 100% certain.",
                    "belief_confidence": 0.2,
                },
            )

        result = _extract(_run(go()))
        assert "gap_score" in result
        assert result["gap_score"] > 0.3  # large gap expected

    def test_context_reports_state(self, server):
        async def go():
            await server.call_tool("aether_remember", {"text": "fact one"})
            await server.call_tool("aether_remember", {"text": "fact two"})
            return await server.call_tool("aether_context", {})

        result = _extract(_run(go()))
        assert result["memory_count"] == 2

    def test_state_persists_across_store_restart(self, tmp_path):
        state_path = str(tmp_path / "mcp_state.json")

        store_a = StateStore(state_path=state_path)
        server_a = build_server(store=store_a)

        async def write():
            await server_a.call_tool(
                "aether_remember", {"text": "persistent fact"},
            )

        _run(write())

        # New store from same path
        store_b = StateStore(state_path=state_path)
        assert store_b.stats()["memory_count"] == 1
