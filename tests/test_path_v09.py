"""Tests for v0.9.0: Dijkstra shortest-path retrieval (`aether_path`).

The "park map" tool. Given a query, walks the BDG backward from the
top-1 cosine match and returns the cheapest dependency chain that fits
in a token budget.
"""

from __future__ import annotations

import asyncio
import json

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

from aether.mcp.server import build_server
from aether.mcp.state import StateStore, _estimate_tokens
from aether.memory import EdgeType


def _extract(call_result):
    return json.loads(call_result[0].text)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path):
    s = StateStore(state_path=str(tmp_path / "state.json"))
    if s._encoder is not None:
        s._encoder._load()
    return s


@pytest.fixture
def server(store):
    return build_server(store=store)


# --------------------------------------------------------------------------
# Token estimator
# --------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_returns_one(self):
        assert _estimate_tokens("") == 1

    def test_short_text(self):
        # "hello world" = 11 chars / 4 = 2 tokens (rough)
        assert _estimate_tokens("hello world") >= 1

    def test_scales_linearly(self):
        short = _estimate_tokens("a" * 40)
        long = _estimate_tokens("a" * 400)
        assert long > short * 5  # roughly 10x


# --------------------------------------------------------------------------
# Empty-substrate cases
# --------------------------------------------------------------------------

@needs_networkx
class TestEmptySubstrate:
    def test_empty_substrate_returns_no_substrate(self, store):
        result = store.compute_path("anything")
        assert result["method"] == "no_substrate"
        assert result["path"] == []
        assert result["target"] is None


# --------------------------------------------------------------------------
# Single-node case (target with no ancestors)
# --------------------------------------------------------------------------

@needs_networkx
class TestSingleNode:
    def test_single_memory_path_is_just_target(self, store):
        store.add_memory("the only fact in this substrate", trust=0.9)
        result = store.compute_path("the only fact")
        assert result["method"] == "dijkstra"
        assert result["path_length"] == 1
        assert result["path"][0]["is_target"] is True
        assert result["closed_paths"] == 0


# --------------------------------------------------------------------------
# Multi-hop dependency chain
# --------------------------------------------------------------------------

@needs_networkx
class TestDependencyChain:
    def _build_chain(self, store):
        """Build A -> B -> C where A supports B supports C (target)."""
        a = store.add_memory("Premise A is true", trust=0.95)["memory_id"]
        b = store.add_memory("Therefore B follows", trust=0.9)["memory_id"]
        c = store.add_memory("Conclusion C follows from B", trust=0.85)["memory_id"]
        store.graph.add_edge(a, b, EdgeType.SUPPORTS, {"weight": 0.8})
        store.graph.add_edge(b, c, EdgeType.SUPPORTS, {"weight": 0.8})
        return a, b, c

    def test_path_walks_back_through_supporters(self, store):
        a, b, c = self._build_chain(store)
        result = store.compute_path("Conclusion C follows from B", max_tokens=2000)
        ids = {entry["memory_id"] for entry in result["path"]}
        # All three should be in the path
        assert c in ids
        assert b in ids
        assert a in ids

    def test_target_is_first_in_path(self, store):
        a, b, c = self._build_chain(store)
        result = store.compute_path("Conclusion C follows from B")
        assert result["path"][0]["memory_id"] == c
        assert result["path"][0]["is_target"] is True
        assert result["target"]["memory_id"] == c


# --------------------------------------------------------------------------
# Token budget enforcement
# --------------------------------------------------------------------------

@needs_networkx
class TestBudgetEnforcement:
    def test_budget_caps_path_length(self, store):
        # Build a long chain
        prev = None
        ids = []
        for i in range(10):
            text = f"Step {i}: " + ("padding " * 20)
            mid = store.add_memory(text, trust=0.9)["memory_id"]
            ids.append(mid)
            if prev:
                store.graph.add_edge(prev, mid, EdgeType.SUPPORTS, {"weight": 0.8})
            prev = mid

        # Tight budget should drop later ancestors
        result = store.compute_path(f"Step 9: padding", max_tokens=100)
        assert result["token_cost"] <= 200  # gives a little slack for target inclusion
        assert result["path_length"] >= 1


# --------------------------------------------------------------------------
# CONTRADICTS edges are skipped (closed paths)
# --------------------------------------------------------------------------

@needs_networkx
class TestClosedPaths:
    def test_contradicts_edge_does_not_route_through(self, store):
        a = store.add_memory("Memory A is the contested one", trust=0.5)["memory_id"]
        b = store.add_memory("Memory B is the target", trust=0.9)["memory_id"]
        c = store.add_memory("Memory C supports B legitimately",
                             trust=0.95)["memory_id"]
        # A is connected to B via CONTRADICTS — must NOT route through
        from aether.memory import ContradictionEdge, Disposition
        edge = ContradictionEdge(disposition=Disposition.HELD.value)
        store.graph.add_contradiction(a, b, edge)
        # C supports B legitimately
        store.graph.add_edge(c, b, EdgeType.SUPPORTS, {"weight": 0.9})

        result = store.compute_path("Memory B is the target")
        ids = {entry["memory_id"] for entry in result["path"]}
        # Target B must be there
        assert b in ids
        # C should be reachable via SUPPORTS
        assert c in ids
        # A must NOT be in the path (only reachable via CONTRADICTS)
        assert a not in ids
        # closed_paths counter should be > 0
        assert result["closed_paths"] >= 1


# --------------------------------------------------------------------------
# MCP tool registration + invocation
# --------------------------------------------------------------------------

@needs_networkx
class TestPathMcpTool:
    def test_aether_path_is_registered(self, server):
        async def listem():
            tools = await server.list_tools()
            return [t.name for t in tools]
        names = _run(listem())
        assert "aether_path" in names

    def test_aether_path_returns_well_shaped_result(self, server, store):
        store.add_memory("the only fact", trust=0.9)

        async def go():
            return await server.call_tool(
                "aether_path", {"query": "the only fact", "max_tokens": 1000},
            )

        result = _extract(_run(go()))
        assert "path" in result
        assert "target" in result
        assert "token_cost" in result
        assert "method" in result
        assert "closed_paths" in result
