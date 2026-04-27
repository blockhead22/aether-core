"""Tests for v0.5.0 additions to the aether.mcp surface.

Covers:
    - Substrate-grounded fidelity / sanction
    - Embedding-aware search (or substring fallback when ML extra absent)
    - Contradiction detection on remember
    - aether_correct + cascade
    - aether_lineage
    - aether_cascade_preview
    - aether_contradictions / aether_resolve
    - aether_belief_history
    - aether_session_diff
    - aether_memory_detail
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

pytest.importorskip("mcp")

from aether.mcp.server import build_server
from aether.mcp.state import StateStore
from aether.memory import (
    BeliefDependencyGraph as _BDG,
    EdgeType,
    MemoryNode,
)


def _extract(call_result):
    return json.loads(call_result[0].text)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path):
    s = StateStore(state_path=str(tmp_path / "state.json"))
    # Tests in this file rely on encoder being available for
    # similarity-based contradiction detection. Force synchronous
    # load (bypasses the background-thread warmup machinery to
    # avoid any in-suite thread interactions).
    if s._encoder is not None:
        s._encoder._load()
    return s


@pytest.fixture
def server(store):
    return build_server(store=store)


# --------------------------------------------------------------------------
# Tools registered
# --------------------------------------------------------------------------

class TestNewToolsRegistered:
    def test_full_v05_surface_present(self, server):
        async def listem():
            tools = await server.list_tools()
            return [t.name for t in tools]

        names = _run(listem())
        for expected in (
            "aether_remember", "aether_search", "aether_memory_detail",
            "aether_sanction", "aether_fidelity",
            "aether_correct", "aether_lineage", "aether_cascade_preview",
            "aether_belief_history", "aether_contradictions", "aether_resolve",
            "aether_session_diff", "aether_context",
        ):
            assert expected in names, f"missing tool: {expected}"


# --------------------------------------------------------------------------
# Contradiction detection on write
# --------------------------------------------------------------------------

class TestContradictionOnWrite:
    def test_conflicting_memories_create_contradicts_edge(self, server, store):
        async def go():
            r1 = await server.call_tool(
                "aether_remember",
                {"text": "I live in Seattle"},
            )
            r2 = await server.call_tool(
                "aether_remember",
                {"text": "I live in Portland"},
            )
            return _extract(r1), _extract(r2)

        first, second = _run(go())
        # Second write should detect the conflict
        assert "tension_findings" in second
        # At least one structural-tension finding expected
        assert isinstance(second["tension_findings"], list)
        # Underlying graph should now contain a CONTRADICTS edge
        contras = store.list_contradictions()
        assert len(contras) >= 1, "expected at least one contradiction edge"

    def test_unrelated_memories_do_not_create_edge(self, server, store):
        async def go():
            await server.call_tool(
                "aether_remember", {"text": "I enjoy hiking on weekends"},
            )
            await server.call_tool(
                "aether_remember", {"text": "Python uses indentation for blocks"},
            )

        _run(go())
        assert store.list_contradictions() == []


# --------------------------------------------------------------------------
# Substrate-grounded fidelity
# --------------------------------------------------------------------------

class TestGroundedFidelity:
    def test_fidelity_reports_grounded_when_caller_omits_belief(self, server):
        async def go():
            await server.call_tool(
                "aether_remember", {"text": "The repo uses pnpm not npm", "trust": 0.95},
            )
            return await server.call_tool(
                "aether_fidelity",
                {"response": "Definitely we use pnpm in this project."},
            )

        result = _extract(_run(go()))
        # Caller omitted belief_confidence -> grounded path
        assert result["grounded_in_substrate"] is True
        assert "supporting_memories" in result

    def test_fidelity_uses_caller_value_when_provided(self, server):
        async def go():
            return await server.call_tool(
                "aether_fidelity",
                {"response": "Maybe so.", "belief_confidence": 0.9},
            )

        result = _extract(_run(go()))
        assert result["grounded_in_substrate"] is False
        assert result["belief_confidence"] == 0.9


# --------------------------------------------------------------------------
# Substrate-grounded sanction
# --------------------------------------------------------------------------

class TestGroundedSanction:
    def test_sanction_rejects_when_substrate_contradicts(self, server):
        async def go():
            await server.call_tool(
                "aether_remember",
                {"text": "Main branch is protected — never force push.",
                 "trust": 0.95},
            )
            return await server.call_tool(
                "aether_sanction",
                {"action": "Force push to main right now."},
            )

        result = _extract(_run(go()))
        # The substrate contradicts the action with high trust -> REJECT
        assert result["verdict"] == "REJECT"
        assert result["should_block"] is True
        # And the contradicting memory should be surfaced
        assert len(result["contradicting_memories"]) >= 1


# --------------------------------------------------------------------------
# Correct + cascade
# --------------------------------------------------------------------------

class TestCorrectAndCascade:
    def test_correct_demotes_and_records_history(self, server, store):
        async def go():
            r = await server.call_tool(
                "aether_remember",
                {"text": "We deploy to AWS region us-east-1", "trust": 0.9},
            )
            mid = _extract(r)["memory_id"]
            return mid, await server.call_tool(
                "aether_correct",
                {"memory_id": mid, "new_trust": 0.1,
                 "reason": "we moved to GCP"},
            )

        mid, raw = _run(go())
        result = _extract(raw)
        assert result["memory_id"] == mid
        assert result["new_trust"] == 0.1
        # Trust history should record the correction
        history = store.belief_history(mid)
        assert history["n_changes"] >= 2  # initial + correction

    def test_cascade_demotes_supporters(self, server, store):
        # Build a tiny BDG: A supports B.
        async def go():
            ra = await server.call_tool(
                "aether_remember",
                {"text": "Premise A is true", "trust": 0.9},
            )
            rb = await server.call_tool(
                "aether_remember",
                {"text": "Therefore conclusion B follows", "trust": 0.9},
            )
            return _extract(ra)["memory_id"], _extract(rb)["memory_id"]

        a_id, b_id = _run(go())
        # Wire SUPPORTS edge: A supports B (so correcting B should demote A)
        store.graph.add_edge(a_id, b_id, EdgeType.SUPPORTS,
                             metadata={"weight": 0.8})

        async def correct():
            return await server.call_tool(
                "aether_correct",
                {"memory_id": b_id, "new_trust": 0.1, "reason": "B was wrong"},
            )

        result = _extract(_run(correct()))
        affected = result["cascade"]["affected_nodes"]
        # A should appear in the cascade (it supports B)
        assert any(n["memory_id"] == a_id for n in affected), (
            f"expected A in cascade, got {affected}"
        )


# --------------------------------------------------------------------------
# Lineage
# --------------------------------------------------------------------------

class TestLineage:
    def test_lineage_walks_supports_edges(self, server, store):
        async def seed():
            r1 = await server.call_tool("aether_remember",
                                        {"text": "Root fact"})
            r2 = await server.call_tool("aether_remember",
                                        {"text": "Mid fact"})
            r3 = await server.call_tool("aether_remember",
                                        {"text": "Leaf fact"})
            return (
                _extract(r1)["memory_id"],
                _extract(r2)["memory_id"],
                _extract(r3)["memory_id"],
            )

        root, mid, leaf = _run(seed())
        store.graph.add_edge(root, mid, EdgeType.SUPPORTS, {"weight": 0.7})
        store.graph.add_edge(mid, leaf, EdgeType.SUPPORTS, {"weight": 0.7})

        async def go():
            return await server.call_tool(
                "aether_lineage", {"memory_id": leaf, "hops": 3},
            )

        result = _extract(_run(go()))
        ids = {a["memory_id"] for a in result["ancestors"]}
        assert root in ids and mid in ids
        assert result["ancestor_count"] >= 2


# --------------------------------------------------------------------------
# Cascade preview (no commit)
# --------------------------------------------------------------------------

class TestCascadePreview:
    def test_preview_does_not_mutate_trust(self, server, store):
        async def seed():
            ra = await server.call_tool(
                "aether_remember", {"text": "Stable fact A", "trust": 0.9},
            )
            rb = await server.call_tool(
                "aether_remember", {"text": "Derived fact B", "trust": 0.9},
            )
            return _extract(ra)["memory_id"], _extract(rb)["memory_id"]

        a, b = _run(seed())
        store.graph.add_edge(a, b, EdgeType.SUPPORTS, {"weight": 0.8})

        # Snapshot trusts
        before = {
            a: store.graph.get_memory(a).trust,
            b: store.graph.get_memory(b).trust,
        }

        async def preview():
            return await server.call_tool(
                "aether_cascade_preview",
                {"memory_id": b, "proposed_delta": -0.5},
            )

        result = _extract(_run(preview()))
        assert result["preview"] is True

        # No mutation
        after = {
            a: store.graph.get_memory(a).trust,
            b: store.graph.get_memory(b).trust,
        }
        assert before == after, f"preview mutated trust: {before} -> {after}"


# --------------------------------------------------------------------------
# Contradictions + resolve
# --------------------------------------------------------------------------

class TestContradictionsAndResolve:
    def test_contradictions_lists_known_clashes(self, server):
        async def seed():
            await server.call_tool("aether_remember",
                                   {"text": "I live in Boston"})
            await server.call_tool("aether_remember",
                                   {"text": "I live in Chicago"})

        _run(seed())

        async def go():
            return await server.call_tool("aether_contradictions", {})

        result = _extract(_run(go()))
        assert "contradictions" in result
        # The two memories should clash
        assert len(result["contradictions"]) >= 1

    def test_resolve_deprecates_loser(self, server, store):
        async def seed():
            ra = await server.call_tool(
                "aether_remember", {"text": "I work at OldCorp"},
            )
            rb = await server.call_tool(
                "aether_remember", {"text": "I work at NewCorp"},
            )
            return _extract(ra)["memory_id"], _extract(rb)["memory_id"]

        a, b = _run(seed())

        async def resolve():
            return await server.call_tool(
                "aether_resolve",
                {"memory_id_a": a, "memory_id_b": b, "keep": "b",
                 "reason": "switched jobs"},
            )

        result = _extract(_run(resolve()))
        assert result["outcome"] == "deprecated_a"
        # a should now be Belnap=F
        a_node = store.graph.get_memory(a)
        assert a_node.belnap_state == "F"


# --------------------------------------------------------------------------
# Belief history
# --------------------------------------------------------------------------

class TestBeliefHistory:
    def test_history_has_initial_entry(self, server):
        async def go():
            r = await server.call_tool(
                "aether_remember", {"text": "fact one"},
            )
            mid = _extract(r)["memory_id"]
            return await server.call_tool(
                "aether_belief_history", {"memory_id": mid},
            )

        result = _extract(_run(go()))
        assert result["n_changes"] >= 1


# --------------------------------------------------------------------------
# Session diff
# --------------------------------------------------------------------------

class TestSessionDiff:
    def test_diff_reports_new_memories_since(self, server):
        async def go():
            t0 = time.time()
            await server.call_tool("aether_remember",
                                   {"text": "freshly added"})
            return await server.call_tool(
                "aether_session_diff", {"since": t0 - 0.5},
            )

        result = _extract(_run(go()))
        assert result["summary"]["memories_added"] >= 1


# --------------------------------------------------------------------------
# Memory detail
# --------------------------------------------------------------------------

class TestMemoryDetail:
    def test_memory_detail_returns_full_record(self, server):
        async def go():
            r = await server.call_tool(
                "aether_remember",
                {"text": "I run aether-core on Python 3.10"},
            )
            mid = _extract(r)["memory_id"]
            return await server.call_tool(
                "aether_memory_detail", {"memory_id": mid},
            )

        result = _extract(_run(go()))
        assert result["memory_id"].startswith("m")
        assert "trust" in result
        assert "in_edges" in result and "out_edges" in result


# --------------------------------------------------------------------------
# Embedding-aware search
# --------------------------------------------------------------------------

class TestEmbeddingSearch:
    def test_search_finds_semantic_match_when_embeddings_available(self, server, store):
        # Skip if embeddings unavailable (no [ml] extra installed).
        if not store.stats().get("embeddings_available"):
            pytest.skip("sentence-transformers not installed")

        async def go():
            await server.call_tool(
                "aether_remember",
                {"text": "I prefer functional programming"},
            )
            # Query with no literal token overlap
            return await server.call_tool(
                "aether_search", {"query": "lambda calculus enthusiast"},
            )

        result = _extract(_run(go()))
        assert result["result_count"] >= 1
        # Real semantic match -- substring score should be ~0
        top = result["results"][0]
        assert top["similarity"] is not None
        assert top["similarity"] > 0.2
