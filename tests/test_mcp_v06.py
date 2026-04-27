"""Tests for v0.6.0 additions:
    - Mutual-exclusion contradiction detection
    - Belnap-state visibility on search results
    - Auto-ingest extractor + ingest_turn helper
    - aether_ingest_turn MCP tool
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from aether.mcp.server import build_server
from aether.mcp.state import StateStore
from aether.memory import extract_facts, ingest_turn, BelnapState
from aether.contradiction import detect_mutex_conflict, MutexConflict


def _extract(call_result):
    return json.loads(call_result[0].text)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path):
    return StateStore(state_path=str(tmp_path / "state.json"))


@pytest.fixture
def server(store):
    return build_server(store=store)


# --------------------------------------------------------------------------
# Mutex detection
# --------------------------------------------------------------------------

class TestMutexDirect:
    def test_aws_vs_gcp(self):
        h = detect_mutex_conflict(
            "We deploy to AWS us-east-1",
            "We deploy to GCP us-central1",
        )
        assert isinstance(h, MutexConflict)
        assert h.class_name == "cloud_provider"
        assert {h.value_a, h.value_b} == {"AWS", "GCP"}

    def test_postgres_vs_mysql(self):
        h = detect_mutex_conflict(
            "The team uses Postgres for the primary database",
            "Our database is MySQL",
        )
        assert h is not None
        assert h.class_name == "database"

    def test_react_vs_vue(self):
        h = detect_mutex_conflict(
            "The frontend is React",
            "We use Vue for the frontend",
        )
        assert h is not None
        assert h.class_name == "frontend_framework"

    def test_unrelated_contexts_do_not_fire(self):
        # No shared cue — should NOT match
        h = detect_mutex_conflict(
            "AWS docs are great",
            "GCP outage was bad",
        )
        assert h is None

    def test_same_value_does_not_fire(self):
        h = detect_mutex_conflict(
            "We deploy to AWS",
            "We deploy to AWS us-east-1",
        )
        assert h is None


class TestMutexOnWrite:
    def test_mutex_creates_contradiction_edge(self, server, store):
        async def go():
            await server.call_tool(
                "aether_remember",
                {"text": "We deploy to AWS us-east-1", "trust": 0.9},
            )
            return await server.call_tool(
                "aether_remember",
                {"text": "We deploy to GCP us-central1", "trust": 0.9},
            )

        result = _extract(_run(go()))
        findings = result["tension_findings"]
        assert any(f.get("kind") == "mutex" for f in findings), (
            f"expected mutex finding, got {findings}"
        )
        contras = store.list_contradictions()
        assert len(contras) >= 1


class TestMutexInGrounding:
    def test_grounding_classifies_mutex_as_contradict(self, store):
        store.add_memory("We deploy to AWS us-east-1", trust=0.95)
        g = store.compute_grounding("We deploy to GCP us-central1")
        assert any(c.get("kind") == "mutex" for c in g["contradict"]), (
            f"expected mutex contradict, got {g['contradict']}"
        )


# --------------------------------------------------------------------------
# Belnap visibility
# --------------------------------------------------------------------------

class TestBelnapWarnings:
    def test_held_memory_carries_warning(self, server, store):
        async def go():
            r1 = await server.call_tool(
                "aether_remember", {"text": "I live in Boston"},
            )
            r2 = await server.call_tool(
                "aether_remember", {"text": "I live in Chicago"},
            )
            mids = (_extract(r1)["memory_id"], _extract(r2)["memory_id"])
            return mids

        a, b = _run(go())
        # Force one into Both manually for the test (the meter chooses
        # held disposition for ambiguous tension; we want determinism).
        store.graph.update_belnap(a, BelnapState.BOTH)
        store.save()

        async def search():
            return await server.call_tool(
                "aether_search", {"query": "live in"},
            )

        result = _extract(_run(search()))
        # The Boston (a) memory should have a contested warning
        target = next(r for r in result["results"] if r["memory_id"] == a)
        assert target["belnap_state"] == "Both"
        assert any("contested" in w for w in target["warnings"])


# --------------------------------------------------------------------------
# Auto-ingest extractor (pure function)
# --------------------------------------------------------------------------

class TestExtractFacts:
    def test_user_preference(self):
        facts = extract_facts(user_message="I prefer Python over Rust")
        assert any(f.signal == "user_preference" for f in facts)

    def test_user_identity(self):
        facts = extract_facts(user_message="I work at Stripe and I live in Boston")
        signals = {f.signal for f in facts}
        assert "user_identity" in signals

    def test_project_fact(self):
        facts = extract_facts(
            user_message="this repo uses pnpm and we deploy to Vercel",
        )
        signals = {f.signal for f in facts}
        assert "project_fact" in signals

    def test_constraint(self):
        facts = extract_facts(user_message="never push directly to main")
        assert any(f.signal == "constraint" for f in facts)

    def test_correction(self):
        facts = extract_facts(
            user_message="actually we removed those FK constraints last sprint",
        )
        assert any(f.signal == "correction" for f in facts)

    def test_decision(self):
        facts = extract_facts(
            user_message="we decided to ship the migration this Friday",
        )
        assert any(f.signal == "decision" for f in facts)

    def test_question_does_not_extract(self):
        facts = extract_facts(user_message="do we use pnpm?")
        # The "we use pnpm" might be matched but the question mark causes
        # garbage filter to reject. Fine if facts is empty.
        for f in facts:
            assert "?" not in f.text

    def test_max_facts_caps_output(self):
        msg = (
            "I prefer Python. I love Rust. We use AWS. "
            "We decided to ship. Never push to main. Actually let's revisit. "
            "I am a developer. I live in Boston."
        )
        facts = extract_facts(user_message=msg, max_facts=3)
        assert len(facts) <= 3


class TestIngestTurn:
    def test_ingest_writes_to_store(self, store):
        writes = ingest_turn(
            store,
            user_message="I prefer pnpm and we deploy to Fly.io",
        )
        assert len(writes) >= 1
        # Each write should have a memory_id
        for w in writes:
            assert w["memory_id"].startswith("m")

    def test_ingest_dedupes_against_substrate(self, store):
        # Pre-populate with a fact
        store.add_memory("User preference: pnpm", trust=0.9)
        # Now an ingest of the same content should be skipped
        writes = ingest_turn(
            store,
            user_message="I prefer pnpm",
        )
        # The dedup logic should drop the near-duplicate
        # (allowed if not dropped — depends on similarity threshold —
        # but we should get fewer writes than candidates)
        assert all(
            "pnpm" not in w["text"].lower()
            or w["trust"] != 0.85  # not the freshly-extracted one
            for w in writes
        ) or len(writes) == 0


# --------------------------------------------------------------------------
# aether_ingest_turn MCP tool
# --------------------------------------------------------------------------

class TestIngestTurnTool:
    def test_tool_extracts_and_writes(self, server):
        async def go():
            return await server.call_tool(
                "aether_ingest_turn",
                {
                    "user_message": (
                        "I prefer pnpm and we deploy to Vercel. "
                        "Never push directly to main."
                    ),
                },
            )

        result = _extract(_run(go()))
        assert result["ingested_count"] >= 2
        # Each write has a signal
        for w in result["writes"]:
            assert "signal" in w

    def test_tool_handles_empty_input(self, server):
        async def go():
            return await server.call_tool(
                "aether_ingest_turn", {},
            )

        result = _extract(_run(go()))
        assert result["ingested_count"] == 0
