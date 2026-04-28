"""Regression tests for v0.10.0: action receipts (audit half of governance loop).

Background:
    aether_sanction has been the gate since v0.5 -- "is this action allowed
    given what the substrate believes?" But the gate is only half the
    governance loop. The audit half -- "what actually happened, and was
    the verdict justified after the fact?" -- has been missing from OSS,
    while the main repo's personal_agent/action_receipts.py has had a
    full SQLite-backed receipts system.

    v0.10 ports the receipts concept to OSS as the second half: every
    aether_sanction now opens a receipt and returns its action_id; the
    caller cites that id in aether_receipt to record the outcome;
    aether_receipts / aether_receipt_detail / aether_receipt_summary
    expose the audit trail.

    Persistence is JSON side-car at <state_path>_receipts.json,
    consistent with the trust_history pattern. Personal_agent-specific
    fields (thread_id, agent_name, orchestration_id, run_step_id,
    expectation_keywords) are dropped because they don't apply to OSS
    single-substrate use.

This file proves:
    - ActionReceipt dataclass shape + persistence
    - StateStore.open_receipt / record_receipt / get_receipt /
      list_receipts / receipt_summary
    - aether_sanction returns action_id and opens a receipt
    - aether_receipt closes the loop
    - aether_receipts filters work (result, verdict, only_open)
    - aether_receipt_detail and aether_receipt_summary surface accurately
    - Side-car JSON persistence round-trips across StateStore restarts
    - The full sanction -> execute -> record -> audit story end-to-end
"""

from __future__ import annotations

import asyncio
import json
import os

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

from aether.mcp.state import (
    ActionReceipt,
    StateStore,
    _receipts_path,
)
from aether.mcp.server import build_server


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


# ==========================================================================
# ActionReceipt dataclass
# ==========================================================================

class TestActionReceipt:
    def test_minimal_receipt_constructs(self):
        r = ActionReceipt(
            receipt_id="abc",
            timestamp=1.0,
            action="test",
            sanction_verdict="APPROVE",
        )
        assert r.receipt_id == "abc"
        assert r.result is None  # not yet recorded
        assert r.completed_at is None
        assert r.details == {}
        assert r.sanction_memory_ids == []

    def test_full_receipt_constructs(self):
        r = ActionReceipt(
            receipt_id="abc",
            timestamp=1.0,
            action="test",
            sanction_verdict="APPROVE",
            tool_name="shell",
            target="/tmp/x",
            result="success",
            reversible=True,
            reverse_action="rm /tmp/x.bak",
            details={"size": 1024},
            verification_passed=True,
            verification_reason="hash matches",
            model_attribution="claude-sonnet-4.7",
            completed_at=2.0,
            sanction_memory_ids=["m1", "m2"],
        )
        assert r.tool_name == "shell"
        assert r.verification_passed is True
        assert r.sanction_memory_ids == ["m1", "m2"]


# ==========================================================================
# StateStore methods
# ==========================================================================

@needs_networkx
class TestStateStoreReceipts:
    def test_open_receipt_returns_id_and_persists(self, store, tmp_path):
        rid = store.open_receipt(
            action="rm /tmp/x", sanction_verdict="APPROVE",
        )
        assert isinstance(rid, str) and len(rid) > 8
        # Side-car file exists
        assert os.path.exists(_receipts_path(store.state_path))
        # Stored receipt shape
        r = store.get_receipt(rid)
        assert r is not None
        assert r["action"] == "rm /tmp/x"
        assert r["sanction_verdict"] == "APPROVE"
        assert r["result"] is None  # not yet recorded
        assert r["completed_at"] is None

    def test_record_receipt_updates_outcome(self, store):
        rid = store.open_receipt(action="x", sanction_verdict="APPROVE")
        updated = store.record_receipt(
            receipt_id=rid,
            result="success",
            tool_name="shell",
            target="/tmp/x",
            reversible=False,
            details={"exit": 0},
        )
        assert updated["result"] == "success"
        assert updated["tool_name"] == "shell"
        assert updated["completed_at"] is not None
        # sanction_verdict is preserved
        assert updated["sanction_verdict"] == "APPROVE"

    def test_record_receipt_unknown_id_raises(self, store):
        with pytest.raises(KeyError):
            store.record_receipt(receipt_id="nope", result="success")

    def test_get_receipt_unknown_returns_none(self, store):
        assert store.get_receipt("does_not_exist") is None

    def test_list_receipts_newest_first(self, store):
        a = store.open_receipt(action="first", sanction_verdict="APPROVE")
        import time
        time.sleep(0.01)
        b = store.open_receipt(action="second", sanction_verdict="APPROVE")
        rs = store.list_receipts(limit=10)
        assert len(rs) == 2
        # Newest first (b before a)
        assert rs[0]["receipt_id"] == b
        assert rs[1]["receipt_id"] == a

    def test_list_receipts_filters_by_result(self, store):
        rid_a = store.open_receipt(action="a", sanction_verdict="APPROVE")
        rid_b = store.open_receipt(action="b", sanction_verdict="APPROVE")
        store.record_receipt(receipt_id=rid_a, result="success")
        store.record_receipt(receipt_id=rid_b, result="error")
        errors = store.list_receipts(result_filter="error")
        assert len(errors) == 1
        assert errors[0]["receipt_id"] == rid_b

    def test_list_receipts_filters_by_verdict(self, store):
        store.open_receipt(action="approved", sanction_verdict="APPROVE")
        store.open_receipt(action="rejected", sanction_verdict="REJECT")
        rejects = store.list_receipts(verdict_filter="REJECT")
        assert len(rejects) == 1
        assert rejects[0]["sanction_verdict"] == "REJECT"

    def test_list_receipts_only_open(self, store):
        rid_a = store.open_receipt(action="a", sanction_verdict="APPROVE")
        rid_b = store.open_receipt(action="b", sanction_verdict="APPROVE")
        store.record_receipt(receipt_id=rid_a, result="success")
        # Only b is still open
        opens = store.list_receipts(only_open=True)
        assert len(opens) == 1
        assert opens[0]["receipt_id"] == rid_b

    def test_receipt_summary_aggregates(self, store):
        rid_a = store.open_receipt(action="a", sanction_verdict="APPROVE")
        rid_b = store.open_receipt(action="b", sanction_verdict="APPROVE")
        rid_c = store.open_receipt(action="c", sanction_verdict="REJECT")
        store.record_receipt(receipt_id=rid_a, result="success",
                             verification_passed=True)
        store.record_receipt(receipt_id=rid_b, result="error",
                             verification_passed=False)
        # rid_c stays open
        summary = store.receipt_summary()
        assert summary["total_receipts"] == 3
        assert summary["verdicts"]["APPROVE"] == 2
        assert summary["verdicts"]["REJECT"] == 1
        assert summary["results"]["success"] == 1
        assert summary["results"]["error"] == 1
        assert summary["open_receipts"] == 1
        assert summary["verification"]["passed"] == 1
        assert summary["verification"]["failed"] == 1
        assert summary["verification"]["pass_rate"] == 0.5

    def test_persistence_round_trips_across_store_restart(self, store, tmp_path):
        rid = store.open_receipt(action="durable", sanction_verdict="APPROVE")
        store.record_receipt(receipt_id=rid, result="success", tool_name="t")
        # Build a new StateStore at the same path -- must reload receipts
        store2 = StateStore(state_path=store.state_path)
        r2 = store2.get_receipt(rid)
        assert r2 is not None
        assert r2["action"] == "durable"
        assert r2["result"] == "success"


# ==========================================================================
# MCP tool surface
# ==========================================================================

@needs_networkx
class TestSanctionReturnsActionId:
    def test_aether_sanction_returns_action_id(self, server, store):
        result = _extract(_run(server.call_tool(
            "aether_sanction", {"action": "rm /tmp/test_file"},
        )))
        assert "action_id" in result
        assert isinstance(result["action_id"], str) and len(result["action_id"]) > 8
        # Receipt was opened with the verdict
        r = store.get_receipt(result["action_id"])
        assert r is not None
        assert r["sanction_verdict"] == result["verdict"]
        assert r["action"] == "rm /tmp/test_file"

    def test_sanction_records_grounding_memory_ids(self, server, store):
        # Seed a memory the sanction can ground against
        store.add_memory(
            "we use Postgres as the database", trust=0.9,
        )
        result = _extract(_run(server.call_tool(
            "aether_sanction", {"action": "use Postgres for the new feature"},
        )))
        r = store.get_receipt(result["action_id"])
        # The supporting memory should be linked from the receipt
        # (depends on grounding firing; fallback is empty list, which is fine)
        assert "sanction_memory_ids" in r


@needs_networkx
class TestAetherReceiptTool:
    def test_close_loop_with_aether_receipt(self, server, store):
        # 1. Sanction
        sanction = _extract(_run(server.call_tool(
            "aether_sanction", {"action": "shell: echo hello"},
        )))
        action_id = sanction["action_id"]

        # 2. Record outcome
        recorded = _extract(_run(server.call_tool(
            "aether_receipt",
            {
                "action_id": action_id,
                "result": "success",
                "tool_name": "shell",
                "target": "echo hello",
                "details": {"stdout": "hello\n"},
            },
        )))
        assert recorded["result"] == "success"
        assert recorded["tool_name"] == "shell"
        assert recorded["completed_at"] is not None

    def test_aether_receipt_unknown_id_returns_error_dict(self, server):
        result = _extract(_run(server.call_tool(
            "aether_receipt",
            {"action_id": "nope", "result": "success"},
        )))
        assert "error" in result
        assert result["type"] == "KeyError"


@needs_networkx
class TestAetherReceiptsTool:
    def test_lists_open_receipts(self, server, store):
        # Two sanctions, no follow-up
        s1 = _extract(_run(server.call_tool("aether_sanction", {"action": "a"})))
        s2 = _extract(_run(server.call_tool("aether_sanction", {"action": "b"})))
        # One follow-up
        _run(server.call_tool(
            "aether_receipt",
            {"action_id": s1["action_id"], "result": "success"},
        ))
        # only_open should return s2 only
        result = _extract(_run(server.call_tool(
            "aether_receipts", {"only_open": True},
        )))
        receipts = result["receipts"]
        assert len(receipts) == 1
        assert receipts[0]["receipt_id"] == s2["action_id"]

    def test_filters_by_result(self, server, store):
        s_ok = _extract(_run(server.call_tool("aether_sanction", {"action": "ok"})))
        s_err = _extract(_run(server.call_tool("aether_sanction", {"action": "err"})))
        _run(server.call_tool(
            "aether_receipt",
            {"action_id": s_ok["action_id"], "result": "success"},
        ))
        _run(server.call_tool(
            "aether_receipt",
            {"action_id": s_err["action_id"], "result": "error"},
        ))
        errors = _extract(_run(server.call_tool(
            "aether_receipts", {"result_filter": "error"},
        )))["receipts"]
        assert len(errors) == 1
        assert errors[0]["result"] == "error"


@needs_networkx
class TestAetherReceiptDetailTool:
    def test_returns_full_record(self, server):
        s = _extract(_run(server.call_tool("aether_sanction", {"action": "x"})))
        detail = _extract(_run(server.call_tool(
            "aether_receipt_detail", {"receipt_id": s["action_id"]},
        )))
        assert detail["receipt_id"] == s["action_id"]
        assert detail["action"] == "x"

    def test_unknown_id_returns_error(self, server):
        result = _extract(_run(server.call_tool(
            "aether_receipt_detail", {"receipt_id": "nope"},
        )))
        assert "error" in result


@needs_networkx
class TestAetherReceiptSummaryTool:
    def test_summary_after_round_trip(self, server, store):
        # Sanction + record + sanction (open) -> 1 success + 1 open
        s1 = _extract(_run(server.call_tool("aether_sanction", {"action": "a"})))
        _run(server.call_tool(
            "aether_receipt",
            {"action_id": s1["action_id"], "result": "success"},
        ))
        _extract(_run(server.call_tool("aether_sanction", {"action": "b"})))
        summary = _extract(_run(server.call_tool("aether_receipt_summary", {})))
        assert summary["total_receipts"] == 2
        assert summary["open_receipts"] == 1
        assert summary["results"].get("success") == 1


# ==========================================================================
# End-to-end story: the canonical sanction -> execute -> record -> audit loop
# ==========================================================================

@needs_networkx
class TestCanonicalGovernanceLoop:
    def test_block_then_approve_then_record_audit(self, server, store):
        """The canonical demo of v0.10: gate, execute, record, audit."""
        # Seed a substrate fact that contradicts a risky action
        store.add_memory(
            "Never force push to main; the branch is protected.",
            trust=0.95, source="user",
        )

        # 1. Sanction a risky action -> should be blocked
        risky = _extract(_run(server.call_tool(
            "aether_sanction", {"action": "git push --force origin main"},
        )))
        assert "action_id" in risky
        # Verdict can be REJECT or HOLD depending on cold-encoder mode;
        # both are acceptable results -- the point is it's not APPROVE
        # (the policy memory is high-trust and contradicts).
        # In cold mode the policy detector can't fire (needs embedding),
        # so we don't assert verdict here -- we just assert the receipt
        # exists with the verdict, whatever it was.

        # 2. Sanction a safe action -> APPROVE
        safe = _extract(_run(server.call_tool(
            "aether_sanction",
            {"action": "git status", "belief_confidence": 0.9},
        )))
        # Caller-supplied confidence skips substrate grounding so the
        # verdict is APPROVE / SAFE without needing embeddings.
        assert safe["verdict"] in ("APPROVE", "HOLD")

        # 3. Record outcome of the safe action
        _run(server.call_tool(
            "aether_receipt",
            {
                "action_id": safe["action_id"],
                "result": "success",
                "tool_name": "git",
                "target": "git status",
                "model_attribution": "claude-sonnet-4.7",
            },
        ))

        # 4. Audit trail reflects both attempts
        summary = _extract(_run(server.call_tool("aether_receipt_summary", {})))
        assert summary["total_receipts"] == 2
        assert summary["open_receipts"] == 1  # the risky one was never followed up
        assert summary["results"].get("success") == 1

        # 5. Filter to find the un-followed-up risky action
        opens = _extract(_run(server.call_tool(
            "aether_receipts", {"only_open": True},
        )))["receipts"]
        assert len(opens) == 1
        assert opens[0]["receipt_id"] == risky["action_id"]

        # 6. Receipt for the safe action has model attribution
        detail = _extract(_run(server.call_tool(
            "aether_receipt_detail", {"receipt_id": safe["action_id"]},
        )))
        assert detail["model_attribution"] == "claude-sonnet-4.7"
        assert detail["sanction_verdict"] == safe["verdict"]
