"""FastMCP server exposing the Aether belief substrate.

Tools registered (v0.4.0):
    aether_remember   write a fact into the belief state
    aether_search     find memories by text overlap
    aether_sanction   pre-action governance gate (APPROVE/HOLD/REJECT)
    aether_fidelity   grade a draft response against belief support
    aether_context    dashboard snapshot of current substrate state

Future (planned):
    aether_correct, aether_lineage, aether_cascade_preview,
    aether_done_check, aether_session_diff, aether_resolve

State is held in process via StateStore. Persistence is a single
JSON file on disk (default: ~/.aether/mcp_state.json).
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from aether.governance import GovernanceTier
from aether.governance.gap_auditor import ResponseAudit
from aether.memory import extract_fact_slots

from .state import StateStore


def build_server(store: Optional[StateStore] = None) -> FastMCP:
    """Construct a fresh FastMCP server bound to the given StateStore.

    Most callers want `run()` instead, which builds the default store
    (loading from disk) and runs the stdio transport.
    """
    store = store or StateStore()
    mcp = FastMCP("aether")

    # ------------------------------------------------------------------
    # Memory tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def aether_remember(
        text: str,
        trust: float = 0.7,
        source: str = "user",
    ) -> dict:
        """Store a new fact in the belief state.

        Trust defaults to 0.7 (user-asserted). Use lower values for
        inferred or low-confidence facts.

        Slot extraction runs automatically; structured slots (location,
        employer, age, etc.) are attached to the memory.

        Returns the assigned memory_id and any extracted slots.
        """
        extracted = extract_fact_slots(text)
        slots = {k: v.normalized for k, v in extracted.items()} or None
        memory_id = store.add_memory(
            text=text, trust=trust, source=source, slots=slots,
        )
        return {
            "memory_id": memory_id,
            "trust": trust,
            "source": source,
            "extracted_slots": slots or {},
        }

    @mcp.tool()
    def aether_search(query: str, limit: int = 5) -> dict:
        """Find memories matching the query.

        Pure structural search: substring + token overlap. Sufficient
        for small belief states (<1000 memories). Returns the top
        `limit` matches ranked by score.
        """
        results = store.search(query, limit=limit)
        return {
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    # ------------------------------------------------------------------
    # Governance tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def aether_sanction(
        action: str,
        belief_confidence: float = 0.5,
    ) -> dict:
        """Pre-action governance gate.

        Runs the proposed action through the four-tier dispatcher and
        the six immune agents. Returns one of:

            APPROVE  the action is well-supported and consistent
            HOLD     the action is borderline; reduce displayed confidence
            REJECT   the action contradicts belief or fails a law
                     (`should_block` is True)

        belief_confidence is the agent's internal estimate of how well
        the belief state supports the action (0.0 = no support,
        1.0 = strong support). Agents that don't track this should
        pass 0.5 and let the governance layer compute its own.

        Always call this before irreversible work.
        """
        result = store.gov.govern_response(
            action, belief_confidence=belief_confidence,
        )
        # Map four internal tiers to a three-state external API.
        tier = result.tier
        if result.should_block:
            verdict = "REJECT"
        elif tier in (GovernanceTier.HEDGE, GovernanceTier.FLAG):
            verdict = "HOLD"
        else:
            verdict = "APPROVE"

        return {
            "verdict": verdict,
            "tier": tier.value,
            "should_block": result.should_block,
            "confidence_adjustment": result.confidence_adjustment,
            "annotations": [
                {
                    "agent": a.agent,
                    "law": a.law,
                    "severity": a.severity,
                    "finding": a.finding,
                }
                for a in result.annotations
            ],
        }

    @mcp.tool()
    def aether_fidelity(
        response: str,
        belief_confidence: float = 0.5,
    ) -> dict:
        """Grade a draft response against belief support.

        Specifically wraps the `GapAuditor` (Law 5: outward confidence
        must be bounded by internal support). Returns the measured
        belief/speech gap, severity, and the recommended adjustment.

        Use this on drafts before sending. If the gap is high, hedge
        the response or block it.
        """
        audit = ResponseAudit(
            response_text=response,
            belief_confidence=belief_confidence,
        )
        verdict = store.gov.gap_auditor.audit(audit)
        return {
            "gap_score": verdict.gap_score,
            "severity": verdict.severity.value,
            "action": verdict.action.value,
            "speech_confidence": verdict.speech_confidence,
            "belief_confidence": verdict.belief_confidence,
            "factors": verdict.contributing_factors,
            "law": verdict.law,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @mcp.tool()
    def aether_context() -> dict:
        """Dashboard snapshot of the current substrate state.

        Returns memory count, edge count, Belnap state distribution,
        and the path to the on-disk state file.
        """
        return store.stats()

    return mcp


def run() -> None:
    """Run the Aether MCP server over stdio.

    This is the entrypoint for `python -m aether.mcp`. State is loaded
    from disk, the server runs until the client disconnects, and state
    is saved on every mutation along the way.
    """
    store = StateStore()
    server = build_server(store=store)
    server.run("stdio")
