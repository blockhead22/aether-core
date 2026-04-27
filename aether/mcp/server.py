"""FastMCP server exposing the Aether belief substrate.

Tools registered (v0.6.0):
    Memory:
        aether_remember        write a fact (auto contradiction-detection)
        aether_search          embedding + substring hybrid search
        aether_memory_detail   single-memory deep view
        aether_ingest_turn     pull high-signal facts from a conversation turn
    Governance:
        aether_sanction        substrate-grounded action gate
        aether_fidelity        substrate-grounded draft auditor
    Substrate ops:
        aether_correct         trust update + cascade through BDG
        aether_lineage         walk SUPPORTS edges back to source
        aether_cascade_preview dry-run a correction
        aether_belief_history  trust-evolution log per memory
        aether_contradictions  list contradictions, optional disposition filter
        aether_resolve         resolve a contradiction (deprecate / hold / drop)
        aether_session_diff    what changed since a given timestamp
    Introspection:
        aether_context         dashboard snapshot

State is held in process via StateStore. Persistence is JSON on disk
(default: ~/.aether/mcp_state.json) plus a side-car trust_history file.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from aether.governance import GovernanceTier
from aether.governance.gap_auditor import ResponseAudit
from aether.memory import extract_fact_slots, ingest_turn

from .state import StateStore, SENTINEL_BELIEF_CONF


def build_server(store: Optional[StateStore] = None) -> FastMCP:
    """Construct a fresh FastMCP server bound to the given StateStore."""
    store = store or StateStore()
    mcp = FastMCP("aether")

    # ==================================================================
    # Memory tools
    # ==================================================================

    @mcp.tool()
    def aether_remember(
        text: str,
        trust: float = 0.7,
        source: str = "user",
        detect_contradictions: bool = True,
    ) -> dict:
        """Store a new fact in the belief state.

        Trust defaults to 0.7 (user-asserted). Use lower values for
        inferred or low-confidence facts.

        Slot extraction runs automatically. Structural tension detection
        runs against the top-K most similar existing memories — if a
        clash is found, a CONTRADICTS edge is added and surfaced in
        `tension_findings`.

        Returns the assigned memory_id, extracted slots, and any
        contradictions detected on write.
        """
        extracted = extract_fact_slots(text)
        slots = {k: v.normalized for k, v in extracted.items()} or None
        return store.add_memory(
            text=text,
            trust=trust,
            source=source,
            slots=slots,
            detect_contradictions=detect_contradictions,
        )

    @mcp.tool()
    def aether_search(query: str, limit: int = 5) -> dict:
        """Find memories matching the query.

        Hybrid search: cosine over local embeddings (sentence-transformers
        all-MiniLM-L6-v2) when available, falling back to substring +
        token overlap when not. Each result includes both the combined
        score and the raw similarity.
        """
        results = store.search(query, limit=limit)
        return {
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    @mcp.tool()
    def aether_memory_detail(memory_id: str) -> dict:
        """Deep view of a single memory: edges, contradictions, history length."""
        return store.memory_detail(memory_id)

    @mcp.tool()
    def aether_ingest_turn(
        user_message: str = "",
        assistant_response: str = "",
        max_facts: int = 8,
    ) -> dict:
        """Pull high-signal facts out of a conversation turn and write them.

        Conservative regex-based extractor. Fires on explicit preferences,
        identity statements, project-config declarations, decisions,
        constraints, and corrections. Each candidate is deduped against
        the substrate before writing. Returns the writes performed.

        Designed to live inside a Claude Code Stop hook, but works as a
        manual tool too. Pass `user_message` and/or `assistant_response`.
        """
        u = user_message or None
        a = assistant_response or None
        writes = ingest_turn(
            store,
            user_message=u,
            assistant_response=a,
            max_facts=max_facts,
        )
        return {
            "ingested_count": len(writes),
            "writes": writes,
        }

    # ==================================================================
    # Governance tools
    # ==================================================================

    @mcp.tool()
    def aether_sanction(
        action: str,
        belief_confidence: float = 0.5,
    ) -> dict:
        """Pre-action governance gate.

        Returns APPROVE / HOLD / REJECT based on:
          - Substrate grounding for the action (auto-computed when
            `belief_confidence` is the sentinel default 0.5)
          - The 6 immune agents in the governance layer
          - Any contradictions found between the action and stored beliefs

        When the substrate contradicts the proposed action with high
        trust, the verdict is REJECT and the contradicting memories
        appear in the `contradicting_memories` field.
        """
        # If caller passed sentinel, ground in substrate
        if abs(belief_confidence - SENTINEL_BELIEF_CONF) < 1e-6:
            grounding = store.compute_grounding(action)
            effective_belief = grounding["belief_confidence"]
            grounded = True
        else:
            grounding = {"support": [], "contradict": [], "method": "caller_supplied"}
            effective_belief = belief_confidence
            grounded = False

        result = store.gov.govern_response(
            action, belief_confidence=effective_belief,
        )
        tier = result.tier
        if result.should_block:
            verdict = "REJECT"
        elif tier in (GovernanceTier.HEDGE, GovernanceTier.FLAG):
            verdict = "HOLD"
        else:
            verdict = "APPROVE"

        # Substrate override: high-trust contradictions force REJECT
        contradicting = grounding.get("contradict", [])
        if any(c["trust"] >= 0.7 for c in contradicting):
            verdict = "REJECT"
            tier = GovernanceTier.ESCALATE

        return {
            "verdict": verdict,
            "tier": tier.value,
            "should_block": verdict == "REJECT",
            "confidence_adjustment": result.confidence_adjustment,
            "belief_confidence": effective_belief,
            "grounded_in_substrate": grounded,
            "supporting_memories": grounding.get("support", [])[:3],
            "contradicting_memories": contradicting[:3],
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
        """Grade a draft response against substrate-backed belief support.

        When `belief_confidence` is the sentinel default 0.5, the function
        searches the substrate to compute a real grounding score from
        supporting / contradicting memories. The caller can override by
        passing a specific value.

        Returns gap_score, severity, action, plus the grounding evidence
        that was used.
        """
        if abs(belief_confidence - SENTINEL_BELIEF_CONF) < 1e-6:
            grounding = store.compute_grounding(response)
            effective_belief = grounding["belief_confidence"]
            grounded = True
        else:
            grounding = {"support": [], "contradict": [], "method": "caller_supplied"}
            effective_belief = belief_confidence
            grounded = False

        audit = ResponseAudit(
            response_text=response,
            belief_confidence=effective_belief,
        )
        verdict = store.gov.gap_auditor.audit(audit)
        return {
            "gap_score": verdict.gap_score,
            "severity": verdict.severity.value,
            "action": verdict.action.value,
            "speech_confidence": verdict.speech_confidence,
            "belief_confidence": verdict.belief_confidence,
            "grounded_in_substrate": grounded,
            "grounding_method": grounding.get("method", "caller_supplied"),
            "supporting_memories": grounding.get("support", [])[:3],
            "contradicting_memories": grounding.get("contradict", [])[:3],
            "factors": verdict.contributing_factors,
            "law": verdict.law,
        }

    # ==================================================================
    # Substrate operations
    # ==================================================================

    @mcp.tool()
    def aether_correct(
        memory_id: str,
        new_trust: float = -1.0,
        replacement_text: str = "",
        reason: str = "",
        source: str = "user",
    ) -> dict:
        """Correct a memory and cascade the trust drop to dependents.

        new_trust:
            -1.0 (default sentinel) - halve current trust as soft demotion
            0.0  - full deprecation
            n    - set to specific value

        Returns the cascade result with affected nodes and depths.
        """
        nt: Optional[float] = None if new_trust < 0 else new_trust
        rt: Optional[str] = replacement_text if replacement_text else None
        return store.correct(
            memory_id=memory_id,
            new_trust=nt,
            replacement_text=rt,
            reason=reason,
            source=source,
        )

    @mcp.tool()
    def aether_lineage(memory_id: str, hops: int = 3) -> dict:
        """Walk SUPPORTS / DERIVED_FROM edges back to source memories.

        Answers: "Why do I believe this?" Returns the chain of memories
        whose trust transitively supports the target.
        """
        return store.lineage(memory_id=memory_id, hops=hops)

    @mcp.tool()
    def aether_cascade_preview(
        memory_id: str,
        proposed_delta: float = -0.4,
    ) -> dict:
        """Dry-run a correction. See the blast radius before committing.

        Same engine as aether_correct, but no state mutation.
        """
        return store.cascade_preview(
            memory_id=memory_id,
            proposed_delta=proposed_delta,
        )

    @mcp.tool()
    def aether_belief_history(memory_id: str) -> dict:
        """How a memory's trust has evolved over time."""
        return store.belief_history(memory_id)

    @mcp.tool()
    def aether_contradictions(
        disposition: str = "",
        limit: int = 50,
    ) -> dict:
        """List contradictions in the substrate.

        disposition: optional filter — "resolvable", "held", "evolving",
                     "contextual", or "" for all.
        """
        disp: Optional[str] = disposition if disposition else None
        return {
            "contradictions": store.list_contradictions(
                disposition=disp, limit=limit,
            ),
        }

    @mcp.tool()
    def aether_resolve(
        memory_id_a: str,
        memory_id_b: str,
        keep: str,
        reason: str = "",
    ) -> dict:
        """Resolve a contradiction.

        keep:
            "a"     deprecate b, supersede with a
            "b"     deprecate a, supersede with b
            "both"  mark both Belnap=BOTH (held)
            "drop"  mark both Belnap=FALSE
        """
        return store.resolve_contradiction(
            memory_id_a=memory_id_a,
            memory_id_b=memory_id_b,
            keep=keep,
            reason=reason,
        )

    @mcp.tool()
    def aether_session_diff(since: float) -> dict:
        """What changed in the substrate since the given timestamp.

        Useful at session start to brief a returning agent on memories
        added, corrections accepted, and contradictions surfaced since
        last connect.
        """
        return store.session_diff(since=since)

    # ==================================================================
    # Introspection
    # ==================================================================

    @mcp.tool()
    def aether_context() -> dict:
        """Dashboard snapshot of the current substrate state."""
        return store.stats()

    return mcp


def run() -> None:
    """Run the Aether MCP server over stdio."""
    store = StateStore()
    server = build_server(store=store)
    server.run("stdio")
