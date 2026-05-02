"""FastMCP server exposing the Aether belief substrate.

Tools registered (v0.9.1):
    Memory:
        aether_remember        write a fact (auto contradiction-detection
                               + auto RELATED_TO link above similarity threshold)
        aether_search          embedding + substring hybrid search
        aether_path            Dijkstra shortest-path retrieval over BDG
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
        aether_link            explicit SUPPORTS / DERIVED_FROM / RELATED_TO edge (v0.9.1)
        aether_session_diff    what changed since a given timestamp
    Introspection:
        aether_context         dashboard snapshot

State is held in process via StateStore. Persistence is JSON on disk
(default: ~/.aether/mcp_state.json) plus a side-car trust_history file.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from pathlib import Path

from aether.governance import GovernanceTier
from aether.governance.gap_auditor import ResponseAudit
from aether.memory import extract_fact_slots, ingest_turn
from aether.substrate import SubstrateGraph

from .state import StateStore, SENTINEL_BELIEF_CONF


_SUBSTRATE_SINGLETON: Optional[SubstrateGraph] = None


def _get_substrate() -> SubstrateGraph:
    """Lazy-load the substrate from ~/.aether/substrate.json on first use."""
    global _SUBSTRATE_SINGLETON
    if _SUBSTRATE_SINGLETON is None:
        sub = SubstrateGraph()
        path = str(Path.home() / ".aether" / "substrate.json")
        sub.load(path)
        _SUBSTRATE_SINGLETON = sub
    return _SUBSTRATE_SINGLETON


def _state_to_dict(s) -> dict:
    """Slim SlotState dict (omit verbose internals for MCP transport)."""
    return {
        "state_id": s.state_id,
        "slot_id": s.slot_id,
        "value": s.value,
        "normalized": s.normalized,
        "trust": s.trust,
        "observed_at": s.observed_at,
        "observation_id": s.observation_id,
        "temporal_status": s.temporal_status,
        "source": s.source,
        "superseded_by": s.superseded_by,
    }


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
            grounding = {
                "support": [], "contradict": [],
                "methodological_concerns": [],
                "method": "caller_supplied",
            }
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

        # v0.9.3 (Layer 2): high-trust methodological concerns force HOLD.
        # We don't REJECT on these — methodological overclaims are about the
        # form of a claim, and the right response is to push back / hedge,
        # not to block the action outright. HOLD signals "review this
        # methodology before proceeding."
        methodological = grounding.get("methodological_concerns", [])
        if any(m["trust"] >= 0.7 for m in methodological) and verdict == "APPROVE":
            verdict = "HOLD"
            if tier == GovernanceTier.SAFE:
                tier = GovernanceTier.FLAG

        # v0.10: open an ActionReceipt as the sanction fires. Caller cites
        # the returned action_id when calling aether_receipt to record what
        # actually happened. The receipt closes the governance loop:
        # sanction is the gate, receipts are the audit trail.
        sanction_memory_ids = (
            [m.get("memory_id") for m in grounding.get("support", [])[:3]
             if m.get("memory_id")]
            + [m.get("memory_id") for m in contradicting[:3]
               if m.get("memory_id")]
            + [m.get("memory_id") for m in methodological[:3]
               if m.get("memory_id")]
        )
        action_id = store.open_receipt(
            action=action,
            sanction_verdict=verdict,
            sanction_memory_ids=sanction_memory_ids,
        )

        return {
            "action_id": action_id,
            "verdict": verdict,
            "tier": tier.value,
            "should_block": verdict == "REJECT",
            "confidence_adjustment": result.confidence_adjustment,
            "belief_confidence": effective_belief,
            "grounded_in_substrate": grounded,
            "supporting_memories": grounding.get("support", [])[:3],
            "contradicting_memories": contradicting[:3],
            "methodological_concerns": methodological[:3],
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
            grounding = {
                "support": [], "contradict": [],
                "methodological_concerns": [],
                "method": "caller_supplied",
            }
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
            # v0.9.3 (Layer 2): methodological overclaims are surfaced in a
            # separate channel from factual contradictions. The substrate
            # had factual contradiction-detection from v0.5; this catches
            # claims like "v3 was worse than v1, so CALIC is bad" against
            # a memory that says "the v1-vs-v3 conclusion is unsupported."
            "methodological_concerns": grounding.get("methodological_concerns", [])[:3],
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
    def aether_path(
        query: str,
        max_tokens: int = 2000,
        max_hops: int = 8,
    ) -> dict:
        """Shortest-path retrieval (Dijkstra) — the substrate "park map".

        Find the dependency chain that grounds the query at the cheapest
        token cost. Walks the BDG backward from the top-1 cosine match
        through SUPPORTS / DERIVED_FROM / RELATED_TO edges, weighted by
        (1 - trust) * token_estimate(text). High-trust memories are
        cheap to include; low-trust ones are expensive. CONTRADICTS
        edges are skipped entirely (held contradictions = closed paths).

        Use this instead of aether_search when you want to *preload* the
        right context for a coding task, not just find topically similar
        memories. The result is a chain of memories that fits in
        max_tokens, ordered by Dijkstra distance from the target.

        Args:
            query: text to ground
            max_tokens: budget for the returned chain (default 2000)
            max_hops: safety limit on backward walk depth (default 8)
        """
        return store.compute_path(
            query=query,
            max_tokens=max_tokens,
            max_hops=max_hops,
        )

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
    def aether_link(
        source_id: str,
        target_id: str,
        edge_type: str = "supports",
        weight: float = 0.7,
        reason: str = "",
    ) -> dict:
        """Add a typed edge between two existing memories (v0.9.1).

        Use this to wire the BDG explicitly when you know two memories
        depend on each other in a way the auto-link similarity heuristic
        won't catch. aether_path walks SUPPORTS / DERIVED_FROM /
        RELATED_TO edges, so adding them here makes the dependency
        chain visible to the park-map retrieval.

        edge_type: one of "supports", "derived_from", "related_to".
            CONTRADICTS is added by aether_remember on write;
            SUPERSEDES is added by aether_resolve.
        weight: edge weight metadata (informational; Dijkstra uses
            (1 - trust) * tokens, not this).
        reason: optional human-readable note.

        SUPPORTS / DERIVED_FROM are directional (source → target).
        RELATED_TO is symmetric and added in both directions.
        """
        try:
            return store.add_link(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                reason=reason,
            )
        except (ValueError, KeyError) as e:
            return {"error": str(e), "type": e.__class__.__name__}

    # ==================================================================
    # Action receipts — the audit half of the governance loop (v0.10)
    # ==================================================================

    @mcp.tool()
    def aether_receipt(
        action_id: str,
        result: str,
        tool_name: str = "",
        target: str = "",
        reversible: bool = False,
        reverse_action: str = "",
        details: Optional[dict] = None,
        verification_passed: Optional[bool] = None,
        verification_reason: str = "",
        model_attribution: str = "",
    ) -> dict:
        """Record the outcome of a sanctioned action (v0.10).

        Every aether_sanction call returns an action_id. After the caller
        executes (or skips) the proposed action, they cite that action_id
        here with the actual result. This closes the governance loop:
        sanction is the gate, receipts are the audit trail.

        Args:
            action_id: the id returned by aether_sanction
            result: "success" | "error" | "partial" | "skipped"
            tool_name: which tool actually ran ("shell", "git", "file_write", etc.)
            target: path / URL / command / memory_id that was acted on
            reversible: can the caller undo this action?
            reverse_action: how to undo (if reversible)
            details: arbitrary structured detail
            verification_passed: did post-action verification succeed?
            verification_reason: why pass/fail
            model_attribution: which LLM produced the action (useful for
                cross-vendor analysis)
        """
        try:
            return store.record_receipt(
                receipt_id=action_id,
                result=result,
                tool_name=tool_name or None,
                target=target or None,
                reversible=reversible,
                reverse_action=reverse_action or None,
                details=details or None,
                verification_passed=verification_passed,
                verification_reason=verification_reason or None,
                model_attribution=model_attribution or None,
            )
        except KeyError as e:
            return {"error": str(e), "type": "KeyError"}

    @mcp.tool()
    def aether_receipts(
        limit: int = 20,
        result_filter: str = "",
        verdict_filter: str = "",
        only_open: bool = False,
    ) -> dict:
        """List receipts, newest first (v0.10).

        Filters:
            result_filter   only receipts with matching `result` field
                            (e.g. "error" to find failed actions)
            verdict_filter  only receipts whose sanction returned this
                            verdict (e.g. "REJECT" to find blocked actions)
            only_open       only receipts that have not yet had their
                            outcome recorded (sanctioned but loop not closed)
        """
        return {
            "receipts": store.list_receipts(
                limit=limit,
                result_filter=result_filter or None,
                verdict_filter=verdict_filter or None,
                only_open=only_open,
            ),
        }

    @mcp.tool()
    def aether_receipt_detail(receipt_id: str) -> dict:
        """Full record for a single receipt (v0.10)."""
        r = store.get_receipt(receipt_id)
        if r is None:
            return {"error": f"unknown receipt_id: {receipt_id}", "type": "KeyError"}
        return r

    @mcp.tool()
    def aether_receipt_summary() -> dict:
        """Aggregate stats over all receipts (v0.10).

        Returns counts of verdicts (APPROVE/HOLD/REJECT), outcomes
        (success/error/partial/skipped), open receipts (sanctioned but
        loop not closed), and verification pass rate. The number to
        watch is `open_receipts` — high counts mean the agent is
        sanctioning actions but not closing the loop.
        """
        return store.receipt_summary()

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

    # ==================================================================
    # Substrate v0.14 — slot-first primitive
    # ==================================================================

    @mcp.tool()
    def aether_substrate_observe(
        namespace: str,
        slot_name: str,
        value: str,
        source_text: str = "",
        trust: float = 0.7,
        source: str = "manual",
    ) -> dict:
        """Record a (namespace, slot_name) = value observation in the substrate.

        Namespaces: 'user', 'code', 'session', 'project', 'meta'.

        If a prior state existed for this slot with a different value, the
        prior state is auto-superseded and a SUPERSEDES edge is added.
        Same-value re-observations are recorded as continued affirmation
        (history grows, no supersession).

        Persists to ~/.aether/substrate.json after the write.
        """
        sub = _get_substrate()
        state = sub.observe(
            namespace=namespace,
            slot_name=slot_name,
            value=value,
            source_text=source_text or value,
            source_type=source,
            trust=trust,
        )
        sub.save()
        return {"state": _state_to_dict(state), "slot_id": state.slot_id}

    @mcp.tool()
    def aether_substrate_current(namespace: str, slot_name: str) -> dict:
        """Most recent non-superseded state for a slot, or null."""
        sub = _get_substrate()
        s = sub.current_state(namespace, slot_name)
        return {"state": _state_to_dict(s) if s else None}

    @mcp.tool()
    def aether_substrate_history(
        namespace: str,
        slot_name: str,
        limit: int = 50,
    ) -> dict:
        """All observed states for a slot, oldest-first. Useful for tracing
        how a belief evolved (e.g. user:location across moves)."""
        sub = _get_substrate()
        history = sub.history(namespace, slot_name)
        return {
            "slot_id": f"{namespace}:{slot_name}",
            "count": len(history),
            "states": [_state_to_dict(s) for s in history[-limit:]],
        }

    @mcp.tool()
    def aether_substrate_contradictions(
        namespace: str = "",
        threshold: float = 0.5,
        limit: int = 50,
    ) -> dict:
        """Find contradicting slot states across the substrate.

        Uses NLI cross-encoder when AETHER_NLI_CONTRADICTION=1 is set
        (paraphrase-aware), else falls back to normalized-value mismatch.

        Returns pairs ordered by contradiction score descending.
        """
        sub = _get_substrate()
        ns = namespace if namespace else None
        pairs = sub.find_contradictions(namespace=ns, threshold=threshold)
        return {
            "count": len(pairs),
            "pairs": [
                {
                    "score": score,
                    "slot_id": a.slot_id,
                    "a": _state_to_dict(a),
                    "b": _state_to_dict(b),
                }
                for a, b, score in pairs[:limit]
            ],
        }

    @mcp.tool()
    def aether_substrate_slots(namespace: str = "") -> dict:
        """List all slots, optionally filtered by namespace."""
        sub = _get_substrate()
        if namespace:
            slots = sub.slots_in_namespace(namespace)
        else:
            slots = list(sub.slots.values())
        return {
            "count": len(slots),
            "slots": [
                {
                    "slot_id": s.slot_id,
                    "namespace": s.namespace,
                    "slot_name": s.slot_name,
                    "created_at": s.created_at,
                    "state_count": len(sub._states_by_slot.get(s.slot_id, [])),
                }
                for s in slots
            ],
        }

    @mcp.tool()
    def aether_substrate_stats() -> dict:
        """Substrate-wide stats: slots, states, observations, edges, namespace breakdown."""
        sub = _get_substrate()
        return sub.stats()

    @mcp.tool()
    def aether_bootstrap() -> dict:
        """Idempotent first-run setup for clients without SessionStart hooks
        (Claude Desktop, Cursor, anything that's not Claude Code).

        Performs three things, each a no-op if already done:

        1. Seeds 7 default policy beliefs (force-push, --no-verify,
           production data safety, rm -rf) so aether_sanction has
           something to gate against. Skipped if any source:default_policy
           memory already exists.
        2. Triggers encoder warmup in the background so the first
           cosine search runs warm. Returns the encoder state at the
           moment of this call ('warm' / 'warming' / 'cold' / 'unavailable').
        3. Reports current substrate stats (memory count, contradictions,
           encoder mode, version).

        Call this once after installing aether-core in a non-Claude-Code
        client. Subsequent calls are cheap and informative — useful for
        confirming aether is wired up at the start of a fresh conversation.
        """
        from aether.cli import DEFAULT_POLICY_BELIEFS

        # Step 1: seed defaults if absent.
        seeded = 0
        existing_default = False
        for node in store.graph.all_memories():
            if any(
                isinstance(t, str) and t.startswith("source:default_policy")
                for t in (node.tags or [])
            ):
                existing_default = True
                break
        if not existing_default:
            # Block briefly on encoder warmup so seeded memories get embeddings.
            if (
                getattr(store, "_encoder", None) is not None
                and hasattr(store._encoder, "wait_until_ready")
            ):
                try:
                    store._encoder.wait_until_ready(timeout=60)
                except Exception:
                    pass
            for belief in DEFAULT_POLICY_BELIEFS:
                store.add_memory(
                    text=belief["text"],
                    trust=belief["trust"],
                    source=belief["source"],
                )
                seeded += 1

        # Step 2: ensure warmup is at least started.
        encoder_mode = "unavailable"
        if getattr(store, "_encoder", None) is not None:
            try:
                if hasattr(store._encoder, "start_warmup"):
                    store._encoder.start_warmup()
            except Exception:
                pass
            stats = store.stats()
            if stats.get("embeddings_loaded"):
                encoder_mode = "warm"
            elif stats.get("embeddings_warming"):
                encoder_mode = "warming"
            else:
                encoder_mode = "cold"

        # Step 3: stats.
        stats = store.stats()
        try:
            from aether import __version__ as _ver
        except ImportError:
            _ver = "unknown"

        return {
            "aether_version": _ver,
            "seeded_default_beliefs": seeded,
            "default_beliefs_already_present": existing_default,
            "encoder_mode": encoder_mode,
            "memory_count": stats.get("memory_count", 0),
            "edge_count": stats.get("edge_count", 0),
            "held_contradictions": stats.get("held_contradictions", 0),
            "state_path": stats.get("state_path"),
            "next_steps": (
                "aether is ready. The model can call aether_remember to store "
                "facts, aether_search to recall, aether_sanction to gate "
                "actions, aether_fidelity to grade drafts. State persists at "
                f"{stats.get('state_path')}."
            ),
        }

    return mcp


def run() -> None:
    """Run the Aether MCP server over stdio."""
    store = StateStore()
    server = build_server(store=store)
    server.run("stdio")
