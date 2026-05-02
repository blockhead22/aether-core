"""Migration: legacy MemoryGraph -> slot-first SubstrateGraph.

Strategy
--------
Walk every memory in the legacy graph. For each, attempt to:

1. Recognize an explicit slot pattern (``user favorite color: orange``,
   ``user location: Seattle``, ``user employer: Microsoft``). These
   come from auto-ingest and have a stable ``user X: Y`` shape.
2. Fall through to ``extract_fact_slots()`` on the memory text.
3. If neither yields a slot, record the memory as a ``meta:archived``
   observation so the audit trail is preserved without forcing a slot.

Trust, tags, created_at, and the original memory_id are preserved on
the resulting Observation.
"""

from __future__ import annotations

import re
from typing import Optional

from .slots import SubstrateGraph, Namespace, TemporalStatus

# Match auto-ingest's canonical "user <slot>: <value>" pattern.
_USER_PATTERN = re.compile(
    r"^user\s+([a-z_]+):\s+(.+?)(?:\s*\(observed\s+\d+x.*\))?$",
    re.IGNORECASE,
)


def migrate_from_memory_graph(memory_graph) -> SubstrateGraph:
    """Build a SubstrateGraph from a v0.13 MemoryGraph.

    Parameters
    ----------
    memory_graph : aether.memory.MemoryGraph
        Loaded legacy graph (call .load() first).

    Returns
    -------
    SubstrateGraph
        Populated. Caller is responsible for saving.
    """
    # Local import to avoid hard dependency
    try:
        from aether.memory import extract_fact_slots
    except Exception:
        extract_fact_slots = None  # noqa: E731

    sub = SubstrateGraph()
    n_user_slots = 0
    n_fallback_slots = 0
    n_archived = 0

    for mem in memory_graph.all_memories():
        text = (mem.text or "").strip()
        if not text:
            continue

        observation_id = None
        slot_recorded = False

        # 1. Direct user:slot:value pattern from auto-ingest
        m = _USER_PATTERN.match(text)
        if m:
            slot_name = m.group(1).strip().lower()
            value = m.group(2).strip().rstrip('.,;').strip('"\'')
            if value:
                state = sub.observe(
                    namespace=Namespace.USER.value,
                    slot_name=slot_name,
                    value=value,
                    source_text=text,
                    source_type="migration",
                    trust=mem.trust,
                    temporal_status=TemporalStatus.ACTIVE.value,
                    source="migration_user_pattern",
                )
                observation_id = state.observation_id
                # carry legacy memory_id in observation metadata
                obs = sub.observations.get(observation_id)
                if obs is not None:
                    obs.metadata["legacy_memory_id"] = mem.memory_id
                    obs.metadata["legacy_tags"] = list(mem.tags or [])
                    obs.metadata["legacy_belnap"] = mem.belnap_state
                    obs.observed_at = mem.created_at or obs.observed_at
                    state.observed_at = mem.created_at or state.observed_at
                n_user_slots += 1
                slot_recorded = True

        # 2. Fall through to regex extractor on free-form text
        if not slot_recorded and extract_fact_slots is not None:
            try:
                facts = extract_fact_slots(text) or {}
            except Exception:
                facts = {}
            if facts:
                # Single observation, multiple emitted states
                from .slots import Observation
                obs = Observation.new(source_text=text, source_type="migration")
                obs.metadata["legacy_memory_id"] = mem.memory_id
                obs.metadata["legacy_tags"] = list(mem.tags or [])
                obs.observed_at = mem.created_at or obs.observed_at
                sub.observations[obs.observation_id] = obs
                for slot_name, fact in facts.items():
                    val = (fact.value or "").strip().strip('"\'').rstrip('.,;')
                    if not val:
                        continue
                    state = sub.observe(
                        namespace=Namespace.USER.value,
                        slot_name=slot_name,
                        value=val,
                        source_text=text,
                        source_type="migration",
                        trust=mem.trust,
                        temporal_status=getattr(fact, "temporal_status", TemporalStatus.ACTIVE.value),
                        observation_id=obs.observation_id,
                        normalized=getattr(fact, "normalized", None),
                        source="migration_regex",
                    )
                    state.observed_at = mem.created_at or state.observed_at
                    n_fallback_slots += 1
                slot_recorded = True

        # 3. Archive un-slotted memories so audit trail is preserved
        if not slot_recorded:
            from .slots import Observation
            obs = Observation.new(source_text=text, source_type="migration_archived")
            obs.metadata["legacy_memory_id"] = mem.memory_id
            obs.metadata["legacy_tags"] = list(mem.tags or [])
            obs.metadata["legacy_trust"] = mem.trust
            obs.observed_at = mem.created_at or obs.observed_at
            sub.observations[obs.observation_id] = obs
            # Emit a meta:archived state so the observation is reachable
            state = sub.observe(
                namespace=Namespace.META.value,
                slot_name="archived_memory",
                value=text[:200],
                source_text=text,
                source_type="migration",
                trust=mem.trust,
                observation_id=obs.observation_id,
                source="migration_archived",
            )
            state.observed_at = mem.created_at or state.observed_at
            n_archived += 1

    sub.stats_migration = {
        "user_pattern_slots": n_user_slots,
        "regex_fallback_slots": n_fallback_slots,
        "archived_memories": n_archived,
    }
    return sub
