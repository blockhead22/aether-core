"""Aether substrate -- slot-first belief primitive (v0.14+).

The substrate inverts the v0.13 architecture: slots are the typed
channel (the anchor), observations are the audit trail. Memory becomes
a derived view over the slot graph, not the primary storage.

Why
---
v0.13's bench (94.92% paraphrase-blind floor on a real GPT corpus)
demonstrated that slot extraction at write time, with slots as
decoration, leaves 95% of conversational data un-anchored. The
slot-first model makes the typed channel first-class: every belief is
a (slot, value, timestamp) tuple, contradictions are slot-state
transitions, and the dependency graph encodes typed slot relations
rather than text-similarity edges.

Status
------
**Additive.** The legacy ``aether.memory`` primitive still works.
Tonight's session lands the slot-first primitive alongside it, with
a migration script that populates the slot graph from existing
memories. The legacy path is removed in v0.15.

Public surface
--------------
::

    from aether.substrate import SubstrateGraph, SlotNode, Observation, Namespace

    sub = SubstrateGraph()
    sub.observe("user", "location", "Milwaukee",
                source_text="I moved to Milwaukee", trust=0.9)

    # Get current value
    state = sub.current_state("user", "location")
    print(state.value)  # "Milwaukee"

    # Walk history
    for s in sub.history("user", "location"):
        print(s.value, s.observed_at, s.trust)

    # Find contradictions across slot states (uses NLI if enabled)
    contras = sub.find_contradictions(namespace="user")

Migration
---------
::

    from aether.substrate import migrate_from_memory_graph
    from aether.memory import MemoryGraph

    mg = MemoryGraph(); mg.load("~/.aether/mcp_state.json")
    sub = migrate_from_memory_graph(mg)
    sub.save("~/.aether/substrate.json")
"""

from .slots import (
    Namespace,
    SlotNode,
    SlotState,
    Observation,
    SubstrateGraph,
)
from .migrate import migrate_from_memory_graph

__all__ = [
    "Namespace",
    "SlotNode",
    "SlotState",
    "Observation",
    "SubstrateGraph",
    "migrate_from_memory_graph",
]
