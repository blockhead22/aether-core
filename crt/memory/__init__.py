"""CRT Memory -- Trust-weighted memory primitives.

Fact slot extraction, memory graphs with typed edges, and belief
dependency graphs for cascade analysis.

Slot extraction (pure regex, no ML):
    from crt.memory import extract_fact_slots, ExtractedFact

    facts = extract_fact_slots("I live in Seattle and work at Microsoft")
    print(facts["location"].value)   # "Seattle"
    print(facts["employer"].value)   # "Microsoft"

Memory graph (requires networkx):
    from crt.memory import MemoryGraph, MemoryNode, EdgeType

    graph = MemoryGraph()
    graph.add_memory(MemoryNode(memory_id="m1", text="User lives in Seattle", ...))
"""

from .slots import (
    ExtractedFact,
    TemporalStatus,
    extract_fact_slots,
    extract_temporal_status,
    extract_direct_correction,
    extract_hedged_correction,
    detect_correction_type,
    create_simple_fact,
    names_are_related,
    names_look_equivalent,
    is_explicit_name_declaration_text,
    is_question,
)

from .graph import (
    MemoryType,
    BelnapState,
    EdgeType,
    Disposition,
    MemoryNode,
    ContradictionEdge,
    MemoryGraph,
    CascadeResult,
    BeliefDependencyGraph,
)

__all__ = [
    # Slots
    "ExtractedFact",
    "TemporalStatus",
    "extract_fact_slots",
    "extract_temporal_status",
    "extract_direct_correction",
    "extract_hedged_correction",
    "detect_correction_type",
    "create_simple_fact",
    "names_are_related",
    "names_look_equivalent",
    "is_explicit_name_declaration_text",
    "is_question",
    # Graph
    "MemoryType",
    "BelnapState",
    "EdgeType",
    "Disposition",
    "MemoryNode",
    "ContradictionEdge",
    "MemoryGraph",
    "CascadeResult",
    "BeliefDependencyGraph",
]
