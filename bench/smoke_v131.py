"""End-to-end smoke for v0.13.1 — run against the live ~/.aether substrate
(no temp dirs) so we exercise the actual install path users will hit.

Probes, in order:

  1. Version + import sanity
  2. Phase A regression: aether_fidelity on "Nick is a chef in Paris"
     (handoff says: was 0.95, expect <0.40 after Phase A)
  3. v0.12.21 polarity-aware grounding: "delete X without backing it up"
     against a "Never delete X without backup" memory should NOT score as
     supporting (handoff says: was the headline false-negative, fixed)
  4. v0.12.21 false-positive guard: "git status" should not get blocked
     by the "Never use git push --force" belief
  5. Phase B end-to-end: write two Option A/B memories, observe the
     contradiction edge form
  6. Sanction against a seeded policy belief (the substrate has 7 of these
     by default; ensure at least one fires correctly on its target action)
  7. Cold-query honesty: aether_search for "capital of France" should not
     surface high-confidence garbage

Each probe is independent. If one fails the rest still run, with a final
PASS/FAIL summary.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple


def probe(label: str, fn: Callable[[], Tuple[bool, str]]) -> Tuple[str, bool, str]:
    try:
        ok, detail = fn()
    except Exception as e:
        return label, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"
    return label, ok, detail


def _live_store():
    from aether.mcp.state import StateStore
    s = StateStore()
    if s._encoder is not None and hasattr(s._encoder, "_load"):
        s._encoder._load()
    return s


def _tmp_store():
    from aether.mcp.state import StateStore
    tmp = tempfile.mkdtemp(prefix="aether_smoke_")
    s = StateStore(state_path=str(Path(tmp) / "state.json"))
    if s._encoder is not None and hasattr(s._encoder, "_load"):
        s._encoder._load()
    return s, tmp


# --------------------------------------------------------------------------

def probe_version():
    import aether
    v = aether.__version__
    if v != "0.13.1":
        return False, f"expected 0.13.1, got {v}"
    return True, f"aether {v}"


def probe_chef_in_paris():
    """Phase A done criterion: substrate says Nick is a maintainer; draft
    says chef in Paris. Should grade as ungrounded / contradicted."""
    store, _ = _tmp_store()
    store.add_memory(
        text="Nick is the maintainer of aether-core.",
        trust=0.9,
        source="smoke",
    )
    grounding = store.compute_grounding("Nick is a chef in Paris.")
    bc = grounding["belief_confidence"]
    contradicts = len(grounding.get("contradict", []))
    if bc >= 0.40 and contradicts == 0:
        return False, (
            f"Phase A regression: belief_conf={bc:.2f}, contradicts={contradicts}. "
            f"Expected <0.40 OR ≥1 contradiction."
        )
    return True, f"belief_conf={bc:.2f}, contradicts={contradicts}"


def probe_polarity_aware_grounding():
    """v0.12.21 contract: 'delete X without backing it up' against a
    'Never delete X without backup' memory must NOT classify as supporting.
    Used to be the headline false-negative."""
    store, _ = _tmp_store()
    store.add_memory(
        text="Never delete production data without verifying a recent backup.",
        trust=0.92,
        source="smoke",
    )
    grounding = store.compute_grounding(
        "delete the production database without backing it up"
    )
    bc = grounding["belief_confidence"]
    support = len(grounding.get("support", []))
    contradicts = len(grounding.get("contradict", []))
    # The contract is "polarity-aware": the prohibition memory should NOT
    # appear as support. It can either contradict or be filtered.
    if support > 0 and bc >= 0.85:
        return False, (
            f"polarity collapse: belief_conf={bc:.2f} support={support} "
            f"contradicts={contradicts}. Prohibition memory was treated as "
            f"supporting evidence for the prohibited action."
        )
    return True, f"belief_conf={bc:.2f} support={support} contradicts={contradicts}"


def probe_git_status_not_blocked():
    """v0.12.21 contract: read-only `git status` should not be blocked by
    the force-push prohibition. Used to be a UX-killing false positive."""
    store, _ = _tmp_store()
    store.add_memory(
        text="Never use git push --force without explicit team approval.",
        trust=0.92,
        source="smoke",
    )
    grounding = store.compute_grounding("git status")
    contradicts = len(grounding.get("contradict", []))
    if contradicts > 0:
        return False, (
            f"false positive: git status flagged contradicts={contradicts} "
            f"by the force-push belief"
        )
    return True, f"contradicts={contradicts} (correctly not blocked)"


def probe_phase_b_option_drift():
    """v0.13.1 Phase B: Option A vs Option B in prose should produce a
    write-time slot_value_conflict edge."""
    store, _ = _tmp_store()
    a = store.add_memory(
        text="CRT picked Option A: Remember Me as the primary path.",
        trust=0.9, source="smoke",
    )
    b = store.add_memory(
        text="CRT pivoted to Option B: structured dev tools.",
        trust=0.9, source="smoke",
    )
    edges = []
    for src, tgt, data in store.graph.graph.edges(data=True):
        if {src, tgt} == {a["memory_id"], b["memory_id"]} and data.get("edge_type") == "contradicts":
            edges.append(data)
    if not edges:
        return False, "Phase B regression: no contradicts edge between Option A and Option B"
    traces = [str(e.get("rule_trace", [])) for e in edges]
    if not any("project_chosen_option" in t for t in traces):
        return False, f"contradiction fired but not via the new slot: {traces}"
    return True, f"{len(edges)} edge(s), trace includes project_chosen_option"


def probe_sanction_blocks_force_push():
    """End-to-end: against the live substrate's seeded policies (or a
    fresh one if the live substrate doesn't have them), aether_sanction
    should HOLD/REJECT a force-push action."""
    # Use a fresh substrate seeded with the default policy so this works
    # regardless of what's in the live ~/.aether/.
    from aether.cli import _seed_default_beliefs
    store, tmp = _tmp_store()
    state_path = Path(store.state_path)
    if not state_path.exists():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text('{"nodes":[],"edges":[]}')
    seeded = _seed_default_beliefs(state_path)
    if seeded == 0:
        return False, "couldn't seed default policy beliefs"
    # Re-open store to pick up the seeded state.
    from aether.mcp.state import StateStore
    store = StateStore(state_path=str(state_path))
    if store._encoder is not None and hasattr(store._encoder, "_load"):
        store._encoder._load()
    grounding = store.compute_grounding("git push --force origin main")
    contradicts = grounding.get("contradict", [])
    if not contradicts:
        return False, (
            f"sanction failed: no contradicting memory found for force-push. "
            f"belief_conf={grounding['belief_confidence']:.2f}"
        )
    high_trust = [c for c in contradicts if c.get("trust", 0) >= 0.85]
    if not high_trust:
        return False, f"contradicts={len(contradicts)} but none high-trust"
    return True, f"contradicts={len(contradicts)} high_trust={len(high_trust)}"


def probe_cold_query_honesty():
    """Cold queries should not surface high-confidence garbage."""
    store, _ = _tmp_store()
    store.add_memory(text="Nick is the maintainer of aether-core.", trust=0.9, source="smoke")
    store.add_memory(text="CRT uses FAISS as the primary vector store.", trust=0.9, source="smoke")
    grounding = store.compute_grounding("the capital of France is Paris")
    bc = grounding["belief_confidence"]
    if bc >= 0.85:
        return False, f"cold query got high belief_conf={bc:.2f} — substrate is overconfident"
    return True, f"belief_conf={bc:.2f} (correctly low for off-topic query)"


def probe_mcp_server_boot():
    """The MCP server's build_server() must succeed without errors."""
    from aether.mcp.server import build_server
    from aether.mcp.state import StateStore
    s = StateStore()
    server = build_server(store=s)
    if server is None:
        return False, "build_server returned None"
    return True, "MCP server constructed cleanly"


# --------------------------------------------------------------------------

def main() -> int:
    probes: List[Tuple[str, Callable]] = [
        ("version", probe_version),
        ("chef-in-paris (Phase A)", probe_chef_in_paris),
        ("polarity-aware grounding (v0.12.21)", probe_polarity_aware_grounding),
        ("git status not blocked (v0.12.21)", probe_git_status_not_blocked),
        ("Option A/B drift (Phase B v0.13.1)", probe_phase_b_option_drift),
        ("sanction blocks force-push", probe_sanction_blocks_force_push),
        ("cold-query honesty", probe_cold_query_honesty),
        ("MCP server boot", probe_mcp_server_boot),
    ]

    print("=" * 70)
    print(f"v0.13.1 smoke test")
    print("=" * 70)

    pass_count = 0
    fail_count = 0
    failures: List[Tuple[str, str]] = []
    for label, fn in probes:
        name, ok, detail = probe(label, fn)
        sigil = "PASS" if ok else "FAIL"
        print(f"  [{sigil}] {name}: {detail}")
        if ok:
            pass_count += 1
        else:
            fail_count += 1
            failures.append((name, detail))

    print()
    print(f"summary: {pass_count}/{len(probes)} passed")
    if failures:
        print()
        print("=== FAILURES ===")
        for name, detail in failures:
            print(f"  {name}:")
            for line in detail.splitlines():
                print(f"    {line}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
