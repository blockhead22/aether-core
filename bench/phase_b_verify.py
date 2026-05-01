"""Phase B verify: do Phase A's new slot extractors flow through
add_memory's write-time contradiction cascade?

Tests #3 Cases A/B/C from the 2026-05-01 reflexive bench:

  A. "CRT's vector dimension is 768" vs "CRT uses 384-dim embeddings now"
     — paraphrased numeric-shape slot conflict
  B. "CRT picked Option A: Remember Me as the primary path" vs
     "CRT pivoted to Option B: structured dev tools"
     — categorical slot conflict in prose
  C. "CRT's launch timeline is 10 weeks" vs "CRT's launch timeline is 6
     weeks" — template-identical numeric conflict (passed pre-Phase A)

For each case, we:
  1. Create a fresh StateStore (tmp).
  2. Run extract_fact_slots on each text — confirm Phase A extractors fire.
  3. add_memory(A) then add_memory(B) — let write-time detector cascade run.
  4. Walk store.graph.edges and report any contradiction edges + rule_trace.

Output:
  - PASS = at least one CONTRADICTS edge between A and B with a useful trace
  - FAIL = no edge formed; print what slots WERE extracted to localize the gap

Reports inline; no file output. Run from repo root with the venv python.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from aether.memory import extract_fact_slots
from aether.mcp.state import StateStore


CASES = [
    ("A", "CRT's vector dimension is 768.",
          "CRT uses 384-dim embeddings now."),
    ("B", "CRT picked Option A: Remember Me as the primary path.",
          "CRT pivoted to Option B: structured dev tools."),
    ("C", "CRT's launch timeline is 10 weeks.",
          "CRT's launch timeline is 6 weeks."),
]


def _slot_summary(text: str) -> str:
    facts = extract_fact_slots(text)
    if not facts:
        return "(no slots)"
    return ", ".join(f"{k}={v.normalized}" for k, v in facts.items())


def _edges_between(store, mem_a_id: str, mem_b_id: str):
    """Return contradiction edges in either direction between A and B."""
    edges = []
    for src, tgt, data in store.graph.graph.edges(data=True):
        if {src, tgt} != {mem_a_id, mem_b_id}:
            continue
        if data.get("edge_type") == "contradicts":
            edges.append({
                "src": src, "tgt": tgt,
                "disposition": data.get("disposition"),
                "rule_trace": data.get("rule_trace"),
                "nli_score": data.get("nli_score"),
            })
    return edges


def main() -> int:
    out = []
    out.append("Phase B verify: write-time slot-conflict detection")
    out.append("v" + __import__("aether").__version__)
    out.append("=" * 60)

    pass_count = 0
    fail_count = 0

    for case_id, text_a, text_b in CASES:
        out.append("")
        out.append(f"### Case {case_id}")
        out.append(f"  A: {text_a}")
        out.append(f"     slots: {_slot_summary(text_a)}")
        out.append(f"  B: {text_b}")
        out.append(f"     slots: {_slot_summary(text_b)}")

        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(state_path=str(Path(tmp) / "state.json"))
            # Sync-load encoder so meter classifies in warm mode (matches
            # production behavior where the SessionStart hook warms the
            # encoder before any user write).
            if store._encoder is not None and hasattr(store._encoder, "_load"):
                try:
                    store._encoder._load()
                except Exception as e:
                    out.append(f"  (encoder load failed: {e})")

            r_a = store.add_memory(text=text_a, trust=0.92, source="case_test")
            r_b = store.add_memory(text=text_b, trust=0.92, source="case_test")
            mem_a, mem_b = r_a["memory_id"], r_b["memory_id"]

            edges = _edges_between(store, mem_a, mem_b)
            if edges:
                pass_count += 1
                out.append(f"  -> {len(edges)} CONTRADICTS edge(s):")
                for e in edges:
                    out.append(
                        f"     disposition={e['disposition']}  "
                        f"trace={e['rule_trace']}  nli={e['nli_score']:.2f}"
                    )
            else:
                fail_count += 1
                out.append("  -> NO CONTRADICTS edge formed")
                # Dump all edges for debugging.
                all_edges = list(store.graph.graph.edges(data=True))
                if all_edges:
                    out.append(f"     ({len(all_edges)} other edge(s) exist):")
                    for src, tgt, data in all_edges:
                        out.append(
                            f"       {src[:8]}->{tgt[:8]} "
                            f"type={data.get('edge_type')} "
                            f"trace={data.get('rule_trace')}"
                        )

    out.append("")
    out.append("=" * 60)
    out.append(f"Pass: {pass_count}  Fail: {fail_count}  Total: {len(CASES)}")
    print("\n".join(out))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
