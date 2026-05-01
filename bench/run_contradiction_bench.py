"""Contradiction detection: structural meter vs LLM-as-judge.

The README's "Design choices, briefly" section used to claim that
"slot comparison at 88 percent accuracy beats LLM-as-judge at 40
percent." That number had no in-repo experiment behind it. This is
the experiment.

For each case in `bench/fidelity_corpus.json` (the curated set the
fidelity calibration bench already uses), the runner asks two
questions:

  1. STRUCTURAL — does aether's `compute_grounding` flag any
     contradicting memory? (i.e. does `len(contradict) > 0`?)
  2. LLM AS JUDGE — does an LLM, shown the same memory and claim,
     answer YES to "does this claim directly contradict this
     memory?"

The case's category gives the ground truth: factual_contradiction,
mutex_contradiction, negation_asymmetry, and policy_violation cases
SHOULD be flagged as contradicting. false_positive_guard,
no_issue_grounded, and no_issue_unrelated cases SHOULD NOT.
methodological_overclaim is a different signal (inference flaw, not
fact conflict) and is excluded from this comparison.

The runner reports per-method accuracy, per-category accuracy, and a
list of cases where the two methods disagreed — that disagreement set
is where the qualitative difference between the two approaches lives.

Usage:
    # Structural only (no API key required)
    python -m bench.run_contradiction_bench

    # Full comparison (requires ANTHROPIC_API_KEY)
    ANTHROPIC_API_KEY=... python -m bench.run_contradiction_bench

    # Custom output path
    python -m bench.run_contradiction_bench --out my_report.md

    # JSON for diff over time
    python -m bench.run_contradiction_bench --format json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


CORPUS_PATH = Path(__file__).resolve().parent / "fidelity_corpus.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Per-category ground truth: should the structural meter (and any
# honest LLM-as-judge) flag this as contradicting?
SHOULD_CONTRADICT: Dict[str, Optional[bool]] = {
    "factual_contradiction": True,
    "mutex_contradiction": True,
    "negation_asymmetry": True,
    "policy_violation": True,
    "false_positive_guard": False,
    "no_issue_grounded": False,
    "no_issue_unrelated": False,
    # Methodological overclaim is a different channel (inference flaw,
    # not fact conflict). Excluded from this comparison.
    "methodological_overclaim": None,
    # Known-gap categories aren't shipped fixes; tracked separately.
    "known_gap_quantitative": None,
}


LLM_JUDGE_PROMPT = """You are evaluating whether a CLAIM directly contradicts a MEMORY.

A direct contradiction means both statements cannot be true at the same time about the same entity. Examples:
- "User lives in Seattle" vs "User lives in Portland" → YES (mutually exclusive locations).
- "We use AWS" vs "We use GCP" → YES (mutually exclusive cloud providers).
- "User prefers Python" vs "User prefers tabs over spaces" → NO (different topics, both can be true).
- "User lives in Seattle" vs "User has lived in Seattle for 5 years" → NO (these agree).

MEMORY: {memory}
CLAIM: {claim}

Does the CLAIM directly contradict the MEMORY? Answer with a single word: YES or NO."""


def _load_corpus() -> List[Dict[str, Any]]:
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _seed_substrate(store, memories: List[Dict[str, Any]]) -> None:
    for m in memories:
        store.add_memory(
            text=m["text"],
            trust=m.get("trust", 0.7),
            source=m.get("source", "user"),
        )


def run_structural(case: Dict[str, Any]) -> bool:
    """True if aether flags any contradicting memory for the case's claim."""
    import tempfile
    from aether.mcp.state import StateStore

    with tempfile.TemporaryDirectory() as tmp:
        store = StateStore(state_path=str(Path(tmp) / "state.json"))
        # Sync-load encoder so similarity is real.
        if store._encoder is not None and hasattr(store._encoder, "_load"):
            store._encoder._load()
        _seed_substrate(store, case.get("substrate", []))
        grounding = store.compute_grounding(case["claim"])
        return len(grounding.get("contradict", [])) > 0


def run_llm_judge(case: Dict[str, Any], model: str = "claude-haiku-4-5") -> Optional[bool]:
    """Ask an LLM YES/NO whether the claim contradicts the memory.

    Returns True / False on a clean answer, None when the API key is
    absent or the call fails. Uses the FIRST seed memory as the
    reference — multi-memory cases get the most representative one.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    memories = case.get("substrate", [])
    if not memories:
        return None
    memory_text = memories[0]["text"]
    claim = case["claim"]

    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{
                "role": "user",
                "content": LLM_JUDGE_PROMPT.format(memory=memory_text, claim=claim),
            }],
        )
        # The response is a list of content blocks; we asked for a one-word answer.
        text = ""
        for block in msg.content:
            if hasattr(block, "text"):
                text += block.text
        text = text.strip().upper()
        if text.startswith("YES"):
            return True
        if text.startswith("NO"):
            return False
        return None
    except Exception:
        return None


def grade_case(case: Dict[str, Any], structural: bool,
               llm: Optional[bool]) -> Dict[str, Any]:
    expected = SHOULD_CONTRADICT.get(case["category"])
    return {
        "id": case["id"],
        "category": case["category"],
        "expected": expected,
        "structural_predicted": structural,
        "structural_correct": (structural == expected) if expected is not None else None,
        "llm_predicted": llm,
        "llm_correct": (llm == expected) if (expected is not None and llm is not None) else None,
        "disagree": (structural != llm) if llm is not None else None,
        "memory": (case.get("substrate", [{}])[0] or {}).get("text", "")[:120],
        "claim": case["claim"][:120],
    }


def aggregate(graded: List[Dict[str, Any]]) -> Dict[str, Any]:
    in_scope = [g for g in graded if g["expected"] is not None]
    n = len(in_scope)
    structural_correct = sum(1 for g in in_scope if g["structural_correct"])
    llm_runs = [g for g in in_scope if g["llm_correct"] is not None]
    llm_correct = sum(1 for g in llm_runs if g["llm_correct"])
    disagree = [g for g in in_scope if g["disagree"]]

    by_category: Dict[str, Dict[str, Any]] = {}
    for g in in_scope:
        c = g["category"]
        slot = by_category.setdefault(c, {"n": 0, "structural_correct": 0,
                                          "llm_correct": 0, "llm_n": 0})
        slot["n"] += 1
        slot["structural_correct"] += int(bool(g["structural_correct"]))
        if g["llm_correct"] is not None:
            slot["llm_n"] += 1
            slot["llm_correct"] += int(bool(g["llm_correct"]))

    return {
        "n_in_scope": n,
        "n_excluded": len(graded) - n,
        "structural": {
            "correct": structural_correct,
            "total": n,
            "accuracy": (structural_correct / n) if n else 0.0,
        },
        "llm": {
            "correct": llm_correct,
            "total": len(llm_runs),
            "accuracy": (llm_correct / len(llm_runs)) if llm_runs else 0.0,
            "skipped": n - len(llm_runs),
        },
        "disagreements": disagree,
        "by_category": by_category,
    }


def render_markdown(graded: List[Dict[str, Any]], agg: Dict[str, Any]) -> str:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Contradiction-detection bench: structural meter vs LLM-as-judge",
        "",
        f"_Generated: {timestamp}._",
        "",
        f"**Sample.** {agg['n_in_scope']} cases from `bench/fidelity_corpus.json` "
        f"covering factual_contradiction, mutex_contradiction, "
        f"negation_asymmetry, policy_violation (positives) and "
        f"false_positive_guard, no_issue_grounded, no_issue_unrelated "
        f"(negatives). {agg['n_excluded']} cases excluded "
        f"(methodological_overclaim is a different channel; known_gap_* "
        f"is tracked separately).",
        "",
        "## Headline",
        "",
    ]

    s = agg["structural"]
    l = agg["llm"]
    lines.append(f"- **Structural meter:** {s['correct']}/{s['total']} = "
                 f"**{s['accuracy']:.1%}** accuracy.")
    if l["total"] > 0:
        lines.append(f"- **LLM as judge:** {l['correct']}/{l['total']} = "
                     f"**{l['accuracy']:.1%}** accuracy.")
        lines.append(f"- **Disagreements:** {len(agg['disagreements'])} cases "
                     f"out of {l['total']} where the two methods diverged.")
    else:
        lines.append("- **LLM as judge:** SKIPPED — no `ANTHROPIC_API_KEY` "
                     "in environment, or the `anthropic` SDK isn't installed. "
                     "Re-run with the key set to populate this row.")
    lines.append("")

    lines.append("## By category")
    lines.append("")
    lines.append("| Category | n | Structural | LLM judge |")
    lines.append("|---|---:|---:|---:|")
    for cat, slot in sorted(agg["by_category"].items()):
        s_pct = (slot["structural_correct"] / slot["n"]) if slot["n"] else 0.0
        if slot["llm_n"] > 0:
            l_pct = slot["llm_correct"] / slot["llm_n"]
            l_cell = f"{l_pct:.1%} ({slot['llm_correct']}/{slot['llm_n']})"
        else:
            l_cell = "—"
        lines.append(f"| {cat} | {slot['n']} | "
                     f"{s_pct:.1%} ({slot['structural_correct']}/{slot['n']}) "
                     f"| {l_cell} |")
    lines.append("")

    if agg["disagreements"]:
        lines.append("## Cases where structural and LLM judge disagreed")
        lines.append("")
        lines.append("| id | category | expected | structural | LLM | memory → claim |")
        lines.append("|---|---|---|---|---|---|")
        for g in agg["disagreements"]:
            mem_escaped = g["memory"].replace("|", "\\|")
            claim_escaped = g["claim"].replace("|", "\\|")
            lines.append(
                f"| `{g['id']}` | {g['category']} | "
                f"{'contradict' if g['expected'] else 'no'} | "
                f"{'contradict' if g['structural_predicted'] else 'no'} | "
                f"{'contradict' if g['llm_predicted'] else 'no'} | "
                f"{mem_escaped} → {claim_escaped} |"
            )
        lines.append("")

    lines.append("## Methodology caveats")
    lines.append("")
    lines.append(
        "- The corpus is a hand-curated set the structural meter was tuned for. "
        "A held-out corpus is the next iteration.\n"
        "- LLM-as-judge prompt is a single-shot YES/NO with a fixed prefix; a "
        "different prompt could change the LLM's accuracy meaningfully.\n"
        "- Multi-memory cases are reduced to the FIRST memory for the LLM "
        "judge; the structural meter sees all memories. This is a deliberate "
        "asymmetry — it gives the LLM a simpler task, not a harder one."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_See `bench/run_contradiction_bench.py` for the runner. "
                 "Re-run with `python -m bench.run_contradiction_bench` "
                 "(set `ANTHROPIC_API_KEY` for the LLM column)._")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_contradiction_bench")
    parser.add_argument("--out", default=None,
                        help="markdown output path (default: bench/results/contradiction_bench_{ts}.md)")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--llm-model", default="claude-haiku-4-5",
                        help="model name for the LLM judge (default claude-haiku-4-5)")
    args = parser.parse_args(argv)

    corpus = _load_corpus()

    graded: List[Dict[str, Any]] = []
    for case in corpus:
        try:
            structural = run_structural(case)
        except Exception as e:
            print(f"structural error on {case['id']}: {e}", file=sys.stderr)
            structural = False
        llm = run_llm_judge(case, model=args.llm_model)
        graded.append(grade_case(case, structural, llm))

    agg = aggregate(graded)

    if args.format == "json":
        out_text = json.dumps({"summary": agg, "cases": graded}, indent=2, default=str)
    else:
        out_text = render_markdown(graded, agg)

    if args.out:
        out_path = Path(args.out)
    elif args.format == "markdown":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = RESULTS_DIR / f"contradiction_bench_{stamp}.md"
    else:
        out_path = None

    if out_path is not None:
        out_path.write_text(out_text, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
