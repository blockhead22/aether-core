"""Validation Chapter, Test #1 — substrate behavior against a fresh-session
question battery.

Runs a fixed set of questions through the substrate's read tools
(aether_search, aether_sanction, aether_fidelity) and produces a
markdown report. No LLM in the loop — so the report is a deterministic
snapshot of what the substrate would feed an LLM if asked.

Purpose: turn "the substrate surfaces useful context" from anecdote
into a re-runnable artifact. Two sessions later, the same questions
produce a new report; diffing the two shows how the substrate has
evolved.

Usage:
    # Run against the live ~/.aether/mcp_state.json (default)
    python -m bench.validation_test1

    # Specific state file
    python -m bench.validation_test1 --state-path ./my_state.json

    # Custom output destination (default: bench/results/validation_test1_{timestamp}.md)
    python -m bench.validation_test1 --out my_report.md

    # JSON output for scripted comparison
    python -m bench.validation_test1 --format json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


QUESTIONS_PATH = Path(__file__).resolve().parent / "validation_test1_questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_questions() -> Dict[str, Any]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_search(store, query: str) -> Dict[str, Any]:
    results = store.search(query, limit=5)
    return {
        "query": query,
        "result_count": len(results),
        "results": [
            {
                "trust": round(r.get("trust", 0.0), 2),
                "similarity": (
                    round(r["similarity"], 3) if r.get("similarity") is not None else None
                ),
                "score": round(r.get("score", 0.0), 3),
                "text": r.get("text", "")[:120],
            }
            for r in results
        ],
    }


def _run_sanction(store, action: str) -> Dict[str, Any]:
    """Replicate aether_sanction's logic without going through the MCP layer.
    Returns the verdict and the supporting / contradicting trace.
    """
    grounding = store.compute_grounding(action)
    gov_result = store.gov.govern_response(
        action, belief_confidence=grounding["belief_confidence"]
    )
    contradicting = grounding.get("contradict", [])
    if any(c.get("trust", 0) >= 0.7 for c in contradicting):
        verdict = "REJECT"
    elif gov_result.should_block:
        verdict = "REJECT"
    elif gov_result.tier.value in ("hedge", "flag"):
        verdict = "HOLD"
    else:
        verdict = "APPROVE"
    return {
        "action": action,
        "verdict": verdict,
        "tier": gov_result.tier.value,
        "belief_confidence": round(grounding["belief_confidence"], 3),
        "supporting_count": len(grounding.get("support", [])),
        "contradicting_count": len(contradicting),
    }


def _run_fidelity(store, draft: str) -> Dict[str, Any]:
    grounding = store.compute_grounding(draft)
    return {
        "draft": draft,
        "belief_confidence": round(grounding["belief_confidence"], 3),
        "supporting_count": len(grounding.get("support", [])),
        "contradicting_count": len(grounding.get("contradict", [])),
        "method": grounding.get("method", "unknown"),
        "supporting": [
            {
                "trust": round(s.get("trust", 0.0), 2),
                "text": s.get("text", "")[:100],
            }
            for s in grounding.get("support", [])[:3]
        ],
        "contradicting": [
            {
                "trust": round(c.get("trust", 0.0), 2),
                "text": c.get("text", "")[:100],
            }
            for c in grounding.get("contradict", [])[:3]
        ],
    }


def _run_question(store, q: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch one question to the right tool, return its captured output."""
    tool = q.get("tool")
    if tool == "search":
        return {"id": q["id"], "tool": tool, "result": _run_search(store, q["query"])}
    if tool == "sanction":
        return {"id": q["id"], "tool": tool, "result": _run_sanction(store, q["action"])}
    if tool == "fidelity":
        return {"id": q["id"], "tool": tool, "result": _run_fidelity(store, q["draft"])}
    return {"id": q["id"], "tool": tool, "result": {"error": f"unknown tool: {tool}"}}


def _render_search(r: Dict[str, Any]) -> str:
    if r["result_count"] == 0:
        return "_no relevant memory_"
    lines = [
        "| trust | sim | score | text |",
        "|------:|----:|------:|------|",
    ]
    for hit in r["results"]:
        sim = hit["similarity"] if hit["similarity"] is not None else "—"
        text = hit["text"].replace("|", "\\|")
        lines.append(f"| {hit['trust']:.2f} | {sim} | {hit['score']:.3f} | {text} |")
    return "\n".join(lines)


def _render_sanction(r: Dict[str, Any]) -> str:
    return (
        f"- verdict: **{r['verdict']}**\n"
        f"- tier: {r['tier']}\n"
        f"- belief_confidence: {r['belief_confidence']}\n"
        f"- supporting / contradicting memories: "
        f"{r['supporting_count']} / {r['contradicting_count']}"
    )


def _render_fidelity(r: Dict[str, Any]) -> str:
    lines = [
        f"- belief_confidence: {r['belief_confidence']}",
        f"- supporting / contradicting memories: "
        f"{r['supporting_count']} / {r['contradicting_count']}",
        f"- grounding method: `{r['method']}`",
    ]
    if r["supporting"]:
        lines.append("- top supporting:")
        for s in r["supporting"]:
            text = s["text"].replace("|", "\\|")
            lines.append(f"  - trust={s['trust']:.2f}: {text}")
    if r["contradicting"]:
        lines.append("- top contradicting:")
        for c in r["contradicting"]:
            text = c["text"].replace("|", "\\|")
            lines.append(f"  - trust={c['trust']:.2f}: {text}")
    return "\n".join(lines)


def _render_markdown(spec: Dict[str, Any], answers: List[Dict[str, Any]],
                     state_path: str, substrate_stats: Dict[str, Any]) -> str:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    by_id = {a["id"]: a for a in answers}

    lines = [
        f"# {spec['_meta']['name']}",
        "",
        f"_Generated: {timestamp}. Substrate: `{state_path}` "
        f"({substrate_stats.get('memory_count', '?')} memories, "
        f"{substrate_stats.get('edge_count', '?')} edges, "
        f"aether-core {substrate_stats.get('aether_version', '?')}, "
        f"encoder: {substrate_stats.get('encoder_mode', '?')})._",
        "",
        f"**Purpose.** {spec['_meta']['purpose']}",
        "",
        f"**Scope.** {spec['_meta']['scope']}",
        "",
    ]

    for cat in spec["categories"]:
        lines.append(f"## Category {cat['id']}: {cat['name']}")
        lines.append("")
        lines.append(f"_{cat['what_to_look_for']}_")
        lines.append("")
        for q in cat["questions"]:
            answer = by_id.get(q["id"], {})
            lines.append(f"### {q['id']}: {q.get('query') or q.get('action') or q.get('draft')}")
            lines.append("")
            lines.append(f"**Tool:** `{q['tool']}`")
            lines.append("")
            lines.append(f"**Expectation.** {q['expectation']}")
            lines.append("")
            lines.append("**Result:**")
            lines.append("")
            r = answer.get("result", {})
            if q["tool"] == "search":
                lines.append(_render_search(r))
            elif q["tool"] == "sanction":
                lines.append(_render_sanction(r))
            elif q["tool"] == "fidelity":
                lines.append(_render_fidelity(r))
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "_Future work: pair this snapshot with a no-substrate baseline "
        "(same questions against an empty StateStore) and a ground-truth "
        "label set. The diff is the substrate's value._"
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validation_test1",
        description="Run validation chapter test #1 against the substrate.",
    )
    parser.add_argument("--state-path", default=None,
                        help="state file to read (default: AETHER_STATE_PATH or ~/.aether/mcp_state.json)")
    parser.add_argument("--out", default=None,
                        help="markdown output path (default: bench/results/validation_test1_{ts}.md)")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown",
                        help="output format (default markdown)")
    parser.add_argument("--wait-warmup", type=float, default=60.0,
                        help="seconds to block on encoder warmup before running queries "
                             "(default 60; pass 0 to skip and run in cold mode)")
    args = parser.parse_args(argv)

    spec = _load_questions()

    try:
        from aether.mcp.state import StateStore
    except ImportError as e:
        print(f"FAIL: aether-core not installed: {e}", file=sys.stderr)
        return 1

    if args.state_path:
        store = StateStore(state_path=args.state_path)
    else:
        store = StateStore()

    # Wait for encoder warmup (or skip explicitly). Without this the
    # harness measures cold-mode behavior, which is meaningfully
    # different from warm-mode and should not be the default snapshot.
    encoder_mode = "cold"
    if args.wait_warmup > 0 and getattr(store, "_encoder", None) is not None:
        if hasattr(store._encoder, "wait_until_ready"):
            ready = store._encoder.wait_until_ready(timeout=args.wait_warmup)
            encoder_mode = "warm" if ready else "cold (warmup timed out)"

    state_path = store.state_path
    substrate_stats = store.stats()
    substrate_stats["encoder_mode"] = encoder_mode
    try:
        from aether import __version__ as _v
    except ImportError:
        _v = "unknown"
    substrate_stats["aether_version"] = _v

    answers: List[Dict[str, Any]] = []
    for cat in spec["categories"]:
        for q in cat["questions"]:
            try:
                answers.append(_run_question(store, q))
            except Exception as e:
                answers.append({
                    "id": q["id"],
                    "tool": q["tool"],
                    "result": {"error": f"{type(e).__name__}: {e}"},
                })

    if args.format == "json":
        payload = {
            "spec": spec["_meta"],
            "state_path": state_path,
            "substrate_stats": substrate_stats,
            "answers": answers,
        }
        out_text = json.dumps(payload, indent=2, default=str)
    else:
        out_text = _render_markdown(spec, answers, state_path, substrate_stats)

    if args.out:
        out_path = Path(args.out)
    elif args.format == "markdown":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = RESULTS_DIR / f"validation_test1_{stamp}.md"
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
