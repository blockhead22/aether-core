"""Fidelity calibration benchmark runner.

Loads a corpus of (substrate, claim, expected_verdict) cases, runs each
through the substrate-grounded fidelity + sanction pipeline, and grades
the results.

Goal: turn "fidelity catches the right things" from anecdote into a
measurable property. Each category has a recall target the runner reports
on; regressions show up as the per-category number going down.

Usage:
    # Run the full corpus, print a markdown report
    python -m bench.run_fidelity_bench

    # JSON output for CI scraping
    python -m bench.run_fidelity_bench --format json

    # Run a single case (debug a failing one)
    python -m bench.run_fidelity_bench --only methodological_001_calic_canonical

The runner is also imported by tests/test_v094_fidelity_calibration.py so
the bench runs as part of `pytest tests/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CORPUS_PATH = Path(__file__).resolve().parent / "fidelity_corpus.json"


# ---------------------------------------------------------------------------
# Per-case grading
# ---------------------------------------------------------------------------

def _check_expected(
    grounding: Dict[str, Any],
    expected: Dict[str, Any],
    sanction: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Return (passed, list of failure reasons).

    Each `*_min` constraint requires at least N items; each `*_max` requires
    at most N. `sanction_verdict_in` requires the sanction verdict to be in
    the given set (only checked if sanction is provided).
    """
    failures: List[str] = []

    n_support = len(grounding.get("support", []))
    n_contradict = len(grounding.get("contradict", []))
    n_method = len(grounding.get("methodological_concerns", []))

    if "supporting_min" in expected and n_support < expected["supporting_min"]:
        failures.append(
            f"supporting_min={expected['supporting_min']}, got {n_support}"
        )
    if "supporting_max" in expected and n_support > expected["supporting_max"]:
        failures.append(
            f"supporting_max={expected['supporting_max']}, got {n_support}"
        )
    if "contradicting_min" in expected and n_contradict < expected["contradicting_min"]:
        failures.append(
            f"contradicting_min={expected['contradicting_min']}, got {n_contradict}"
        )
    if "contradicting_max" in expected and n_contradict > expected["contradicting_max"]:
        failures.append(
            f"contradicting_max={expected['contradicting_max']}, got {n_contradict}"
        )
    if "methodological_min" in expected and n_method < expected["methodological_min"]:
        failures.append(
            f"methodological_min={expected['methodological_min']}, got {n_method}"
        )
    if "methodological_max" in expected and n_method > expected["methodological_max"]:
        failures.append(
            f"methodological_max={expected['methodological_max']}, got {n_method}"
        )

    if sanction is not None and "sanction_verdict_in" in expected:
        verdict = sanction.get("verdict")
        if verdict not in expected["sanction_verdict_in"]:
            failures.append(
                f"sanction_verdict in {expected['sanction_verdict_in']}, "
                f"got {verdict}"
            )

    return (len(failures) == 0, failures)


def _seed_substrate(store, memories: List[Dict[str, Any]]) -> None:
    """Add the case's seed memories to the store."""
    for m in memories:
        store.add_memory(
            text=m["text"],
            trust=m.get("trust", 0.7),
            source=m.get("source", "user"),
        )


def _needs_sanction(case: Dict[str, Any]) -> bool:
    return "sanction_verdict_in" in case.get("expected", {})


# ---------------------------------------------------------------------------
# Corpus runner
# ---------------------------------------------------------------------------

def run_case(case: Dict[str, Any], store_factory) -> Dict[str, Any]:
    """Run one case end-to-end. Returns a dict of grading + raw outputs."""
    store = store_factory()
    _seed_substrate(store, case.get("substrate", []))

    grounding = store.compute_grounding(case["claim"])

    sanction_result: Optional[Dict[str, Any]] = None
    if _needs_sanction(case):
        sanction_result = _run_sanction(store, case["claim"])

    passed, failures = _check_expected(
        grounding,
        case.get("expected", {}),
        sanction_result,
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "failures": failures,
        "n_support": len(grounding.get("support", [])),
        "n_contradict": len(grounding.get("contradict", [])),
        "n_methodological": len(grounding.get("methodological_concerns", [])),
        "belief_confidence": grounding.get("belief_confidence"),
        "sanction_verdict": (
            sanction_result.get("verdict") if sanction_result else None
        ),
    }


def _run_sanction(store, claim: str) -> Dict[str, Any]:
    """Run aether_sanction logic against the given store + claim.

    Replicates the server.py wrapper so we don't need an MCP transport in
    the benchmark.
    """
    from aether.governance import GovernanceTier
    from aether.mcp.state import SENTINEL_BELIEF_CONF  # noqa: F401

    grounding = store.compute_grounding(claim)
    effective_belief = grounding["belief_confidence"]
    result = store.gov.govern_response(claim, belief_confidence=effective_belief)
    tier = result.tier
    if result.should_block:
        verdict = "REJECT"
    elif tier in (GovernanceTier.HEDGE, GovernanceTier.FLAG):
        verdict = "HOLD"
    else:
        verdict = "APPROVE"

    contradicting = grounding.get("contradict", [])
    if any(c["trust"] >= 0.7 for c in contradicting):
        verdict = "REJECT"

    methodological = grounding.get("methodological_concerns", [])
    if any(m["trust"] >= 0.7 for m in methodological) and verdict == "APPROVE":
        verdict = "HOLD"

    return {"verdict": verdict, "tier": tier.value}


def run_corpus(
    corpus_path: Path = CORPUS_PATH,
    only: Optional[str] = None,
    warm_encoder: bool = True,
) -> Dict[str, Any]:
    """Run all cases (or a single one if `only` is set). Returns a summary."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    if only:
        corpus = [c for c in corpus if c["id"] == only]
        if not corpus:
            raise SystemExit(f"no case with id={only!r}")

    # Pre-warm the encoder once so per-case StateStore creation is fast.
    # All subsequent stores share _MODEL_CACHE.
    if warm_encoder:
        from aether.mcp.state import StateStore
        with tempfile.TemporaryDirectory() as td:
            warm_store = StateStore(state_path=str(Path(td) / "warm.json"))
            if warm_store._encoder is not None:
                warm_store._encoder._load()

    def _store_factory():
        from aether.mcp.state import StateStore
        td = tempfile.mkdtemp(prefix="bench_")
        return StateStore(state_path=str(Path(td) / "state.json"))

    results = [run_case(c, _store_factory) for c in corpus]

    # Aggregate by category
    per_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
    for r in results:
        if r["passed"]:
            per_category[r["category"]]["pass"] += 1
        else:
            per_category[r["category"]]["fail"] += 1

    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)

    # Categories prefixed `known_gap_` are tracked but not blockers.
    # The bench reports their current pass rate so improvements / regressions
    # are visible; they don't trigger a non-zero exit code.
    blocker_results = [
        r for r in results if not r["category"].startswith("known_gap_")
    ]
    blocker_pass = sum(1 for r in blocker_results if r["passed"])
    blocker_total = len(blocker_results)

    return {
        "results": results,
        "per_category": dict(per_category),
        "total_pass": total_pass,
        "total": total,
        "pass_rate": round(total_pass / total, 3) if total else 0.0,
        "blocker_pass": blocker_pass,
        "blocker_total": blocker_total,
        "blocker_pass_rate": round(blocker_pass / blocker_total, 3) if blocker_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_markdown(summary: Dict[str, Any]) -> str:
    """Render the summary as a human-readable markdown report."""
    lines: List[str] = []
    lines.append("# Fidelity Calibration Bench -- Results")
    lines.append("")
    lines.append(
        f"**Blocker categories: {summary['blocker_pass']} / "
        f"{summary['blocker_total']} pass** "
        f"({summary['blocker_pass_rate']*100:.1f}%)"
    )
    lines.append("")
    lines.append(
        f"All categories (incl. tracked-but-not-blocking known gaps): "
        f"{summary['total_pass']} / {summary['total']} "
        f"({summary['pass_rate']*100:.1f}%)"
    )
    lines.append("")

    # Per-category breakdown
    lines.append("## Per-category recall")
    lines.append("")
    lines.append("| Category | Pass | Fail | Rate |")
    lines.append("|---|---|---|---|")
    for cat in sorted(summary["per_category"].keys()):
        cell = summary["per_category"][cat]
        total_cat = cell["pass"] + cell["fail"]
        rate = cell["pass"] / total_cat if total_cat else 0.0
        lines.append(
            f"| {cat} | {cell['pass']} | {cell['fail']} | {rate*100:.1f}% |"
        )
    lines.append("")

    # Failing cases
    failing = [r for r in summary["results"] if not r["passed"]]
    if failing:
        lines.append("## Failing cases")
        lines.append("")
        for r in failing:
            lines.append(f"### `{r['id']}` ({r['category']})")
            lines.append("")
            lines.append(
                f"- support={r['n_support']}, "
                f"contradict={r['n_contradict']}, "
                f"methodological={r['n_methodological']}"
            )
            lines.append(f"- belief_confidence={r['belief_confidence']}")
            if r.get("sanction_verdict"):
                lines.append(f"- sanction_verdict={r['sanction_verdict']}")
            for fail in r["failures"]:
                lines.append(f"- FAIL: {fail}")
            lines.append("")
    else:
        lines.append("## All cases passed")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="aether-fidelity-bench")
    parser.add_argument("--corpus", default=str(CORPUS_PATH))
    parser.add_argument("--only", default=None, help="run only the case with this id")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--no-warm", action="store_true",
                        help="skip encoder warmup (faster but less accurate)")
    args = parser.parse_args(argv)

    summary = run_corpus(
        corpus_path=Path(args.corpus),
        only=args.only,
        warm_encoder=not args.no_warm,
    )

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print(render_markdown(summary))

    # Exit non-zero only if a non-known-gap category failed.
    # Known gaps are tracked but don't block the bench.
    return 0 if summary["blocker_pass"] == summary["blocker_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
