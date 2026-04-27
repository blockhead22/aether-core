"""`aether` CLI: scaffold, inspect, and validate a substrate.

Subcommands:
    aether init       create .aether/ in the current directory
    aether status     show substrate stats
    aether check      run fidelity on a commit message file (pre-commit hook)
    aether contradictions    list current contradictions

Each subcommand is a thin wrapper over `aether.mcp.state.StateStore`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def _make_store(state_path: Optional[str] = None):
    """Lazy import so `aether init` works even before deps are installed."""
    from aether.mcp.state import StateStore
    return StateStore(state_path=state_path)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

INIT_README = """\
# .aether/

This directory is the project's belief substrate. It records facts about
this codebase that survive across sessions and developers.

## What lives here

- `state.json` — the substrate. Memories, edges, Belnap states.
- `state_trust_history.json` — append-only log of every trust change.
- `state_embeddings.npz` — sentence-transformers vectors (if [ml] is installed).

## Sharing with your team

Commit `state.json` and `state_trust_history.json` to git. Onboarding a
new developer pulls the project's accumulated decisions for free.

Don't commit `state_embeddings.npz` — it's regenerated from text on
demand, costs cycles to load, and can be huge. Add it to `.gitignore`
(this `init` already did that for you).

## Working with it

In any agent client that speaks MCP (Claude Code, Cursor, Cline, etc.),
the aether-core MCP server will discover this directory automatically
when started from anywhere inside the project tree. Tools like
`aether_remember`, `aether_search`, `aether_sanction`, `aether_fidelity`
will read and write here instead of the user-global substrate at
`~/.aether/mcp_state.json`.

To force the user-global substrate instead, set
`AETHER_NO_REPO_DISCOVERY=1`.
"""


GITIGNORE_LINES = """\
# Aether substrate -- regenerable artifacts
state_embeddings.npz
"""


def cmd_init(args) -> int:
    target = Path(args.dir or os.getcwd()) / ".aether"
    if target.exists() and not args.force:
        print(f"refusing to overwrite existing {target} (use --force)")
        return 1

    target.mkdir(parents=True, exist_ok=True)

    state_file = target / "state.json"
    if not state_file.exists() or args.force:
        state_file.write_text(json.dumps({"nodes": [], "edges": []}, indent=2))

    history_file = target / "state_trust_history.json"
    if not history_file.exists() or args.force:
        history_file.write_text("{}\n")

    readme = target / "README.md"
    if not readme.exists() or args.force:
        readme.write_text(INIT_README)

    gitignore = target / ".gitignore"
    if not gitignore.exists() or args.force:
        gitignore.write_text(GITIGNORE_LINES)

    print(f"initialized aether substrate at {target}")
    print()
    print("Next steps:")
    try:
        rel = target.relative_to(Path.cwd())
        print(f"  1. git add {rel}")
    except ValueError:
        # target on a different drive or above cwd — show absolute
        print(f"  1. git add {target}")
    print("  2. Commit so your team inherits the substrate")
    print(
        "  3. Run any MCP-aware agent in this tree -- "
        "aether-core will discover this dir automatically"
    )
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args) -> int:
    store = _make_store()
    stats = store.stats()
    print(f"substrate: {stats['state_path']}")
    print(f"  memories     : {stats['memory_count']}")
    print(f"  edges        : {stats['edge_count']}")
    print(f"  belnap       : {stats.get('belnap_states', {})}")
    print(f"  edge types   : {stats.get('edge_types', {})}")
    print(f"  held        : {stats.get('held_contradictions', 0)}")
    print(f"  evolving    : {stats.get('evolving_contradictions', 0)}")
    print(f"  embeddings   : configured={stats.get('embeddings_available', False)}, "
          f"loaded={stats.get('embeddings_loaded', False)}, "
          f"warming={stats.get('embeddings_warming', False)}")
    return 0


# ---------------------------------------------------------------------------
# contradictions
# ---------------------------------------------------------------------------

def cmd_contradictions(args) -> int:
    store = _make_store()
    contras = store.list_contradictions(disposition=args.disposition or None)
    if not contras:
        print("no contradictions in substrate")
        return 0
    for c in contras:
        a = c["memory_a"]
        b = c["memory_b"]
        print(f"[{c['disposition']:11}] {c['tension_score']:.2f}")
        print(f"    A: {a['text'][:100]}  (trust {a['trust']:.2f})")
        print(f"    B: {b['text'][:100]}  (trust {b['trust']:.2f})")
        print()
    return 0


# ---------------------------------------------------------------------------
# check (pre-commit / CI fidelity)
# ---------------------------------------------------------------------------

def cmd_check(args) -> int:
    """Run fidelity on a text source. Designed for pre-commit / CI.

    Inputs (any of, combined):
        --message-file PATH    Read text from a file (e.g. .git/COMMIT_EDITMSG)
        --message TEXT         Inline message text
        --diff PATH            Read a diff hunk to include as context
        stdin                  When --stdin is passed
    """
    parts: list[str] = []

    if args.message:
        parts.append(args.message)
    if args.message_file:
        try:
            parts.append(Path(args.message_file).read_text(encoding="utf-8"))
        except OSError as e:
            print(f"could not read message file: {e}", file=sys.stderr)
            return 2
    if args.diff:
        try:
            parts.append(Path(args.diff).read_text(encoding="utf-8"))
        except OSError as e:
            print(f"could not read diff: {e}", file=sys.stderr)
            return 2
    if args.stdin:
        parts.append(sys.stdin.read())

    body = "\n".join(p for p in parts if p).strip()
    if not body:
        print("no input provided. pass --message, --message-file, --diff, or --stdin", file=sys.stderr)
        return 2

    store = _make_store()
    grounding = store.compute_grounding(body)
    belief_conf = grounding["belief_confidence"]

    from aether.governance.gap_auditor import ResponseAudit, Severity
    audit_input = ResponseAudit(
        response_text=body,
        belief_confidence=belief_conf,
    )
    verdict = store.gov.gap_auditor.audit(audit_input)

    block_threshold = args.fail_severity or "CRITICAL"
    sev_order = {"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2}
    block_at = sev_order.get(block_threshold.upper(), 2)
    actual = sev_order.get(verdict.severity.value, 0)

    out = {
        "severity": verdict.severity.value,
        "action": verdict.action.value,
        "gap_score": verdict.gap_score,
        "speech_confidence": verdict.speech_confidence,
        "belief_confidence": verdict.belief_confidence,
        "grounding_method": grounding.get("method"),
        "supporting": grounding["support"][:3],
        "contradicting": grounding["contradict"][:3],
        "block_threshold": block_threshold.upper(),
    }

    if args.format == "json":
        print(json.dumps(out, indent=2))
    else:
        print(f"aether check: severity={verdict.severity.value} "
              f"action={verdict.action.value} gap={verdict.gap_score:.2f}")
        if grounding["contradict"]:
            print()
            print("Contradicting memories in substrate:")
            for c in grounding["contradict"][:3]:
                kind = c.get("kind", "?")
                print(f"  [{kind}] (trust {c['trust']:.2f}) {c['text'][:120]}")
        if grounding["support"]:
            print()
            print("Supporting memories in substrate:")
            for s in grounding["support"][:3]:
                print(f"  (trust {s['trust']:.2f}) {s['text'][:120]}")

    if actual >= block_at:
        return 1  # Block the commit / fail CI
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aether",
        description="Aether: belief substrate CLI.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # init
    p_init = sub.add_parser("init", help="create .aether/ in the current directory")
    p_init.add_argument("--dir", default=None, help="where to create .aether/")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.set_defaults(func=cmd_init)

    # status
    p_status = sub.add_parser("status", help="show substrate stats")
    p_status.set_defaults(func=cmd_status)

    # contradictions
    p_con = sub.add_parser("contradictions", help="list contradictions")
    p_con.add_argument("--disposition", default=None,
                       choices=["resolvable", "held", "evolving", "contextual"],
                       help="filter by disposition")
    p_con.set_defaults(func=cmd_contradictions)

    # check
    p_check = sub.add_parser("check", help="run fidelity on text (pre-commit / CI)")
    p_check.add_argument("--message", default=None,
                         help="inline message text")
    p_check.add_argument("--message-file", default=None,
                         help="read message from a file")
    p_check.add_argument("--diff", default=None,
                         help="optional diff file to include as context")
    p_check.add_argument("--stdin", action="store_true",
                         help="read message from stdin")
    p_check.add_argument("--format", choices=["text", "json"], default="text")
    p_check.add_argument("--fail-severity", default="CRITICAL",
                         choices=["SAFE", "ELEVATED", "CRITICAL"],
                         help="exit non-zero at this severity or above")
    p_check.set_defaults(func=cmd_check)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
