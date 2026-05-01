"""`aether` CLI: scaffold, inspect, and validate a substrate.

Subcommands:
    aether init             create .aether/ in the current directory
    aether status           show substrate stats
    aether check            run fidelity on a commit message file (pre-commit hook)
    aether contradictions   list current contradictions
    aether backfill-edges   retroactively wire RELATED_TO edges (v0.9.1)
    aether doctor           diagnose install / substrate / hook health (v0.12.4)

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
# backfill-edges (v0.9.1)
# ---------------------------------------------------------------------------

def cmd_backfill_edges(args) -> int:
    """Wire RELATED_TO edges into substrates built before v0.9.1.

    v0.9.0 had a bug: aether_remember produced orphan nodes. Without
    SUPPORTS / RELATED_TO edges, aether_path always returned just the
    target. This command retroactively wires RELATED_TO edges between
    similar memories so existing substrates benefit from aether_path
    without re-ingesting.
    """
    store = _make_store()
    if store._encoder is not None and not args.no_wait:
        # Block on warmup so similarity uses real embeddings, not the
        # token-overlap fallback. Backfill is a one-shot operation —
        # the wait is acceptable.
        try:
            store._encoder._load()
        except Exception as e:
            print(f"warning: encoder load failed ({e}); using token-overlap fallback",
                  file=sys.stderr)

    threshold = args.threshold
    if threshold is None:
        from aether.mcp.state import AUTO_LINK_THRESHOLD
        threshold = AUTO_LINK_THRESHOLD

    if args.dry_run:
        # Count without writing by temporarily monkey-patching the
        # graph add_edge. Simpler: fork the math here.
        memories = list(store.graph.all_memories())
        would_add = 0
        compared = 0
        for i, a in enumerate(memories):
            a_emb = store.graph.get_embedding(a.memory_id)
            for b in memories[i + 1:]:
                compared += 1
                if (store.graph.graph.has_edge(a.memory_id, b.memory_id) or
                        store.graph.graph.has_edge(b.memory_id, a.memory_id)):
                    continue
                b_emb = store.graph.get_embedding(b.memory_id)
                if a_emb is not None and b_emb is not None:
                    sim = store._cosine(a_emb, b_emb)
                else:
                    ta = set(a.text.lower().split())
                    tb = set(b.text.lower().split())
                    sim = len(ta & tb) / max(len(ta | tb), 1)
                if sim >= threshold:
                    would_add += 1
        out = {
            "dry_run": True,
            "would_add": would_add,
            "compared_pairs": compared,
            "threshold": threshold,
            "total_memories": len(memories),
        }
    else:
        out = store.backfill_edges(threshold=threshold)

    if args.format == "json":
        print(json.dumps(out, indent=2))
    else:
        if args.dry_run:
            print(f"backfill (dry-run): would add {out['would_add']} RELATED_TO "
                  f"edges across {out['total_memories']} memories "
                  f"(threshold={out['threshold']:.2f})")
        else:
            print(f"backfill: added {out['added']} RELATED_TO edges "
                  f"across {out['total_memories']} memories "
                  f"(threshold={out['threshold']:.2f})")
            print(f"  compared pairs       : {out['compared_pairs']}")
            print(f"  skipped (had edge)   : {out['skipped_existing_edge']}")
            print(f"  skipped (low sim)    : {out['skipped_low_sim']}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# doctor (v0.12.4)
# ---------------------------------------------------------------------------
#
# Surfaces the kind of silent breakage we kept hitting in v0.9-v0.12:
# F#1 (mcp extra missing networkx, install crashed at first tool call),
# F#3 (corrupt nodes crash read tools), F#7 (auto-ingest hook firing but
# server clobbering its writes). Each check runs in seconds and either
# returns OK or points at a concrete fix. `aether doctor` is the first
# thing to run when "nothing seems to be working."

# Status sigils — kept ASCII so `aether doctor` looks right in any terminal.
_OK = "[ OK ]"
_WARN = "[WARN]"
_FAIL = "[FAIL]"


def _doctor_install_imports() -> dict:
    """Verify core + optional imports the tools depend on."""
    issues = []
    for pkg, extra in (
        ("aether", None),
        ("networkx", "graph"),
        ("mcp", "mcp"),
    ):
        try:
            __import__(pkg)
        except ImportError as e:
            extra_hint = f" (install with `pip install aether-core[{extra}]`)" if extra else ""
            issues.append(f"{pkg} import failed: {e}{extra_hint}")

    # sentence-transformers is optional — warn, don't fail.
    warns = []
    try:
        __import__("sentence_transformers")
    except ImportError:
        warns.append(
            "sentence_transformers not installed; substrate runs in cold "
            "mode (Jaccard fallback). Install with `pip install aether-core[ml]` "
            "for full warm-mode grounding."
        )

    if issues:
        return {"status": "fail", "name": "install_imports", "messages": issues}
    if warns:
        return {"status": "warn", "name": "install_imports", "messages": warns}
    return {"status": "ok", "name": "install_imports",
            "messages": ["all required + optional packages importable"]}


def _doctor_state_file(state_path: Optional[str]) -> dict:
    """Confirm the substrate file is readable, parseable, and not corrupt."""
    from aether.mcp.state import _default_state_path
    path = Path(state_path or _default_state_path())
    if not path.exists():
        return {
            "status": "warn",
            "name": "state_file",
            "messages": [
                f"state file does not exist yet: {path}",
                "this is fine on a fresh install — first write will create it.",
            ],
        }
    try:
        size = path.stat().st_size
    except OSError as e:
        return {"status": "fail", "name": "state_file",
                "messages": [f"state file unreadable ({path}): {e}"]}
    if size == 0:
        return {"status": "warn", "name": "state_file",
                "messages": [f"state file is empty: {path}"]}

    try:
        with path.open("r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "status": "fail",
            "name": "state_file",
            "messages": [
                f"state file is corrupt ({path}): {e}",
                "consider restoring from a backup in the same directory "
                "(e.g. mcp_state.pre_v096_cleanup_*.json).",
            ],
        }

    # F#3 echo: scan nodes for the corrupt-deserialization shape.
    corrupt_ids = []
    for node in data.get("nodes", []):
        if not all(k in node for k in ("id", "memory_id", "text", "created_at")):
            corrupt_ids.append(node.get("id") or "<missing-id>")

    msgs = [
        f"state file: {path}",
        f"size: {size:,} bytes",
        f"nodes: {len(data.get('nodes', []))}",
        f"edges: {len(data.get('edges', []))}",
        f"aether_version on disk: {data.get('aether_version', '<pre-v0.12.1, unstamped>')}",
    ]
    if corrupt_ids:
        msgs.append(
            f"!!! {len(corrupt_ids)} corrupt node(s) detected: "
            f"{corrupt_ids[:5]}{'...' if len(corrupt_ids) > 5 else ''}. "
            "These will crash aether_memory_detail / aether_lineage / "
            "aether_cascade_preview when accessed (F#3). Repair via a "
            "manual JSON edit + server restart."
        )
        return {"status": "fail", "name": "state_file", "messages": msgs}
    return {"status": "ok", "name": "state_file", "messages": msgs}


def _doctor_substrate_activity(state_path: Optional[str]) -> dict:
    """Has the auto-ingest hook (or anything else) been writing recently?

    A long-stale substrate in the presence of active sessions implies
    the Stop hook isn't firing. Surfaced this exact silent failure
    today (the 3-day no-op transcript_path bug).
    """
    import time
    from aether.mcp.state import _default_state_path
    path = Path(state_path or _default_state_path())
    if not path.exists():
        return {"status": "warn", "name": "substrate_activity",
                "messages": ["no state file yet — nothing to measure"]}
    try:
        mtime = path.stat().st_mtime
    except OSError as e:
        return {"status": "warn", "name": "substrate_activity",
                "messages": [f"could not stat state file: {e}"]}
    age_s = time.time() - mtime
    age_h = age_s / 3600
    msg = f"last substrate write: {age_h:.1f} hours ago"
    if age_h > 24 * 7:
        return {"status": "warn", "name": "substrate_activity",
                "messages": [msg + " (>7 days). Auto-ingest hook may be silently no-oping."]}
    return {"status": "ok", "name": "substrate_activity", "messages": [msg]}


def _doctor_backups(state_path: Optional[str]) -> dict:
    """Confirm rotating backups exist and surface their freshness.

    The substrate file is single-point-of-failure; without backups a
    corrupt write or bad import takes the substrate with it. v0.12.10
    introduced rotating backups in `{state_dir}/backups/` on every
    save. This check verifies they're being written and warns if the
    most recent one is stale (suggests `_rotate_backups` is no-op'ing
    silently — possibly AETHER_DISABLE_BACKUPS=1 left set).
    """
    import time
    from aether.mcp.state import _default_state_path
    sp = Path(state_path or _default_state_path())
    if not sp.exists():
        return {
            "status": "warn",
            "name": "backups",
            "messages": ["no state file yet — nothing to back up"],
        }
    backup_dir = sp.parent / "backups"
    if not backup_dir.exists() or not backup_dir.is_dir():
        return {
            "status": "warn",
            "name": "backups",
            "messages": [
                f"no backups directory at {backup_dir}",
                "rotating backups land here on every save (v0.12.10+). "
                "Will be created on the next write unless "
                "AETHER_DISABLE_BACKUPS=1 is set.",
            ],
        }
    backups = sorted(
        backup_dir.glob(f"{sp.stem}.*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not backups:
        return {
            "status": "warn",
            "name": "backups",
            "messages": [
                f"backups directory exists but contains no rotations for "
                f"{sp.stem}: {backup_dir}",
            ],
        }
    newest_age_h = (time.time() - backups[0].stat().st_mtime) / 3600
    state_age_h = (time.time() - sp.stat().st_mtime) / 3600
    msgs = [
        f"backup directory: {backup_dir}",
        f"rotations on disk: {len(backups)}",
        f"newest backup: {backups[0].name} ({newest_age_h:.1f}h ago)",
    ]
    # If the state file has been written more recently than the newest
    # backup, rotation isn't firing. That's a regression worth flagging.
    if state_age_h + 0.05 < newest_age_h:
        # state newer than backup is normal (state just saved, backup
        # is from the *previous* state). Only flag the inverse.
        pass
    if newest_age_h > state_age_h + 24:
        return {
            "status": "warn",
            "name": "backups",
            "messages": msgs + [
                "newest backup is much older than the state file. Rotation "
                "may have stopped — check AETHER_DISABLE_BACKUPS.",
            ],
        }
    return {"status": "ok", "name": "backups", "messages": msgs}


def _doctor_encoder() -> dict:
    """Probe the lazy encoder synchronously and report its state."""
    try:
        from aether._lazy_encoder import LazyEncoder
    except ImportError as e:
        return {"status": "fail", "name": "encoder",
                "messages": [f"encoder module unavailable: {e}"]}

    enc = LazyEncoder()
    # _load() is synchronous — tolerable in a CLI diagnostic.
    enc._load()
    if enc.is_unavailable:
        return {
            "status": "warn",
            "name": "encoder",
            "messages": [
                "sentence-transformers cannot load (install via "
                "`pip install aether-core[ml]`). Substrate falls back "
                "to Jaccard token overlap; warm-mode grounding "
                "thresholds won't be reached.",
            ],
        }
    if enc.is_loaded:
        return {"status": "ok", "name": "encoder",
                "messages": [f"encoder loaded: {enc.model_name}"]}
    return {"status": "warn", "name": "encoder",
            "messages": ["encoder neither loaded nor flagged unavailable — odd state"]}


def _doctor_claude_code_hook() -> dict:
    """Look for a Stop hook pointing at an aether ingest script.

    Claude Code searches a few config locations; we walk up from cwd
    looking for `.claude/settings.json[.local]` (project-level), then
    fall back to `~/.claude/settings.json` (user-global). Walking up
    matters because users run `aether doctor` from anywhere in the
    project tree, not necessarily the root that owns the .claude dir.

    Best-effort: finding no hook isn't fatal (the user might use a
    different client) but is worth surfacing because we hit the silent
    no-op transcript_path bug for 3 days before noticing.
    """
    candidates = []
    cwd = Path.cwd()
    for ancestor in (cwd, *cwd.parents):
        for fname in ("settings.local.json", "settings.json"):
            p = ancestor / ".claude" / fname
            if p not in candidates:
                candidates.append(p)
    candidates.append(Path.home() / ".claude" / "settings.json")
    found = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stop_hooks = (cfg.get("hooks", {}) or {}).get("Stop", [])
        for entry in stop_hooks:
            for h in entry.get("hooks", []):
                cmd = str(h.get("command", ""))
                if "aether" in cmd.lower() or "auto_ingest" in cmd.lower():
                    found.append(f"{p}: {cmd}")
    if found:
        return {"status": "ok", "name": "claude_code_hook",
                "messages": ["Stop hook(s) wired:", *[f"  {f}" for f in found]]}
    return {
        "status": "warn",
        "name": "claude_code_hook",
        "messages": [
            "no aether Stop hook found in known config locations.",
            "auto-ingest won't fire after each Claude Code turn.",
            "see examples/claude-code-hooks/auto_ingest_hook.py to wire one.",
        ],
    }


def _doctor_mcp_registration() -> dict:
    """Look for an MCP server registration that points at aether."""
    candidates = [
        Path.cwd() / ".mcp.json",
        Path.home() / ".mcp.json",
    ]
    found = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for name, server in (cfg.get("mcpServers", {}) or {}).items():
            args = " ".join(str(a) for a in server.get("args", []))
            if "aether" in name.lower() or "aether" in args.lower():
                found.append(f"{p}: {name} = {server.get('command')} {args}")
    if found:
        return {"status": "ok", "name": "mcp_registration",
                "messages": ["aether MCP server registered:", *[f"  {f}" for f in found]]}
    return {
        "status": "warn",
        "name": "mcp_registration",
        "messages": [
            "no aether MCP server registration found in .mcp.json files.",
            "the MCP tools won't be available to your client.",
            "add an entry like {\"mcpServers\":{\"aether\":{\"command\":\"python\",\"args\":[\"-m\",\"aether.mcp\"]}}}",
        ],
    }


def cmd_doctor(args) -> int:
    state_path = getattr(args, "state_path", None)
    results = [
        _doctor_install_imports(),
        _doctor_state_file(state_path),
        _doctor_substrate_activity(state_path),
        _doctor_backups(state_path),
        _doctor_encoder(),
        _doctor_claude_code_hook(),
        _doctor_mcp_registration(),
    ]

    if args.format == "json":
        print(json.dumps({"checks": results}, indent=2))
    else:
        for r in results:
            sigil = {"ok": _OK, "warn": _WARN, "fail": _FAIL}[r["status"]]
            print(f"{sigil} {r['name']}")
            for m in r["messages"]:
                print(f"       {m}")
        n_fail = sum(1 for r in results if r["status"] == "fail")
        n_warn = sum(1 for r in results if r["status"] == "warn")
        n_ok = sum(1 for r in results if r["status"] == "ok")
        print(f"\nsummary: {n_ok} ok, {n_warn} warn, {n_fail} fail")

    return 1 if any(r["status"] == "fail" for r in results) else 0


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

    # backfill-edges (v0.9.1)
    p_back = sub.add_parser(
        "backfill-edges",
        help="retroactively wire RELATED_TO edges (v0.9.1 fix for v0.9.0 substrates)",
    )
    p_back.add_argument("--threshold", type=float, default=None,
                        help="similarity threshold (default: AUTO_LINK_THRESHOLD)")
    p_back.add_argument("--dry-run", action="store_true",
                        help="count what would be added without writing")
    p_back.add_argument("--no-wait", action="store_true",
                        help="skip blocking on encoder warmup (uses token-overlap fallback)")
    p_back.add_argument("--format", choices=["text", "json"], default="text")
    p_back.set_defaults(func=cmd_backfill_edges)

    # doctor (v0.12.4)
    p_doc = sub.add_parser(
        "doctor",
        help="diagnose install / substrate / hook / mcp wiring health",
    )
    p_doc.add_argument("--state-path", default=None,
                       help="state file to check (default: AETHER_STATE_PATH or ~/.aether/mcp_state.json)")
    p_doc.add_argument("--format", choices=["text", "json"], default="text")
    p_doc.set_defaults(func=cmd_doctor)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
