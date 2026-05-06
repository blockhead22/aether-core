"""CRT integration adapter.

Targets a CRT (Coherent Recall Triplet) facts store. The schema we read is::

    facts(id, slot, value, trust, source, timestamp, is_current, superseded_by, thread_id)

Configuration
-------------
Env vars (all optional; integration is off unless the mode flag is set):

  * ``AETHER_CRT_INTEGRATION``  — ``read`` | ``write`` | ``consult`` | ``full``
  * ``AETHER_CRT_FACTS_DB``     — path to crt_facts.db.
                                  Default: ``D:/AI_round2/personal_agent/crt_facts.db``.
                                  Override for portability.
  * ``AETHER_CRT_THREAD_ID``    — restrict to a single thread when set.
  * ``AETHER_CRT_INCLUDE_LAB``  — set truthy to include ``lab_*`` thread rows
                                  (deliberate contradiction-injection tests).
                                  Default: excluded.

Mode coverage in this module:

  * ``read``    — implemented. ``search()`` queries the CRT facts table
                  and returns substrate-shaped result dicts.
  * ``write``   — implemented. ``sync_to_substrate()`` ingests current
                  CRT facts as substrate observations. Idempotent across
                  re-runs via an in-memory imported-fact-id set.
  * ``consult`` — implemented. ``consult_substrate_for_action()`` is a
                  read-only gate CRT calls before committing a fact
                  write. Returns a verdict (pass/affirm/supersede/warn)
                  with evidence drawn from current substrate state.

Read mode is safe even when the DB is missing or unreadable — every
failure path returns ``[]`` rather than raising.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import is_read_enabled, is_write_enabled, is_consult_enabled

DEFAULT_CRT_FACTS_DB = "D:/AI_round2/personal_agent/crt_facts.db"
SOURCE = "crt"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _facts_db_path() -> Optional[str]:
    """Resolve crt_facts.db path. Returns None if file is missing."""
    p = os.environ.get("AETHER_CRT_FACTS_DB", DEFAULT_CRT_FACTS_DB).strip()
    if not p:
        return None
    if not Path(p).exists():
        return None
    return p


def _thread_filter() -> Optional[str]:
    raw = os.environ.get("AETHER_CRT_THREAD_ID", "").strip()
    return raw or None


_TRUTHY = {"1", "true", "yes", "on"}


def _include_lab() -> bool:
    return os.environ.get("AETHER_CRT_INCLUDE_LAB", "").strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Read mode (C1)
# ---------------------------------------------------------------------------


def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Substring search over CRT facts. Returns substrate-shaped dicts.

    Each result mirrors the keys used by ``StateStore.search`` so callers
    can merge the two without case-by-case handling. Distinguishing
    field: ``"crt_origin": True`` and ``source.startswith("crt:")``.

    Returns [] when:
      * read mode is not enabled
      * facts DB is missing
      * query is empty
      * any error path fires
    """
    if not is_read_enabled(SOURCE):
        return []
    if not query or not query.strip():
        return []
    db = _facts_db_path()
    if db is None:
        return []

    q_lower = query.lower().strip()
    thread = _thread_filter()

    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        # Pull current facts; light substring filter happens in Python so
        # we can re-use the same scoring shape as the substrate search.
        params: List[Any] = []
        sql = "SELECT id, slot, value, trust, source, timestamp FROM facts WHERE is_current = 1"
        if thread:
            sql += " AND thread_id = ?"
            params.append(thread)
        elif not _include_lab():
            # Exclude deliberate contradiction-injection test threads. They
            # carry is_current=1 but represent research artifacts, not real
            # preference signals — leaving them in pollutes value rollups.
            sql += " AND (thread_id IS NULL OR thread_id NOT LIKE 'lab_%')"
        rows = cur.execute(sql, params).fetchall()
        con.close()
    except sqlite3.Error:
        return []

    q_tokens = set(q_lower.split())

    # Aggregate by (slot, normalized value). CRT is append-only and
    # value-frequency itself is signal — collapsing 134 rows of
    # `favorite_color: orange` into a single result with count=N and
    # max trust gives a far more honest picture than returning 5
    # arbitrary rows.
    agg: Dict[tuple[str, str], Dict[str, Any]] = {}

    for fact_id, slot, value, trust, src, ts in rows:
        text = f"{slot}: {value}"
        text_lower = text.lower()
        text_normalized = text_lower.replace("_", " ").replace(".", " ")
        substring_score = 0.0
        if q_lower in text_normalized:
            substring_score += 1.0
        t_tokens = set(text_normalized.split())
        overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
        substring_score += overlap * 0.5
        if substring_score <= 0:
            continue

        norm_value = (value or "").strip().lower()
        key = (slot, norm_value)
        trust_f = float(trust or 0.0)

        existing = agg.get(key)
        if existing is None:
            agg[key] = {
                "memory_id": f"crt:{fact_id}",
                "text": text,
                "trust": trust_f,
                "source": f"crt:{src}",
                "substring_score": substring_score,
                "similarity": None,
                "timestamp": ts,
                "slot": slot,
                "value": value,
                "crt_origin": True,
                "count": 1,
                "max_trust": trust_f,
            }
        else:
            existing["count"] += 1
            if trust_f > existing["max_trust"]:
                existing["max_trust"] = trust_f
                existing["trust"] = trust_f
                existing["text"] = text
                existing["value"] = value
            if ts and (not existing["timestamp"] or ts > existing["timestamp"]):
                existing["timestamp"] = ts

    # Final scoring: substring × trust × log-count bonus. Frequent values
    # rank above one-offs even when both match equally.
    import math
    out: List[Dict[str, Any]] = []
    for d in agg.values():
        count_bonus = 1.0 + math.log1p(d["count"] - 1) * 0.5
        combined = d["substring_score"] * (0.3 + 0.7 * d["max_trust"]) * count_bonus
        d["score"] = combined
        d.pop("substring_score", None)
        d.pop("max_trust", None)
        out.append(d)

    out.sort(key=lambda r: -r["score"])
    return out[:limit]


# ---------------------------------------------------------------------------
# Write mode (C2) — sync CRT facts into substrate via observe()
# ---------------------------------------------------------------------------


def _split_slot(slot: str) -> tuple[str, str]:
    """CRT slots are dotted (``user.favorite_color``). Map to substrate's
    (namespace, slot_name). Slots without a dot fall under the ``crt``
    namespace so the source is still distinguishable in the graph.
    """
    if "." in slot:
        ns, _, name = slot.partition(".")
        return ns or "crt", name or slot
    return "crt", slot


def sync_to_substrate(substrate, max_facts: int = 0, dry_run: bool = False) -> Dict[str, Any]:
    """Pull current CRT facts into the substrate via ``observe()``.

    Idempotent across re-runs: each sync tracks imported CRT fact ids on
    the substrate via the ``_crt_imported_fact_ids`` attribute and skips
    facts already seen. (In-memory only — a substrate reload starts the
    set fresh; document if you need cross-process persistence.)

    Args:
        substrate: a ``SubstrateGraph``-like object exposing ``observe()``.
        max_facts: cap the number of newly-imported facts in this call.
            ``0`` means no cap. The cap counts only facts actually
            observed, not skipped duplicates.
        dry_run: if True, plan the work and return counts without calling
            ``observe()``. Useful for previewing a sync.

    Returns a status dict::

        {
          "status":   "ok" | "disabled" | "no_db" | "error",
          "mode":     "write",
          "enabled":  bool,
          "imported": int,   # facts newly observed this call
          "skipped":  int,   # facts already imported in a prior call
          "scanned":  int,   # facts considered after thread/lab filter
          "dry_run":  bool,
        }
    """
    if not is_write_enabled(SOURCE):
        return {"status": "disabled", "mode": "write", "enabled": False,
                "imported": 0, "skipped": 0, "scanned": 0, "dry_run": dry_run}
    if substrate is None:
        return {"status": "error", "mode": "write", "enabled": True,
                "imported": 0, "skipped": 0, "scanned": 0, "dry_run": dry_run,
                "error": "substrate is None"}

    db = _facts_db_path()
    if db is None:
        return {"status": "no_db", "mode": "write", "enabled": True,
                "imported": 0, "skipped": 0, "scanned": 0, "dry_run": dry_run}

    thread = _thread_filter()
    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        params: List[Any] = []
        sql = ("SELECT id, slot, value, trust, source, timestamp "
               "FROM facts WHERE is_current = 1")
        if thread:
            sql += " AND thread_id = ?"
            params.append(thread)
        elif not _include_lab():
            sql += " AND (thread_id IS NULL OR thread_id NOT LIKE 'lab_%')"
        sql += " ORDER BY timestamp ASC, id ASC"
        rows = cur.execute(sql, params).fetchall()
        con.close()
    except sqlite3.Error as e:
        return {"status": "error", "mode": "write", "enabled": True,
                "imported": 0, "skipped": 0, "scanned": 0, "dry_run": dry_run,
                "error": str(e)}

    imported_ids = getattr(substrate, "_crt_imported_fact_ids", None)
    if imported_ids is None:
        imported_ids = set()
        try:
            substrate._crt_imported_fact_ids = imported_ids
        except AttributeError:
            # Substrate-likes that forbid attribute assignment: fall back
            # to a per-call set, losing cross-call idempotency. Caller's
            # choice.
            pass

    imported = 0
    skipped = 0
    scanned = len(rows)

    for fact_id, slot, value, trust, src, ts in rows:
        if fact_id in imported_ids:
            skipped += 1
            continue
        if max_facts and imported >= max_facts:
            break
        ns, slot_name = _split_slot(slot or "")
        if not slot_name:
            continue
        if dry_run:
            imported += 1
            continue
        try:
            substrate.observe(
                ns,
                slot_name,
                str(value if value is not None else ""),
                source_text=f"crt-fact:{fact_id}",
                source_type=SOURCE,
                trust=float(trust if trust is not None else 0.7),
                source=f"crt:{src or 'unknown'}",
            )
        except Exception as e:  # noqa: BLE001 — sync must not crash caller
            return {"status": "error", "mode": "write", "enabled": True,
                    "imported": imported, "skipped": skipped, "scanned": scanned,
                    "dry_run": dry_run, "error": f"observe failed on fact {fact_id}: {e}"}
        imported_ids.add(fact_id)
        imported += 1

    return {"status": "ok", "mode": "write", "enabled": True,
            "imported": imported, "skipped": skipped, "scanned": scanned,
            "dry_run": dry_run}


# ---------------------------------------------------------------------------
# Consult mode (C3) — substrate-grounded gate for CRT-side fact writes
# ---------------------------------------------------------------------------


# Verdicts the consult gate emits. Caller decides what to do with each;
# the gate itself is read-only and never mutates the substrate.
VERDICT_PASS = "pass"               # no prior state — proposed write is novel
VERDICT_AFFIRM = "affirm"           # same normalized value as current state
VERDICT_SUPERSEDE = "supersede"     # different value, proposed trust >= current
VERDICT_WARN = "warn"               # different value, proposed trust < current
VERDICT_PASSTHROUGH = "pass-through"  # consult disabled or action unsupported


def consult_substrate_for_action(
    substrate: Any,
    action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read-only substrate gate for a proposed CRT action.

    CRT calls this before committing a fact write so it can decide
    whether to proceed, defer to substrate's prior belief, or surface
    a contradiction. The gate never mutates the substrate.

    ``action`` shape (only ``write`` is gated today)::

        {"kind": "write", "slot": "user.favorite_color",
         "value": "cyan", "trust": 0.9}

    Verdicts:
      * ``pass``        — slot has no prior state.
      * ``affirm``      — proposed value matches current normalized value.
      * ``supersede``   — different value, proposed trust >= current
                          effective trust. Caller is overriding cleanly.
      * ``warn``        — different value, proposed trust < current
                          effective trust. Caller may still proceed but
                          should know the substrate disagrees.
      * ``pass-through`` — consult disabled, substrate missing, or
                          unsupported action kind. Always-call-safe.

    The shape always carries ``proposed`` (echoes the input) and, when
    a current state exists, ``evidence`` with the substrate's view.
    """
    enabled = is_consult_enabled(SOURCE)
    base = {
        "status": "ok",
        "mode": "consult",
        "enabled": enabled,
        "verdict": VERDICT_PASSTHROUGH,
    }

    if not enabled:
        base["status"] = "disabled"
        return base
    if substrate is None:
        base["status"] = "error"
        base["error"] = "substrate is None"
        return base
    if not isinstance(action, dict):
        base["status"] = "error"
        base["error"] = "action must be a dict"
        return base

    kind = str(action.get("kind", "write")).lower()
    slot = action.get("slot")
    if not slot or not isinstance(slot, str):
        base["status"] = "error"
        base["error"] = "action.slot is required"
        return base

    proposed_value = action.get("value", "")
    proposed_value_str = "" if proposed_value is None else str(proposed_value)
    try:
        proposed_trust = float(action.get("trust", 0.7))
    except (TypeError, ValueError):
        proposed_trust = 0.7

    base["proposed"] = {
        "kind": kind,
        "slot": slot,
        "value": proposed_value_str,
        "trust": proposed_trust,
    }

    # Only fact-write actions are gated. Anything else (responses,
    # tool calls, etc.) passes through with consult still recorded as
    # enabled — leaves room to extend without breaking callers.
    if kind != "write":
        return base

    ns, slot_name = _split_slot(slot)
    try:
        current = substrate.current_state(ns, slot_name)
    except Exception as e:  # noqa: BLE001 — gate must not crash caller
        base["status"] = "error"
        base["error"] = f"current_state failed: {e}"
        return base

    if current is None:
        base["verdict"] = VERDICT_PASS
        return base

    try:
        current_trust = float(current.effective_trust())
    except Exception:  # noqa: BLE001
        current_trust = float(getattr(current, "trust", 0.0))

    base["evidence"] = {
        "namespace": ns,
        "slot_name": slot_name,
        "value": current.value,
        "normalized": getattr(current, "normalized", (current.value or "").strip().lower()),
        "trust": float(getattr(current, "trust", 0.0)),
        "effective_trust": current_trust,
        "observed_at": float(getattr(current, "observed_at", 0.0)),
        "source": getattr(current, "source", "unknown"),
        "state_id": getattr(current, "state_id", None),
    }

    proposed_norm = proposed_value_str.strip().lower()
    if proposed_norm == base["evidence"]["normalized"]:
        base["verdict"] = VERDICT_AFFIRM
        return base

    if proposed_trust >= current_trust:
        base["verdict"] = VERDICT_SUPERSEDE
    else:
        base["verdict"] = VERDICT_WARN
    return base


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def diagnostics() -> Dict[str, Any]:
    """Snapshot of CRT integration state. Useful for `aether doctor`."""
    db = _facts_db_path()
    fact_count: Optional[int] = None
    if db:
        try:
            con = sqlite3.connect(db)
            fact_count = con.execute(
                "SELECT count(*) FROM facts WHERE is_current = 1"
            ).fetchone()[0]
            con.close()
        except sqlite3.Error:
            fact_count = -1
    return {
        "source": SOURCE,
        "mode": os.environ.get("AETHER_CRT_INTEGRATION", "").strip().lower(),
        "read_enabled": is_read_enabled(SOURCE),
        "write_enabled": is_write_enabled(SOURCE),
        "consult_enabled": is_consult_enabled(SOURCE),
        "facts_db": db,
        "facts_db_set_via_env": bool(os.environ.get("AETHER_CRT_FACTS_DB")),
        "thread_filter": _thread_filter(),
        "current_facts": fact_count,
    }
