"""Fact-checker with CRT volatility triage.

Ported from personal_agent/auto_fact_checker.py on 2026-05-01 as part of
the full-OSS merge.  Original ran GroundCheck in a daemon thread; this
version exposes synchronous functions and lets callers decide concurrency.

The module stores findings in ``~/.aether/fact_checks.db`` (SQLite) with
CRT-computed **volatility scores** for prioritization:

    High volatility (V >= theta_reflect)  ->  "critical"
    Medium volatility                     ->  "warning"
    Low volatility                        ->  "info"

CRT math integration:
- compute_volatility(drift, alignment, contradiction, fallback) for triage
- should_reflect(V) to flag responses needing deeper review
- detect_contradiction() with entity swap / negation / paraphrase tolerance
  as a second-pass filter to reduce false positives

Optional dependency: ``groundcheck`` (pip install groundcheck).  Without
it, only the CRT-based scoring and contradiction filtering are available.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aether.crt.core import (
    CRTConfig,
    CRTMath,
    MemorySource,
    encode_vector,
)

logger = logging.getLogger(__name__)

__all__ = [
    "compute_finding_volatility",
    "crt_filter_contradiction",
    "run_verification",
    "get_pending_fact_checks",
    "resolve_fact_check",
]

# ---------------------------------------------------------------------------
# CRT math singleton
# ---------------------------------------------------------------------------
_crt_math: Optional[CRTMath] = None


def _get_crt_math() -> Optional[CRTMath]:
    global _crt_math
    if _crt_math is not None:
        return _crt_math
    try:
        _crt_math = CRTMath(CRTConfig())
        logger.info("[FACT_CHECK] CRT math loaded")
        return _crt_math
    except Exception as e:
        logger.debug("[FACT_CHECK] CRT math unavailable (%s)", e)
        return None


def _encode_text(text: str):
    try:
        return encode_vector(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CRT volatility scoring
# ---------------------------------------------------------------------------

def compute_finding_volatility(
    response_text: str,
    memories: List[Dict[str, Any]],
    has_contradiction: bool,
    has_hallucination: bool,
) -> Tuple[float, str]:
    """Compute CRT volatility score for a set of findings.

    Returns (volatility, severity) where severity is "critical"/"warning"/"info".
    """
    crt = _get_crt_math()
    if crt is None:
        if has_contradiction:
            return 0.7, "warning"
        if has_hallucination:
            return 0.5, "warning"
        return 0.3, "info"

    drift = 0.3
    memory_alignment = 0.5
    resp_vec = _encode_text(response_text[:500])

    if resp_vec is not None and memories:
        drifts = []
        alignments = []
        for m in memories[:5]:
            mem_text = m.get("text", "")
            if not mem_text:
                continue
            mem_vec = _encode_text(mem_text)
            if mem_vec is not None:
                drifts.append(crt.drift_meaning(resp_vec, mem_vec))
                alignments.append(crt.similarity(resp_vec, mem_vec))
        if drifts:
            drift = sum(drifts) / len(drifts)
        if alignments:
            memory_alignment = sum(alignments) / len(alignments)

    is_fallback = any(
        m.get("source", "").lower() in ("fallback", "llm_output")
        for m in memories
    )

    volatility = crt.compute_volatility(
        drift=drift,
        memory_alignment=memory_alignment,
        is_contradiction=has_contradiction,
        is_fallback=is_fallback,
    )

    if crt.should_reflect(volatility):
        severity = "critical"
    elif volatility >= crt.config.theta_reflect * 0.6:
        severity = "warning"
    else:
        severity = "info"

    return round(volatility, 4), severity


# ---------------------------------------------------------------------------
# CRT contradiction second-pass filter
# ---------------------------------------------------------------------------

def crt_filter_contradiction(
    claim_text: str,
    memory_text: str,
    slot: str = "",
) -> bool:
    """Second-pass CRT filter to reduce false positive contradictions.

    Returns True if CRT confirms it IS a real contradiction.
    """
    crt = _get_crt_math()
    if crt is None:
        return True

    vec_claim = _encode_text(claim_text)
    vec_mem = _encode_text(memory_text)
    if vec_claim is None or vec_mem is None:
        return True

    drift = crt.drift_meaning(vec_claim, vec_mem)

    is_contra, reason = crt.detect_contradiction(
        drift=drift,
        confidence_new=0.7,
        confidence_prior=0.7,
        source=MemorySource.USER,
        text_new=claim_text,
        text_prior=memory_text,
        slot=slot or None,
    )

    if not is_contra:
        logger.debug("[FACT_CHECK] Filtered false positive: %s", reason)

    return is_contra


# ---------------------------------------------------------------------------
# DB — findings stored in ~/.aether/fact_checks.db
# ---------------------------------------------------------------------------

def _find_db() -> Path:
    env = os.environ.get("AETHER_FACT_CHECK_DB", "").strip()
    if env:
        return Path(env)
    db_dir = Path.home() / ".aether"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "fact_checks.db"


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_fact_checks (
            id          TEXT PRIMARY KEY,
            thread_id   TEXT NOT NULL,
            query       TEXT NOT NULL,
            response    TEXT NOT NULL,
            claim       TEXT NOT NULL,
            slot        TEXT DEFAULT '',
            issue_type  TEXT NOT NULL,
            details     TEXT DEFAULT '',
            status      TEXT NOT NULL DEFAULT 'pending',
            volatility  REAL DEFAULT 0.0,
            severity    TEXT DEFAULT 'info',
            created_at  INTEGER NOT NULL,
            resolved_at INTEGER DEFAULT NULL
        )
    """)
    for col, dtype, default in [
        ("volatility", "REAL", "0.0"),
        ("severity", "TEXT", "'info'"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE pending_fact_checks ADD COLUMN {col} {dtype} DEFAULT {default}"
            )
        except sqlite3.OperationalError:
            pass


# ---------------------------------------------------------------------------
# Verification (optional groundcheck dependency)
# ---------------------------------------------------------------------------

def run_verification(
    thread_id: str,
    query: str,
    response: str,
    memories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run GroundCheck verification and store findings.

    Returns the list of findings stored, or empty list if verification
    passes or groundcheck is not installed.
    """
    try:
        from groundcheck import GroundCheck
        from groundcheck.types import Memory
    except ImportError:
        logger.debug("[FACT_CHECK] groundcheck not installed — skipping")
        return []

    gc_memories = []
    for m in memories:
        if isinstance(m, dict) and m.get("text"):
            gc_memories.append(
                Memory(
                    id=m.get("memory_id", m.get("id", "")),
                    text=m["text"],
                    trust=float(m.get("trust", 0.7)),
                    timestamp=m.get("timestamp"),
                )
            )

    if not gc_memories:
        return []

    try:
        gc = GroundCheck()
        report = gc.verify(response, gc_memories, mode="permissive")
    except Exception as e:
        logger.debug("[FACT_CHECK] Verification failed: %s", e)
        return []

    if report.passed:
        return []

    now = int(time.time())
    findings: List[Dict[str, str]] = []
    has_contradiction = False
    has_hallucination = False

    raw_hallu = getattr(report, "hallucinations", None) or getattr(report, "hallucinated_facts", None) or []
    if isinstance(raw_hallu, dict):
        items = raw_hallu.items()
    elif isinstance(raw_hallu, list):
        items = [(str(i), h) for i, h in enumerate(raw_hallu)]
    else:
        items = []
    for slot, fact in items:
        value = fact.value if hasattr(fact, "value") else str(fact)
        has_hallucination = True
        findings.append({
            "claim": f"{slot}: {value}",
            "slot": slot if not slot.isdigit() else "unknown",
            "issue_type": "hallucination",
            "details": f"Claimed '{value}' but no supporting memory found",
        })

    for contra in report.contradicted_claims or []:
        claim_text = contra.get("claim", "") if isinstance(contra, dict) else str(contra)
        slot = contra.get("slot", "") if isinstance(contra, dict) else ""

        memory_text = contra.get("memory_text", "") if isinstance(contra, dict) else ""
        if memory_text and claim_text:
            if not crt_filter_contradiction(claim_text, memory_text, slot):
                logger.debug("[FACT_CHECK] CRT filtered: %s", claim_text[:80])
                continue

        has_contradiction = True
        findings.append({
            "claim": claim_text,
            "slot": slot,
            "issue_type": "contradiction",
            "details": json.dumps(contra) if isinstance(contra, dict) else str(contra),
        })

    if not findings and not report.passed:
        findings.append({
            "claim": response[:200],
            "slot": "",
            "issue_type": "low_confidence",
            "details": f"Verification confidence {report.confidence:.2f} below threshold",
        })

    if not findings:
        return []

    volatility, severity = compute_finding_volatility(
        response, memories, has_contradiction, has_hallucination
    )

    db_path = _find_db()
    try:
        conn = sqlite3.connect(str(db_path))
        _ensure_table(conn)

        for f in findings:
            check_id = f"fc_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """INSERT INTO pending_fact_checks
                   (id, thread_id, query, response, claim, slot, issue_type, details,
                    status, volatility, severity, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (
                    check_id,
                    thread_id,
                    query[:500],
                    response[:1000],
                    f["claim"][:500],
                    f["slot"],
                    f["issue_type"],
                    f["details"][:1000],
                    volatility,
                    severity,
                    now,
                ),
            )

        conn.commit()
        conn.close()
        logger.info("[FACT_CHECK] Stored %d findings for thread=%s", len(findings), thread_id)
    except Exception as e:
        logger.warning("[FACT_CHECK] Failed to store findings: %s", e)

    return findings


# ---------------------------------------------------------------------------
# Query / resolve
# ---------------------------------------------------------------------------

def get_pending_fact_checks(
    thread_id: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Retrieve pending fact-check findings, ordered by volatility."""
    db_path = _find_db()
    if not db_path.is_file():
        return []

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        _ensure_table(conn)

        if thread_id:
            rows = conn.execute(
                """SELECT * FROM pending_fact_checks
                   WHERE status = 'pending' AND thread_id = ?
                   ORDER BY volatility DESC, created_at DESC LIMIT ?""",
                (thread_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM pending_fact_checks
                   WHERE status = 'pending'
                   ORDER BY volatility DESC, created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("[FACT_CHECK] Failed to read pending checks: %s", e)
        return []


def resolve_fact_check(check_id: str, resolution: str = "acknowledged") -> bool:
    """Mark a pending fact-check as resolved."""
    db_path = _find_db()
    if not db_path.is_file():
        return False

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE pending_fact_checks SET status = ?, resolved_at = ? WHERE id = ?",
            (resolution, int(time.time()), check_id),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.warning("[FACT_CHECK] Failed to resolve check: %s", e)
        return False
