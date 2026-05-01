"""Regression net for the fact-checker port (2026-05-01).

Verifies:
1. Public exports import cleanly from aether.crt
2. CRT volatility scoring returns plausible values
3. CRT contradiction filter respects paraphrase tolerance
4. DB operations (store/query/resolve) round-trip correctly
5. Graceful degradation when groundcheck is not installed
"""

from __future__ import annotations

import sqlite3
import tempfile
import os

import pytest

from aether.crt import (
    compute_finding_volatility,
    crt_filter_contradiction,
    get_pending_fact_checks,
    resolve_fact_check,
)
from aether.crt.fact_checker import _find_db, _ensure_table


def test_public_surface_imports():
    """All fact-checker names are importable from aether.crt."""
    from aether.crt import __all__ as exported
    for name in ("compute_finding_volatility", "crt_filter_contradiction",
                 "run_verification", "get_pending_fact_checks", "resolve_fact_check"):
        assert name in exported, f"{name} not in __all__"


def test_volatility_contradiction_higher_than_baseline():
    """Contradictions should produce higher volatility than no-issue baseline."""
    vol_contra, sev_contra = compute_finding_volatility(
        "The user works at Google",
        [{"text": "The user works at Anthropic", "source": "user"}],
        has_contradiction=True,
        has_hallucination=False,
    )
    vol_clean, sev_clean = compute_finding_volatility(
        "The user works at Anthropic",
        [{"text": "The user works at Anthropic", "source": "user"}],
        has_contradiction=False,
        has_hallucination=False,
    )
    assert vol_contra > vol_clean
    assert vol_contra > 0.0
    assert sev_contra in ("critical", "warning", "info")
    assert sev_clean in ("critical", "warning", "info")


def test_volatility_hallucination_mid_range():
    vol, sev = compute_finding_volatility(
        "I'm sure the moon is made of cheese",
        [{"text": "Basic astronomy facts", "source": "user"}],
        has_contradiction=False,
        has_hallucination=True,
    )
    assert 0.0 < vol <= 1.0
    assert sev in ("critical", "warning", "info")


def test_crt_filter_passes_real_contradiction():
    """Entity swap should be confirmed as real contradiction."""
    assert crt_filter_contradiction(
        "The user works at Google",
        "The user works at Anthropic",
        slot="employer",
    ) is True


def test_crt_filter_identical_text_not_contradiction():
    """Identical text should not be flagged as contradiction."""
    result = crt_filter_contradiction(
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        slot="",
    )
    assert result is False


def test_db_round_trip():
    """Store, query, and resolve a fact-check finding via the DB layer."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_fc.db")
        os.environ["AETHER_FACT_CHECK_DB"] = db_path

        try:
            conn = sqlite3.connect(db_path)
            _ensure_table(conn)

            import time, uuid
            check_id = f"fc_{uuid.uuid4().hex[:12]}"
            now = int(time.time())
            conn.execute(
                """INSERT INTO pending_fact_checks
                   (id, thread_id, query, response, claim, slot, issue_type,
                    details, status, volatility, severity, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (check_id, "t1", "what color?", "blue", "color is blue",
                 "color", "contradiction", "{}", 0.65, "warning", now),
            )
            conn.commit()
            conn.close()

            pending = get_pending_fact_checks(thread_id="t1")
            assert len(pending) == 1
            assert pending[0]["id"] == check_id
            assert pending[0]["volatility"] == 0.65

            assert resolve_fact_check(check_id, "corrected") is True

            pending_after = get_pending_fact_checks(thread_id="t1")
            assert len(pending_after) == 0
        finally:
            os.environ.pop("AETHER_FACT_CHECK_DB", None)


def test_run_verification_without_groundcheck():
    """run_verification gracefully returns [] when groundcheck isn't installed."""
    from aether.crt.fact_checker import run_verification
    result = run_verification(
        thread_id="test",
        query="test",
        response="test response",
        memories=[{"text": "some memory", "id": "m1"}],
    )
    assert result == []
