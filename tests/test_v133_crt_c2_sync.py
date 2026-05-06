"""C2: CRT → substrate sync via observe()."""

from __future__ import annotations

import sqlite3

import pytest

from aether.integrations import crt
from aether.substrate import SubstrateGraph


def _make_db(path, rows):
    """rows: iterable of (id, slot, value, trust, source, timestamp,
    is_current, superseded_by, thread_id)."""
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE facts ("
        "id INTEGER PRIMARY KEY, slot TEXT, value TEXT, trust REAL, "
        "source TEXT, timestamp TEXT, is_current INTEGER, "
        "superseded_by INTEGER, thread_id TEXT)"
    )
    con.executemany(
        "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    con.close()


@pytest.fixture
def crt_db(tmp_path, monkeypatch):
    db = tmp_path / "crt_facts.db"
    _make_db(db, [
        (1, "user.favorite_color", "orange", 0.95, "user_stated",
         "2026-04-13T08:00:00", 1, None, "t1"),
        (2, "user.favorite_color", "cyan",   0.95, "user_stated",
         "2026-04-13T09:00:00", 1, None, "t1"),
        (3, "user.location",       "Seattle", 0.9, "user_stated",
         "2026-04-13T10:00:00", 1, None, "t1"),
        # Stale fact — should not be ingested.
        (4, "user.location",       "Boise",   0.9, "user_stated",
         "2026-04-12T10:00:00", 0, 3,    "t1"),
        # Lab thread — excluded by default.
        (5, "user.favorite_color", "lab_red", 0.95, "user_stated",
         "2026-04-13T08:30:00", 1, None, "lab_x"),
    ])
    monkeypatch.setenv("AETHER_CRT_FACTS_DB", str(db))
    return db


def test_disabled_when_mode_unset(crt_db, monkeypatch):
    monkeypatch.delenv("AETHER_CRT_INTEGRATION", raising=False)
    sub = SubstrateGraph()
    out = crt.sync_to_substrate(sub)
    assert out["status"] == "disabled"
    assert out["imported"] == 0
    assert sub.slots == {}


def test_sync_ingests_current_facts(crt_db, monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "write")
    sub = SubstrateGraph()
    out = crt.sync_to_substrate(sub)
    assert out["status"] == "ok"
    # 3 current facts: ids 1,2,3 (lab and is_current=0 excluded)
    assert out["scanned"] == 3
    assert out["imported"] == 3
    assert "user:favorite_color" in sub.slots
    assert "user:location" in sub.slots
    # Last-written favorite_color wins (id 2 = cyan, ordered by ts asc)
    assert sub.current_state("user", "favorite_color").value == "cyan"
    # And the prior (orange) is superseded.
    hist = sub.history("user", "favorite_color")
    assert hist[0].value == "orange"
    assert hist[0].superseded_by == hist[1].state_id


def test_sync_is_idempotent(crt_db, monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "write")
    sub = SubstrateGraph()
    a = crt.sync_to_substrate(sub)
    b = crt.sync_to_substrate(sub)
    assert a["imported"] == 3
    assert b["imported"] == 0
    assert b["skipped"] == 3
    # No duplicate states added on second pass.
    assert len(sub.history("user", "favorite_color")) == 2
    assert len(sub.history("user", "location")) == 1


def test_lab_threads_included_when_flag_set(crt_db, monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "write")
    monkeypatch.setenv("AETHER_CRT_INCLUDE_LAB", "1")
    sub = SubstrateGraph()
    out = crt.sync_to_substrate(sub)
    # Now lab fact (id 5) is in scope: 4 current facts.
    assert out["scanned"] == 4
    assert out["imported"] == 4


def test_max_facts_caps_imports(crt_db, monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "write")
    sub = SubstrateGraph()
    out = crt.sync_to_substrate(sub, max_facts=2)
    assert out["imported"] == 2
    # Resuming a capped sync picks up where it left off.
    out2 = crt.sync_to_substrate(sub)
    assert out2["imported"] == 1
    assert out2["skipped"] == 2


def test_dry_run_observes_nothing(crt_db, monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "write")
    sub = SubstrateGraph()
    out = crt.sync_to_substrate(sub, dry_run=True)
    assert out["status"] == "ok"
    assert out["dry_run"] is True
    assert out["imported"] == 3
    assert sub.slots == {}


def test_slot_without_dot_falls_under_crt_namespace(tmp_path, monkeypatch):
    db = tmp_path / "crt_facts.db"
    _make_db(db, [
        (1, "bare_slot", "v", 0.9, "user_stated",
         "2026-04-13T08:00:00", 1, None, None),
    ])
    monkeypatch.setenv("AETHER_CRT_FACTS_DB", str(db))
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "write")
    sub = SubstrateGraph()
    crt.sync_to_substrate(sub)
    assert "crt:bare_slot" in sub.slots


def test_full_mode_enables_write(crt_db, monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "full")
    sub = SubstrateGraph()
    out = crt.sync_to_substrate(sub)
    assert out["status"] == "ok"
    assert out["imported"] == 3


def test_no_db_returns_clean_status(monkeypatch, tmp_path):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "write")
    monkeypatch.setenv("AETHER_CRT_FACTS_DB", str(tmp_path / "missing.db"))
    sub = SubstrateGraph()
    out = crt.sync_to_substrate(sub)
    assert out["status"] == "no_db"
    assert out["imported"] == 0
