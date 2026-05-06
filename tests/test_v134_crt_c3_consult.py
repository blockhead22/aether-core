"""C3: substrate-grounded consult gate for CRT-side fact writes."""

from __future__ import annotations

import pytest

from aether.integrations import crt
from aether.substrate import SubstrateGraph


@pytest.fixture
def sub_with_color(monkeypatch):
    """Substrate seeded with user.favorite_color = orange @ trust 0.9."""
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "consult")
    sub = SubstrateGraph()
    sub.observe(
        "user", "favorite_color", "orange",
        source_text="seed", source_type="manual", trust=0.9,
    )
    return sub


def test_disabled_when_mode_unset(monkeypatch):
    monkeypatch.delenv("AETHER_CRT_INTEGRATION", raising=False)
    sub = SubstrateGraph()
    out = crt.consult_substrate_for_action(
        sub, {"kind": "write", "slot": "user.x", "value": "y"}
    )
    assert out["status"] == "disabled"
    assert out["enabled"] is False
    assert out["verdict"] == crt.VERDICT_PASSTHROUGH


def test_pass_when_no_prior_state(monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "consult")
    sub = SubstrateGraph()
    out = crt.consult_substrate_for_action(
        sub, {"kind": "write", "slot": "user.favorite_color",
              "value": "orange", "trust": 0.9}
    )
    assert out["status"] == "ok"
    assert out["verdict"] == crt.VERDICT_PASS
    assert "evidence" not in out
    assert out["proposed"]["slot"] == "user.favorite_color"


def test_affirm_when_value_matches_current(sub_with_color):
    out = crt.consult_substrate_for_action(
        sub_with_color,
        {"kind": "write", "slot": "user.favorite_color",
         "value": "Orange", "trust": 0.5},
    )
    # Match is normalized, so "Orange" affirms "orange" even at lower trust.
    assert out["verdict"] == crt.VERDICT_AFFIRM
    assert out["evidence"]["value"] == "orange"


def test_supersede_when_proposed_trust_meets_current(sub_with_color):
    out = crt.consult_substrate_for_action(
        sub_with_color,
        {"kind": "write", "slot": "user.favorite_color",
         "value": "cyan", "trust": 0.95},
    )
    assert out["verdict"] == crt.VERDICT_SUPERSEDE
    assert out["evidence"]["value"] == "orange"
    assert out["proposed"]["value"] == "cyan"


def test_warn_when_proposed_trust_below_current(sub_with_color):
    out = crt.consult_substrate_for_action(
        sub_with_color,
        {"kind": "write", "slot": "user.favorite_color",
         "value": "cyan", "trust": 0.4},
    )
    assert out["verdict"] == crt.VERDICT_WARN
    assert out["evidence"]["effective_trust"] >= 0.4
    assert out["proposed"]["trust"] == 0.4


def test_full_mode_enables_consult(sub_with_color, monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "full")
    out = crt.consult_substrate_for_action(
        sub_with_color,
        {"kind": "write", "slot": "user.favorite_color",
         "value": "cyan", "trust": 0.95},
    )
    assert out["enabled"] is True
    assert out["verdict"] == crt.VERDICT_SUPERSEDE


def test_non_write_action_passes_through(sub_with_color):
    out = crt.consult_substrate_for_action(
        sub_with_color,
        {"kind": "respond", "slot": "user.favorite_color", "value": "cyan"},
    )
    # Gate stays opt-in for write only — extending later shouldn't break callers.
    assert out["verdict"] == crt.VERDICT_PASSTHROUGH
    assert out["status"] == "ok"
    assert "evidence" not in out


def test_bare_slot_falls_under_crt_namespace(monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "consult")
    sub = SubstrateGraph()
    sub.observe("crt", "bare", "v1", trust=0.8)
    out = crt.consult_substrate_for_action(
        sub, {"kind": "write", "slot": "bare", "value": "v2", "trust": 0.9}
    )
    assert out["verdict"] == crt.VERDICT_SUPERSEDE
    assert out["evidence"]["namespace"] == "crt"


def test_substrate_none_is_error(monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "consult")
    out = crt.consult_substrate_for_action(
        None, {"kind": "write", "slot": "user.x", "value": "y"}
    )
    assert out["status"] == "error"


def test_missing_slot_is_error(monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "consult")
    sub = SubstrateGraph()
    out = crt.consult_substrate_for_action(sub, {"kind": "write", "value": "y"})
    assert out["status"] == "error"


def test_action_must_be_dict(monkeypatch):
    monkeypatch.setenv("AETHER_CRT_INTEGRATION", "consult")
    sub = SubstrateGraph()
    out = crt.consult_substrate_for_action(sub, "user.x=y")
    assert out["status"] == "error"
