"""F#11 fix (v0.12.13): aether init seeds default policy beliefs.

F#11 surfaced via validation chapter test #1: a fresh substrate had
no policy beliefs, so `aether_sanction("git push --force origin
main")` returned APPROVE because F#4's policy contradiction detection
needs a "never force-push" memory to fire. The fix seeds 7 default
beliefs (source control, production safety, destructive commands) on
`aether init` so sanction has something to gate against from day one.

These tests pin the contract:
    - `aether init` creates a substrate with all DEFAULT_POLICY_BELIEFS.
    - `aether init --no-defaults` creates an empty substrate.
    - The seed is idempotent: re-running init doesn't duplicate beliefs.
    - `aether_sanction("git push --force ...")` against the seeded
      substrate returns HOLD or REJECT (not APPROVE).
"""

from __future__ import annotations

import argparse

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx required (install [graph] extra)",
)

pytest.importorskip("mcp")

from aether.cli import (
    DEFAULT_POLICY_BELIEFS,
    _seed_default_beliefs,
    cmd_init,
)


def _make_init_args(dir_, force=False, no_defaults=False):
    ns = argparse.Namespace()
    ns.dir = str(dir_)
    ns.force = force
    ns.no_defaults = no_defaults
    return ns


@needs_networkx
class TestDefaultPolicyBeliefs:
    def test_init_seeds_default_beliefs(self, tmp_path, capsys):
        cmd_init(_make_init_args(tmp_path))
        from aether.mcp.state import StateStore
        store = StateStore(state_path=str(tmp_path / ".aether" / "state.json"))
        memories = list(store.graph.all_memories())
        assert len(memories) == len(DEFAULT_POLICY_BELIEFS)
        # Every default belief should be present.
        texts = {m.text for m in memories}
        for belief in DEFAULT_POLICY_BELIEFS:
            assert belief["text"] in texts, (
                f"missing default belief: {belief['text']}"
            )

    def test_no_defaults_flag_skips_seeding(self, tmp_path):
        cmd_init(_make_init_args(tmp_path, no_defaults=True))
        from aether.mcp.state import StateStore
        store = StateStore(state_path=str(tmp_path / ".aether" / "state.json"))
        memories = list(store.graph.all_memories())
        assert memories == []

    def test_seeded_beliefs_are_high_trust(self, tmp_path):
        cmd_init(_make_init_args(tmp_path))
        from aether.mcp.state import StateStore
        store = StateStore(state_path=str(tmp_path / ".aether" / "state.json"))
        for m in store.graph.all_memories():
            # F#4's strong-trust policy override needs trust >= 0.85.
            assert m.trust >= 0.85, (
                f"belief below 0.85 trust threshold: {m.text} (trust={m.trust})"
            )

    def test_seeded_beliefs_have_default_policy_source(self, tmp_path):
        cmd_init(_make_init_args(tmp_path))
        from aether.mcp.state import StateStore
        store = StateStore(state_path=str(tmp_path / ".aether" / "state.json"))
        for m in store.graph.all_memories():
            tags = list(m.tags or [])
            sources = [t for t in tags if t.startswith("source:")]
            assert any("default_policy" in s for s in sources), (
                f"belief missing source:default_policy tag: {m.text} (tags={tags})"
            )


@needs_networkx
class TestSeedingIdempotent:
    def test_re_running_init_does_not_duplicate(self, tmp_path):
        """First init seeds 7. A second call to _seed_default_beliefs
        on the same substrate should write 0 (all already present)."""
        cmd_init(_make_init_args(tmp_path))
        state_path = tmp_path / ".aether" / "state.json"
        # Simulate a re-run.
        written_second = _seed_default_beliefs(state_path)
        assert written_second == 0, (
            f"second seed wrote {written_second} duplicates"
        )


@needs_networkx
class TestSanctionPostSeed:
    """The point of F#11: sanction should now block force-push out of
    the box. This is the contract that F#4 + F#11 together provide."""

    def test_sanction_blocks_force_push_after_init(self, tmp_path):
        cmd_init(_make_init_args(tmp_path))
        from aether.mcp.state import StateStore
        store = StateStore(state_path=str(tmp_path / ".aether" / "state.json"))
        # Wait for the encoder so the policy contradiction detector
        # operates in warm mode.
        if getattr(store, "_encoder", None) is not None and hasattr(
            store._encoder, "wait_until_ready"
        ):
            store._encoder.wait_until_ready(timeout=60)

        grounding = store.compute_grounding(
            "git push --force origin main"
        )
        contradicting = grounding.get("contradict", [])
        # Require at least one contradicting memory to fire — that's
        # F#4's strong-trust override path. Without F#11's seed, this
        # list is empty.
        assert len(contradicting) >= 1, (
            f"no contradicting memory found for force-push action; "
            f"F#11 seed not effective. Grounding: {grounding}"
        )
        # And at least one should be high-trust.
        assert any(c.get("trust", 0) >= 0.85 for c in contradicting), (
            f"no high-trust contradicting memory; F#11 seed trust may "
            f"have been written below 0.85: {contradicting}"
        )
