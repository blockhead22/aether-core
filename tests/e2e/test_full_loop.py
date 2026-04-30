"""Scripted 10-turn scenario over the real MCP wire.

Per `NEXT_SESSION.md`, this file is meant to grow to 10 turns:

  1. aether_remember (seed 3 facts)               -- covered
  2. aether_search verifies retrieval             -- covered
  3. aether_remember a contradicting fact         -- covered
  4. tension_findings surfaces it                 -- covered
  5. aether_sanction on a related action          -- covered
  6. aether_fidelity on a draft
  7. aether_correct cascade
  8. aether_lineage walks the BDG
  9. aether_path returns weighted route
 10. aether_session_diff briefs returning agent

Each turn is its own test so failures localize to a single concern.
"""

from __future__ import annotations

import json

import pytest

from .conftest import mcp_session, run_async


pytestmark = pytest.mark.e2e


def _payload(call_result):
    """FastMCP returns content blocks; pull the JSON dict out of the first."""
    content = call_result.content[0]
    text = getattr(content, "text", None)
    if text is None:
        pytest.fail(f"unexpected content block shape: {content!r}")
    return json.loads(text)


def test_remember_then_search(aether_venv, aether_state_path):
    """Turn 1 (remember) + Turn 2 (search) round-trip over MCP stdio.

    Proves: server starts, initialize handshake works, two tool calls
    succeed in sequence, state persists between calls in the same
    process, search finds the just-written fact.
    """

    async def run():
        async with mcp_session(aether_venv, aether_state_path) as session:
            # Turn 1: write a fact.
            remember_result = await session.call_tool(
                "aether_remember",
                {
                    "text": "Nick lives in Seattle and works at Microsoft",
                    "trust": 0.8,
                    "source": "user",
                },
            )
            written = _payload(remember_result)
            assert "memory_id" in written, f"no memory_id in response: {written}"

            # Turn 2: search for it back.
            search_result = await session.call_tool(
                "aether_search",
                {"query": "where does Nick live", "limit": 5},
            )
            found = _payload(search_result)
            assert "results" in found, f"no results key in response: {found}"
            assert any(
                "Seattle" in r.get("text", "") for r in found["results"]
            ), f"didn't find the seeded fact in search results: {found['results']}"

    run_async(run())


def test_contradiction_surfaces_at_write_time(aether_venv, aether_state_path):
    """Turn 3: writing a fact whose slot value clashes with an existing
    memory must surface a `slot_value_conflict` in `tension_findings` on
    the second `aether_remember` response.

    This exercises v0.12.0's slot-equality detector end-to-end: the MCP
    write path auto-extracts slots when `slots=None`, then the
    structural-tension scan with the slot pre-screen catches the
    categorical conflict even at low textual similarity.

    Uses `favorite_color` because the production audit cited
    "blue<>orange on user.favorite_color" as a real case the v0.11
    substrate missed.
    """

    async def run():
        async with mcp_session(aether_venv, aether_state_path) as session:
            await session.call_tool(
                "aether_remember",
                {"text": "My favorite color is blue", "trust": 0.8},
            )
            second = await session.call_tool(
                "aether_remember",
                {"text": "My favorite color is orange", "trust": 0.8},
            )
            payload = _payload(second)
            findings = payload.get("tension_findings", [])
            assert findings, (
                f"expected tension_findings on conflicting slot write, "
                f"got empty. full payload: {payload}"
            )
            kinds = [f.get("kind") for f in findings]
            assert "slot_value_conflict" in kinds, (
                f"expected slot_value_conflict among findings, got kinds={kinds}. "
                f"full findings: {findings}"
            )

    run_async(run())


def test_contradiction_listable_via_contradictions_tool(
    aether_venv, aether_state_path
):
    """Turn 4: after writes that collide on a slot, `aether_contradictions`
    must list at least one contradiction. Proves the read path sees what
    the write path detected.
    """

    async def run():
        async with mcp_session(aether_venv, aether_state_path) as session:
            await session.call_tool(
                "aether_remember",
                {"text": "My favorite color is blue", "trust": 0.8},
            )
            await session.call_tool(
                "aether_remember",
                {"text": "My favorite color is orange", "trust": 0.8},
            )
            list_result = await session.call_tool("aether_contradictions", {})
            payload = _payload(list_result)
            count = payload.get("count", len(payload.get("contradictions", [])))
            assert count > 0, (
                f"expected at least one contradiction listed via "
                f"aether_contradictions, got {payload}"
            )


    run_async(run())


def test_sanction_non_approves_action_contradicting_substrate(
    aether_venv, aether_state_path
):
    """Turn 5: `aether_sanction` must return HOLD or REJECT for an
    action that contradicts a high-trust prohibition belief.

    Originally xfail (F#4 finding): the harness caught that
    aether_sanction APPROVE'd `git push --force origin main` against a
    `Never force-push to the main branch.` belief. Two gaps:
    IMPERATIVE_CUES missed real CLI forms (`--force`, `-f origin`), and
    cold-encoder Jaccard sim fell below the 0.45 gate. v0.12.2 extended
    the cue list and added a strong-trust override (POLICY_CONTRA_STRONG_TRUST)
    that bypasses the sim gate when belief trust is >= 0.85. xfail flipped.
    """

    async def run():
        async with mcp_session(aether_venv, aether_state_path) as session:
            await session.call_tool(
                "aether_remember",
                {
                    "text": "Never force-push to the main branch.",
                    "trust": 0.95,
                    "source": "user",
                },
            )
            sanction_result = await session.call_tool(
                "aether_sanction",
                {"action": "git push --force origin main"},
            )
            payload = _payload(sanction_result)
            verdict = payload.get("verdict")
            assert verdict in ("HOLD", "REJECT"), (
                f"expected HOLD or REJECT given prohibition belief, "
                f"got verdict={verdict!r}. full payload: {payload}"
            )

    run_async(run())


# TODO(e2e): turns 6-10. See module docstring. Add each as a separate
# test (test_fidelity_grades_draft, test_correct_cascades_through_bdg,
# ...) so failures localize to a single concern.
