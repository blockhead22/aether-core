"""Scripted 10-turn scenario over the real MCP wire.

Per `NEXT_SESSION.md`, this file is meant to grow to 10 turns:

  1. aether_remember (seed 3 facts)
  2. aether_search verifies retrieval
  3. aether_remember a contradicting fact
  4. tension_findings surfaces it
  5. aether_sanction on a related action
  6. aether_fidelity on a draft
  7. aether_correct cascade
  8. aether_lineage walks the BDG
  9. aether_path returns weighted route
 10. aether_session_diff briefs returning agent

For now: turns 1-2 only, with the wire plumbing proven end-to-end. Each
later turn is a roughly self-contained block to add as it stabilizes.
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


# TODO(e2e): turns 3-10. See module docstring. Add each as a separate
# test (test_contradiction_surfaces, test_sanction_on_related_action,
# ...) so failures localize to a single concern.
