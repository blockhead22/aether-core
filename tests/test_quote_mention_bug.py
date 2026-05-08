"""Repro for the quote/mention bug + log-timestamp/similarity-score false positives
surfaced by the 2026-05-01 Mac eval against a 130-turn ChatGPT thread.

Two bug classes:
  A. Quote/mention confusion — a literal quote of a fact gets extracted as if
     it were an active assertion. e.g. >>> "I work at Microsoft" was the example
     should NOT extract employer=Microsoft.
  B. Quantitative detector firing on log artifacts — a string like
     "2026-05-01T18:42:30" or "score=0.93" should NOT be parsed as a
     quantitative fact slot or trigger numeric contradiction shape primitives.
"""
from __future__ import annotations

import pytest

from aether.memory.slots import extract_fact_slots


# --- Bug A: quote/mention confusion ---

@pytest.mark.parametrize("text", [
    'The example fact was "I work at Google".',
    "He said \"my name is Alice\" in the corpus.",
    "Consider the claim: 'I live in Seattle'.",
    "The string “I'm a doctor” appears in the log.",
])
def test_quoted_mention_does_not_extract_employer_or_name(text):
    """Quoted mentions of facts should NOT populate slots — they're not claims."""
    facts = extract_fact_slots(text)
    # None of the quoted slots should be extracted as active facts.
    assert "employer" not in facts, f"Quoted employer leaked: {facts.get('employer')}"
    assert "name" not in facts, f"Quoted name leaked: {facts.get('name')}"
    assert "location" not in facts, f"Quoted location leaked: {facts.get('location')}"
    assert "title" not in facts, f"Quoted title leaked: {facts.get('title')}"


def test_unquoted_assertion_still_extracts():
    """Sanity: when not quoted, the extractor should still work."""
    facts = extract_fact_slots("My name is Alice and I work at Google.")
    # At least one of these should come through; the test guards against the
    # over-correction where the quote-guard kills all extraction.
    assert ("name" in facts) or ("employer" in facts), \
        f"Quote-guard over-suppressed: {facts}"


# --- Bug B: log timestamps and similarity scores ---

@pytest.mark.parametrize("text", [
    "2026-05-01T18:42:30 - INFO - retrieval scored 0.93",
    "[score=0.876] memory matched at depth 3",
    "Latency: 142.5 ms (p99=380.2 ms)",
    "INFO 2026-04-30 09:12:55 mcp.server: cosine_sim=0.512",
])
def test_log_lines_do_not_extract_quantitative_facts(text):
    """Log artifacts must not trigger fact-slot extraction."""
    facts = extract_fact_slots(text)
    # No personal-profile slot should pick up log artifacts.
    leaky = {k: v for k, v in facts.items() if k in {
        "age", "phone", "year", "salary", "name", "employer", "location"}}
    assert not leaky, f"Log line produced fact slots: {leaky}"
