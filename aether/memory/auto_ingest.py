"""Auto-ingest: extract high-signal facts from a conversation turn.

Most users won't manually call `aether_remember` mid-flow. The
substrate stays empty, the value disappears. This module bridges the
gap with a conservative heuristic extractor — it scans a turn (user
message and/or assistant response) and returns a list of candidate
facts to remember.

Conservative on purpose. Noisy substrate is worse than empty
substrate. Default rules only fire on strong signals: explicit
preferences, identity statements, project-config declarations,
corrections, and constraint statements.

Usage:
    from aether.memory.auto_ingest import extract_facts

    facts = extract_facts(
        user_message="we use pnpm in this repo, not npm",
        assistant_response=None,
    )
    for f in facts:
        # f.text, f.trust, f.source, f.signal
        store.add_memory(text=f.text, trust=f.trust, source=f.source)

This module is pure stdlib (regex). No network, no LLM calls. Designed
to live inside a Claude Code Stop hook or any equivalent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CandidateFact:
    """A fact pulled out of a conversation turn."""
    text: str
    trust: float
    source: str         # "user_preference" | "user_correction" | etc.
    signal: str         # Which rule fired
    raw_match: str      # Original snippet the rule matched


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

# Each rule is (label, source, trust, regex with one capture group).
# The capture group becomes the fact text; the rule label is the
# `signal` attribute on the candidate.

# All patterns are case-insensitive. They match the *start* of the
# carrying clause to avoid pulling fragments out of the middle of a
# sentence.

USER_RULES: List[tuple] = [
    # Preferences
    (
        "user_preference",
        "user_preference",
        0.85,
        re.compile(
            r"\b(?:i|we) (?:prefer|love|hate|dislike|enjoy|always use|usually use) "
            r"([^.,;\n!?]{3,160})",
            re.I,
        ),
    ),
    # Identity / role
    (
        "user_identity",
        "user_identity",
        0.9,
        re.compile(
            r"\b(?:i am|i'm|i work as|i work at|i live in|i'm based in|"
            r"my name is|my role is|my company is) "
            r"([^.,;\n!?]{2,120})",
            re.I,
        ),
    ),
    # Project / repo configuration
    (
        "project_fact",
        "user_project_fact",
        0.85,
        re.compile(
            r"\b(?:this (?:repo|codebase|project|service|app) "
            r"|the (?:repo|codebase|project|service|app) "
            r"|we (?:use|are using|run|run on|deploy to|deploy with"
            r"|host on|host with|build with) )"
            r"([^.,;\n!?]{2,160})",
            re.I,
        ),
    ),
    # Decisions
    (
        "decision",
        "user_decision",
        0.8,
        re.compile(
            r"\b(?:we (?:decided|agreed|chose|picked|will use|will go with) "
            r"|let's (?:go with|use|pick) "
            r"|the (?:plan|decision) is to )"
            r"([^.,;\n!?]{3,160})",
            re.I,
        ),
    ),
    # Constraints / prohibitions (high trust because explicit)
    (
        "constraint",
        "user_constraint",
        0.92,
        re.compile(
            r"\b(?:never|don't|do not|must not|should not|"
            r"please don't|please do not|you should never) "
            r"([^.,;\n!?]{3,160})",
            re.I,
        ),
    ),
    # Corrections — distinguished by "actually", "wait", "no, "
    (
        "correction",
        "user_correction",
        0.93,
        re.compile(
            r"\b(?:actually|wait,? (?:no|that's wrong)|no,? (?:it'?s|that'?s|we'?re)"
            r"|that'?s wrong|that's not right|to be clear) "
            r"([^.,;\n!?]{3,200})",
            re.I,
        ),
    ),
]


ASSISTANT_RULES: List[tuple] = [
    # Confirmed observations from the assistant get LOWER trust.
    # E.g. "I see that this codebase uses pnpm" -> trust 0.6
    # because the assistant might be wrong.
    (
        "observation",
        "assistant_observation",
        0.6,
        re.compile(
            r"\b(?:i see (?:that|the)|it looks like|i notice (?:that|the)|"
            r"based on (?:the code|the file|the repo)|"
            r"the (?:repo|codebase|project) (?:uses|appears to use|seems to use)) "
            r"([^.,;\n!?]{3,160})",
            re.I,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

# Reject fragments that look like questions, fragments, or boilerplate.
_REJECT_PATTERNS = [
    re.compile(r"^[\s]*$"),
    re.compile(r"^(thanks|thank you|please|sure|ok|okay|yeah|yes|no|maybe)$", re.I),
    re.compile(r"\?\s*$"),  # ends with question mark
    re.compile(r"^\s*(a|an|the|some|any|all|every)\s*$", re.I),
]


def _looks_like_garbage(text: str) -> bool:
    text = text.strip()
    if len(text) < 3:
        return True
    for p in _REJECT_PATTERNS:
        if p.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_facts(
    user_message: Optional[str] = None,
    assistant_response: Optional[str] = None,
    max_facts: int = 8,
) -> List[CandidateFact]:
    """Pull high-signal facts out of a turn.

    Args:
        user_message: The user's last message in the turn. Optional.
        assistant_response: The assistant's response, if available.
                            Lower-trust rules apply here.
        max_facts: Cap on returned candidates per call. Prevents a
                   verbose turn from flooding the substrate.

    Returns:
        List of `CandidateFact`. Empty list if nothing matched. The
        caller is responsible for calling `aether_remember` on each.
    """
    found: List[CandidateFact] = []

    if user_message:
        for label, source, trust, pat in USER_RULES:
            for m in pat.finditer(user_message):
                snippet = (m.group(1) or "").strip(" ,.;:")
                if _looks_like_garbage(snippet):
                    continue
                # Reconstruct a normalized fact text. For corrections
                # and constraints we keep the prefix so the meaning
                # carries. For preferences we keep "I prefer X" form.
                rebuilt = _rebuild_for_signal(label, snippet)
                found.append(CandidateFact(
                    text=rebuilt,
                    trust=trust,
                    source=source,
                    signal=label,
                    raw_match=m.group(0),
                ))

    if assistant_response:
        for label, source, trust, pat in ASSISTANT_RULES:
            for m in pat.finditer(assistant_response):
                snippet = (m.group(1) or "").strip(" ,.;:")
                if _looks_like_garbage(snippet):
                    continue
                rebuilt = _rebuild_for_signal(label, snippet)
                found.append(CandidateFact(
                    text=rebuilt,
                    trust=trust,
                    source=source,
                    signal=label,
                    raw_match=m.group(0),
                ))

    # Deduplicate within the call (same text, keep highest trust)
    by_text: dict[str, CandidateFact] = {}
    for f in found:
        key = f.text.lower().strip()
        existing = by_text.get(key)
        if existing is None or f.trust > existing.trust:
            by_text[key] = f

    deduped = list(by_text.values())
    deduped.sort(key=lambda c: -c.trust)
    return deduped[:max_facts]


def _rebuild_for_signal(label: str, snippet: str) -> str:
    """Turn a captured fragment into a self-standing fact text."""
    snippet = snippet.strip()
    if label == "user_preference":
        return f"User preference: {snippet}"
    if label == "user_identity":
        return f"User: {snippet}"
    if label == "project_fact":
        return f"This project: {snippet}"
    if label == "decision":
        return f"Decision: {snippet}"
    if label == "constraint":
        return f"Constraint — never/avoid: {snippet}"
    if label == "correction":
        return f"Correction: {snippet}"
    if label == "observation":
        return f"Observed: {snippet}"
    return snippet


def ingest_turn(
    store,
    user_message: Optional[str] = None,
    assistant_response: Optional[str] = None,
    max_facts: int = 8,
    dedup_against_substrate: bool = True,
) -> List[dict]:
    """End-to-end: extract candidates, write each via the store.

    Returns the list of writes performed (each dict matches the
    `store.add_memory` return shape).
    """
    candidates = extract_facts(
        user_message=user_message,
        assistant_response=assistant_response,
        max_facts=max_facts,
    )
    writes: List[dict] = []
    for c in candidates:
        if dedup_against_substrate:
            # Skip if a near-duplicate already exists at high trust
            existing = store.search(c.text, limit=3)
            if existing and existing[0].get("similarity") is not None:
                if existing[0]["similarity"] > 0.92 and existing[0]["trust"] >= c.trust:
                    continue
            elif existing and existing[0]["score"] > 1.0:
                # Substring fallback: very high overlap means duplicate
                continue

        result = store.add_memory(
            text=c.text,
            trust=c.trust,
            source=c.source,
        )
        result["signal"] = c.signal
        writes.append(result)
    return writes
