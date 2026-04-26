"""Wrap an LLM call with Aether -- full integration shape.

Shows the three touchpoints from the README:
    BEFORE the LLM call: extract facts, check tension between memories
    YOUR LLM CALL: unchanged
    AFTER the LLM call: govern the response against your belief state

This example fakes the LLM (just returns a string) so it runs offline.
The integration shape is identical with a real LLM.

Run:
    pip install aether-core
    python examples/02_full_pipeline.py
"""

from aether.governance import GovernanceLayer
from aether.contradiction import StructuralTensionMeter
from aether.memory import extract_fact_slots


def fake_llm(prompt: str) -> tuple[str, float]:
    """Stand-in for a real LLM call.

    Returns (response_text, internal_belief_confidence).
    A real LLM does not surface belief_confidence directly; you derive it
    from your memory layer (trust scores of supporting facts), retrieval
    similarity, or a separate confidence head.
    """
    return (
        "Based on what you've told me, you definitely live in Seattle "
        "and absolutely work at Microsoft.",
        0.3,  # low because the supporting memory is contradicted (see below)
    )


def banner(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    gov = GovernanceLayer()
    meter = StructuralTensionMeter()

    # ------------------------------------------------------------------
    # Stored memory from prior turns
    stored = "I live in Seattle and work at Microsoft"

    # New incoming user message that contradicts the stored memory
    user_message = "I live in Portland now and work for myself"

    # ------------------------------------------------------------------
    # BEFORE the LLM: extract facts + measure tension
    banner("STEP 1 -- Extract facts from the new message")
    new_facts = extract_fact_slots(user_message)
    if not new_facts:
        print("  (no facts extracted)")
    for slot, fact in new_facts.items():
        print(f"  {slot}: {fact.value!r}  (normalized: {fact.normalized!r})")

    banner("STEP 2 -- Measure tension against stored memory")
    tension = meter.measure(stored, user_message, trust_a=0.8, trust_b=0.7)
    print(f"  Tension score:    {tension.tension_score:.2f}")
    print(f"  Relationship:     {tension.relationship.name}")
    print(f"  Suggested action: {tension.action.name}")

    # ------------------------------------------------------------------
    # YOUR LLM CALL (unchanged)
    banner("STEP 3 -- Call the LLM (unchanged from your existing code)")
    response, internal_confidence = fake_llm(user_message)
    print(f"  LLM response:        {response!r}")
    print(f"  Internal confidence: {internal_confidence}")

    # ------------------------------------------------------------------
    # AFTER the LLM: govern the response
    banner("STEP 4 -- Govern the response")
    verdict = gov.govern_response(response, belief_confidence=internal_confidence)
    print(f"  Tier:           {verdict.tier.value}")
    print(f"  Block to user?  {verdict.should_block}")
    print(f"  Adjust by:      {verdict.confidence_adjustment:+.2f}")
    print()
    print("  Findings:")
    for ann in verdict.annotations:
        print(f"    [{ann.severity}] {ann.law}")
        print(f"      {ann.finding[:100]}")

    # ------------------------------------------------------------------
    # What you do with the verdict is product-specific
    if verdict.should_block:
        final = (
            "I'm not confident enough to answer that -- your latest message "
            "contradicts what I had on file. Can you confirm Portland and freelance?"
        )
    elif verdict.tier.value in ("hedge", "flag"):
        final = response + "  (-- but please double-check; this conflicts with prior context.)"
    else:
        final = response

    banner("STEP 5 -- Final response delivered to user")
    print(f"  {final}")


if __name__ == "__main__":
    main()
