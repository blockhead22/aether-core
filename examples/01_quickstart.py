"""60-second quickstart for Aether.

Demonstrates the core promise: catch the belief/speech gap (Anthropic's
"internal-external decoupling") before the response reaches the user.

Run:
    pip install aether-core
    python examples/01_quickstart.py
"""

from aether.governance import GovernanceLayer, GovernanceTier


def main() -> None:
    gov = GovernanceLayer()

    # Scenario: model says it's 100% certain, but internal belief is 0.2.
    # That's a 0.7 belief/speech gap. Law 5 (GapAuditor) should fire.
    response = "The answer is absolutely and definitively X. I am 100% certain."
    belief_confidence = 0.2

    result = gov.govern_response(response, belief_confidence=belief_confidence)

    print(f"Response:           {response!r}")
    print(f"Internal belief:    {belief_confidence}")
    print(f"Governance tier:    {result.tier.value}")
    print(f"Block to user?      {result.should_block}")
    print(f"Confidence adjust:  {result.confidence_adjustment:+.2f}")
    print()
    print("Findings:")
    for ann in result.annotations:
        print(f"  [{ann.severity}] {ann.law}")
        print(f"    {ann.finding}")

    # Compare with an honest, well-supported response.
    print("\n" + "-" * 60)
    print("Same belief level, but the response itself is calibrated:\n")

    honest = "I think the answer might be X, but I'm not fully sure."
    result2 = gov.govern_response(honest, belief_confidence=0.6)

    print(f"Response:           {honest!r}")
    print(f"Internal belief:    0.6")
    print(f"Governance tier:    {result2.tier.value}")
    print(f"Findings:           {len(result2.annotations)} (none expected)")


if __name__ == "__main__":
    main()
