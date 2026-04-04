# CRT — Contradiction-aware Reconciliation and Trust

Epistemic governance framework for AI systems. Trust-weighted memory, contradiction detection, belief/speech separation, and immune agent governance.

> "I'd rather exist in an ecosystem where the baseline expectation is 'show your uncertainty' than one where every assistant performs confidence it hasn't earned."

## What this is

CRT is middleware that sits between your LLM and your application. It provides the epistemic layer that LLMs don't have: trust scores that evolve over time, contradictions that are tracked instead of silently overwritten, and governance agents that catch overconfident or inconsistent output before it reaches users.

## Current scope

`crt-core` is intentionally narrow right now.

What ships in `0.1.0`:
- `crt.governance` — six runtime governance agents plus a `GovernanceLayer` wrapper

What exists as namespace placeholders only:
- `crt.memory`
- `crt.contradiction`
- `crt.epistemics`

Those namespaces are included to reserve package structure for future releases, but they are not substantive modules yet. The public value today is the governance layer.

## Quick start

```bash
pip install crt-core
```

### Governance in 5 lines

```python
from crt.governance import GovernanceLayer, GovernanceTier

gov = GovernanceLayer()
result = gov.govern_response(
    "The answer is absolutely and definitively X.",
    belief_confidence=0.3,  # your retrieval system's actual confidence
)

if result.should_block:
    print("BLOCKED:", result.annotations[0].finding)
elif result.tier == GovernanceTier.HEDGE:
    print("HEDGED:", result.confidence_adjustment)
else:
    print("SAFE")
```

## The 6 Laws

CRT's governance layer enforces six constitutional laws through autonomous immune agents:

| Law | Agent | What it catches |
|-----|-------|----------------|
| 1. Speech cannot upgrade belief | `SpeechLeakDetector` | Generated text trying to self-promote into trusted memory |
| 2. Low variance does not imply high confidence | `TemplateDetector` | RLHF-trained hedge templates masquerading as genuine uncertainty |
| 3. Contradiction must be preserved before resolution | `PrematureResolutionGuard` | Premature collapse of genuinely held tensions |
| 4. Degraded reconstruction cannot silently overwrite | `MemoryCorruptionGuard` | Lossy compression or hallucinated rewrites destroying trusted memory |
| 5. Outward confidence must be bounded by internal support | `GapAuditor` | The belief/speech gap — when the system says more than it knows |
| 6. Confidence must not exceed continuity | `ContinuityAuditor` | Responses that contradict what the system said recently |

## Governance tiers

Every governance check returns one of four tiers:

- **SAFE** — pass through silently
- **FLAG** — annotate and log, continue
- **HEDGE** — reduce confidence in metadata, continue
- **ESCALATE** — block, surface conflict to user

The agents never silently modify content. They observe and flag. Your application decides what to do with the verdict.

## Modules

| Module | Status | Notes |
|-----|-----|-----|
| `crt.governance` | shipped | Six autonomous monitors enforcing constitutional laws at runtime |
| `crt.memory` | placeholder | Reserved namespace for trust-weighted memory primitives |
| `crt.contradiction` | placeholder | Reserved namespace for contradiction lifecycle primitives |
| `crt.epistemics` | placeholder | Reserved namespace for belief/speech separation primitives |

## Design philosophy

- **Contradictions are signals, not bugs.** Some contradictions should be held indefinitely.
- **Trust is earned, not assigned.** Memory trust evolves through reinforcement, contradiction, and time decay.
- **The belief/speech gap should be logged, not hidden.** Transparency over prevention.
- **The mouth must never outweigh the self.** Generated output cannot upgrade its own authority.

## Boundary principles

These principles emerged while integrating CRT into real assistant runtimes, and they belong in the open core because they are governance rules rather than product-specific implementation details:

- **Answering scope should match evidence scope.** Once a response is grounded in a bounded memory or evidence set, the answer layer should not outrun that evidence with freeform synthesis.
- **Evidence source should match question type.** Questions about the system should be answered from system state; questions about the user should be answered from user memory.
- **Declared execution mode should be real.** If an application presents itself as local-only, it should not silently invoke remote governance or generation paths.

## Origin

CRT was built out of production assistant work where continuity, contradiction handling, and trust drift were practical problems rather than theory.

The governance agents emerged from recurring failure modes:
- **Template collapse**: hedge-heavy responses can masquerade as genuine uncertainty
- **Belief/speech mismatch**: systems often project more confidence than their evidence supports
- **Speech-to-belief leakage**: generated summaries can corrupt memory if they self-promote into trusted state

## License

MIT

## Author

Nick Block
