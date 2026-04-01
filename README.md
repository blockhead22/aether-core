# CRT — Contradiction-aware Reconciliation and Trust

Epistemic governance framework for AI systems. Trust-weighted memory, contradiction detection, belief/speech separation, and immune agent governance.

> "I'd rather exist in an ecosystem where the baseline expectation is 'show your uncertainty' than one where every assistant performs confidence it hasn't earned."

## What this is

CRT is middleware that sits between your LLM and your application. It provides the epistemic layer that LLMs don't have: trust scores that evolve over time, contradictions that are tracked instead of silently overwritten, and governance agents that catch overconfident or inconsistent output before it reaches users.

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

### `crt.governance` — Immune agents (shipped)
Six autonomous monitors enforcing constitutional laws at runtime. Zero LLM calls. Pure boundary inspection.

### `crt.memory` — Trust-weighted memory (coming)
Gaussian splat representations, trust evolution (decay/reinforcement/correction), belief dependency graphs.

### `crt.contradiction` — Contradiction lifecycle (coming)
Active/Settling/Settled/Archived lifecycle, predictive detection via splat trajectory convergence, disposition classification.

### `crt.epistemics` — Belief/speech separation (coming)
Auditable gap between what the system believes and what it says. Belief classification (fact vs position).

## Design philosophy

- **Contradictions are signals, not bugs.** Some contradictions should be held indefinitely.
- **Trust is earned, not assigned.** Memory trust evolves through reinforcement, contradiction, and time decay.
- **The belief/speech gap should be logged, not hidden.** Transparency over prevention.
- **The mouth must never outweigh the self.** Generated output cannot upgrade its own authority.

## Origin

CRT was built as the epistemic backbone of [Aether](https://aeteros.com), a personal AI assistant. It has been validated against production conversation data (600+ memory nodes, real contradiction cascades, multi-model governance testing across Qwen3, Mistral, DeepSeek, and GPT-4o).

The governance agents emerged from empirical findings:
- **Template collapse**: RLHF training collapses moral-domain responses into hedge templates with near-zero variance (Law 2)
- **Domain inversion**: Qwen3 and Mistral show *inverted* confidence patterns — high variance where they should be certain, low where they should hedge (Law 5)
- **Speech-to-belief leakage**: Generated summaries, if stored at the same trust as user input, corrupt the memory over time (Law 1)

## License

MIT

## Author

Nick Block / [Aeteros](https://aeteros.com)
