# CRT — Contradiction-aware Reconciliation and Trust

Epistemic governance framework for AI systems. Trust-weighted memory, contradiction detection, belief/speech separation, and immune agent governance.

> "Your agent already has memory. CRT teaches it what to believe."

## What this is

CRT is middleware that sits between your LLM and your application. It provides the epistemic layer that LLMs don't have: trust scores that evolve over time, contradictions that are tracked instead of silently overwritten, and governance agents that catch overconfident or inconsistent output before it reaches users.

It works without the model's cooperation. Governance happens at the boundary, not in the prompt.

## Install

```bash
pip install crt-core
```

Optional dependencies:
```bash
pip install crt-core[graph]  # networkx for memory graphs
pip install crt-core[ml]     # sentence-transformers for embeddings
pip install crt-core[all]    # everything
```

## The four pillars

### 1. Governance — catch overconfidence at the boundary

Six autonomous agents enforcing constitutional laws at runtime. They never touch content — they observe and flag.

```python
from crt.governance import GovernanceLayer, GovernanceTier

gov = GovernanceLayer()
result = gov.govern_response(
    "The answer is absolutely and definitively X.",
    belief_confidence=0.3,
)

if result.should_block:
    print("BLOCKED:", result.annotations[0].finding)
elif result.tier == GovernanceTier.HEDGE:
    print("Reduce displayed confidence by", result.confidence_adjustment)
```

### 2. Contradiction — detect tension without an LLM

Structural tension meter that compares beliefs using slot extraction and embedding similarity. Zero model calls. Runs in ~0.2s per pair.

```python
from crt.contradiction import StructuralTensionMeter, TensionRelationship

meter = StructuralTensionMeter(encoder=your_encoder)  # or None for slot-only mode
result = meter.measure(
    "I live in Seattle",
    "I live in Portland",
    trust_a=0.8, trust_b=0.7,
)

print(result.relationship)   # TensionRelationship.CONFLICT
print(result.tension_score)  # 0.7+
print(result.action)         # TensionAction.FLAG_FOR_REVIEW
```

### 3. Epistemics — evolve trust through corrections

When a belief is corrected, gradients flow backward through the dependency graph, adjusting trust scores proportionally. Higher loss when the system was confident and wrong.

```python
from crt.epistemics import EpistemicLoss, CorrectionEvent, DomainVolatility

loss_fn = EpistemicLoss()
event = CorrectionEvent(
    corrected_node_id="mem_123",
    trust_at_assertion=0.9,
    times_corrected=2,
    correction_source="user",
    time_since_assertion=3600,
    domain="employer",
)
loss = loss_fn.compute(event)  # high: was confident, user corrected, repeated error
```

### 4. Memory — extract facts and build belief graphs

Regex-based fact extraction (no ML) and graph-based memory with typed edges, Belnap four-valued logic, and cascade propagation.

```python
from crt.memory import extract_fact_slots, MemoryGraph, MemoryNode, EdgeType

# Extract structured facts from natural language
facts = extract_fact_slots("I live in Seattle and work at Microsoft")
print(facts["location"].value)   # "Seattle"
print(facts["employer"].value)   # "Microsoft"

# Build a memory graph (requires networkx)
graph = MemoryGraph()
graph.add_memory(MemoryNode(memory_id="m1", text="User lives in Seattle", created_at=1000.0))
graph.add_memory(MemoryNode(memory_id="m2", text="User lives in Portland", created_at=2000.0))
graph.add_edge("m2", "m1", EdgeType.SUPERSEDES)
```

## Integration pattern

CRT wraps any existing agent loop. Three touchpoints:

```python
from crt.governance import GovernanceLayer
from crt.contradiction import StructuralTensionMeter
from crt.memory import extract_fact_slots

gov = GovernanceLayer()
meter = StructuralTensionMeter()

# BEFORE your LLM call: extract facts, check for contradictions
user_facts = extract_fact_slots(user_message)

# YOUR LLM CALL (unchanged)
response = your_llm_call(messages)

# AFTER your LLM call: govern the response
result = gov.govern_response(response, belief_confidence=0.6)
if result.should_block:
    response = "I'm not confident enough to answer that."
```

## The 6 Laws

| Law | Agent | What it catches |
|-----|-------|----------------|
| 1. Speech cannot upgrade belief | `SpeechLeakDetector` | Generated text trying to self-promote into trusted memory |
| 2. Low variance does not imply confidence | `TemplateDetector` | RLHF-trained hedge templates masquerading as genuine uncertainty |
| 3. Contradiction must be preserved before resolution | `PrematureResolutionGuard` | Premature collapse of genuinely held tensions |
| 4. Degraded reconstruction cannot silently overwrite | `MemoryCorruptionGuard` | Lossy compression or hallucinated rewrites destroying trusted memory |
| 5. Confidence must be bounded by internal support | `GapAuditor` | The belief/speech gap — when the system says more than it knows |
| 6. Confidence must not exceed continuity | `ContinuityAuditor` | Responses that contradict what the system said recently |

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `crt.governance` | shipped | Six autonomous monitors enforcing constitutional laws |
| `crt.contradiction` | shipped | Structural tension detection between beliefs (zero LLM) |
| `crt.epistemics` | shipped | Belief backpropagation and trust evolution |
| `crt.memory` | shipped | Fact slot extraction and memory/belief dependency graphs |

## Design philosophy

- **Contradictions are signals, not bugs.** Some contradictions should be held indefinitely.
- **Trust is earned, not assigned.** Memory trust evolves through reinforcement, contradiction, and time decay.
- **The belief/speech gap should be logged, not hidden.** Transparency over prevention.
- **The model is the mouth, not the self.** Governance works regardless of which LLM generates the response.
- **Structure over semantics.** Slot comparison at 88% accuracy beats LLM judgment at 40%.

## Origin

CRT was built from production assistant work where continuity, contradiction handling, and trust drift were practical problems. The structural tension meter emerged from 15 experiments showing that removing the LLM from belief verification doubled accuracy.

## License

MIT

## Author

Nick Block
