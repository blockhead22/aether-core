# Aether — A Belief Substrate for AI Systems

> The model is the mouth. The substrate is the self.

Aether is a runtime belief substrate: persistent, contradiction-aware trust state that survives across model swaps, vendor changes, and session boundaries. It sits between your application and any LLM and provides the epistemic layer LLMs don't have on their own — trust scores that evolve under correction, contradictions tracked rather than silently overwritten, cascade pressure measured as a first-class signal, and governance gates that act on belief state instead of output strings.

## Why a belief substrate, not just a memory layer

Memory layers (Mem0, Letta, Zep, Cognee, LinkedIn CMA) store *what was said*. A belief substrate tracks *what is believed, how trust evolved, and which contradictions remain unresolved on purpose*. The two operate at different abstraction layers and complement each other — Aether can run on top of any storage tier.

The case for it is no longer theoretical:

- **Anthropic, April 2026** — [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html). 171 emotion concept vectors with **internal-external decoupling**: internal state often does not surface in text. Pushing the desperation vector +0.05 raises blackmail rate from 22% → 72%; reward hacking 5% → 70%. Pushing calm to +0.05 drops blackmail to 0%. These representations are real, causal, and invisible to output-based monitoring.
- **Science, April 2026, N=1,604** — One conversation with a frontier LLM made participants **50% more likely to affirm harmful behavior**. The effect is structurally invisible to text-level review. **Only 21% of enterprises deploying agentic AI have a mature governance model.**
- **Microsoft, April 2 2026** — open-sourced the [Agent Governance Toolkit](https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit-open-source-runtime-security-for-ai-agents/). Sub-millisecond *policy* enforcement against the OWASP agentic-AI risks — what the agent is *allowed* to do.

Aether answers a different question than policy-based governance: not "is the action permitted," but **"does the agent's belief state justify the action, and how confident is it?"** The two layers compose.

The framing — "internal-external decoupling" in Anthropic's words — is what this library has been calling the **belief/speech gap** since the first commit. Law 5 of the governance layer (`GapAuditor`) measures it directly.

## Install

```bash
pip install aether-core
```

Optional dependencies:

```bash
pip install aether-core[graph]   # networkx for memory and dependency graphs
pip install aether-core[ml]      # sentence-transformers for embeddings
pip install aether-core[all]     # everything
```

## The four pillars

### 1. Governance — catch overconfidence at the boundary

Six autonomous immune agents enforcing constitutional laws at runtime. They never touch content; they observe and flag.

```python
from aether.governance import GovernanceLayer, GovernanceTier

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

Structural tension meter that compares beliefs using slot extraction and embedding similarity. Zero model calls. ~0.2s per pair. Contradictions are signals, not bugs — some are meant to be **held**, not resolved.

```python
from aether.contradiction import StructuralTensionMeter, TensionRelationship

meter = StructuralTensionMeter(encoder=your_encoder)
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
from aether.epistemics import EpistemicLoss, CorrectionEvent, DomainVolatility

loss_fn = EpistemicLoss()
event = CorrectionEvent(
    corrected_node_id="mem_123",
    trust_at_assertion=0.9,
    times_corrected=2,
    correction_source="user",
    time_since_assertion=3600,
    domain="employer",
)
loss = loss_fn.compute(event)
```

### 4. Memory — extract facts and build belief graphs

Regex-based fact extraction (no ML) and graph-based memory with typed edges, Belnap four-valued logic, and **cascade propagation with measurable pressure**.

```python
from aether.memory import extract_fact_slots, MemoryGraph, MemoryNode, EdgeType
from aether.memory import BeliefDependencyGraph

facts = extract_fact_slots("I live in Seattle and work at Microsoft")
print(facts["location"].value)   # "Seattle"
print(facts["employer"].value)   # "Microsoft"

bdg = BeliefDependencyGraph()
# ... add beliefs and dependencies ...
result = bdg.propagate_cascade(corrected_node_id, delta_0=1.0)
print(result.max_pressure, result.avg_pressure)  # cascade pressure as a signal
```

## Integration pattern

Aether wraps any existing agent loop. Three touchpoints:

```python
from aether.governance import GovernanceLayer
from aether.contradiction import StructuralTensionMeter
from aether.memory import extract_fact_slots

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
| 2. Low variance does not imply confidence | `TemplateDetector` | RLHF hedge templates masquerading as genuine uncertainty |
| 3. Contradiction must be preserved before resolution | `PrematureResolutionGuard` | Premature collapse of genuinely held tensions |
| 4. Degraded reconstruction cannot silently overwrite | `MemoryCorruptionGuard` | Lossy compression or hallucinated rewrites destroying trusted memory |
| 5. Confidence must be bounded by internal support | `GapAuditor` | The belief/speech gap — saying more than you know (= internal-external decoupling) |
| 6. Confidence must not exceed continuity | `ContinuityAuditor` | Responses that contradict what the system said recently |

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `aether.governance` | shipped | Six immune agents enforcing constitutional laws |
| `aether.contradiction` | shipped | Structural tension detection between beliefs (zero LLM) |
| `aether.epistemics` | shipped | Belief backpropagation and trust evolution |
| `aether.memory` | shipped | Fact slot extraction, memory graphs, BDG with cascade pressure |
| `aether.mcp` | planned | MCP server exposing `aether_sanction`, `aether_lineage`, `aether_cascade_preview`, `aether_fidelity`, `aether_done_check` |
| `aether.adapters` | planned | Cross-vendor adapters (Anthropic, OpenAI, Ollama) — substrate portability |

## Design philosophy

- **Contradictions are signals, not bugs.** Some contradictions should be held indefinitely.
- **Trust is earned, not assigned.** Memory trust evolves through reinforcement, contradiction, and time decay.
- **The belief/speech gap should be logged, not hidden.** Transparency over prevention.
- **The model is the mouth, not the self.** Governance works regardless of which LLM generates the response.
- **Structure over semantics.** Slot comparison at 88% accuracy beats LLM judgment at 40%.
- **Cascade pressure is measurable.** Belief revisions propagate through dependency graphs with bounded depth and damping. The math is the moat.

## Where this fits next to other tools

| | Storage scope | Tracks contradiction | Belief/speech gap | Cross-vendor portable | Cascade pressure |
|---|---|---|---|---|---|
| Mem0, Letta, Zep, Cognee | memory layer | as overwrite | no | partial | no |
| Microsoft Agent Governance Toolkit | runtime policy | no | no | yes | no |
| Anthropic / OpenAI memory features | per-vendor | no | no | no | no |
| **Aether** | **belief substrate** | **first-class state (held/settling/settled)** | **measured (Law 5)** | **yes** | **yes (BDG, theorems)** |

## Origin

Aether grew out of production assistant work where continuity, contradiction handling, and trust drift were practical problems, not philosophy. The structural tension meter emerged from 15 experiments showing that removing the LLM from belief verification doubled accuracy. The cascade complexity result and belief-backpropagation engine come from the same line of work.

Earlier the architecture was named **CRT** (Contradiction-aware Reconciliation and Trust). The substrate is now branded **Aether**, which is what it has always actually been.

## License

MIT

## Author

Nick Block — [@blockhead22](https://github.com/blockhead22)
