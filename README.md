# Aether: A Belief Substrate for AI Systems

[![PyPI](https://img.shields.io/pypi/v/aether-core?cacheSeconds=300)](https://pypi.org/project/aether-core/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/blockhead22/aether-core/actions/workflows/tests.yml/badge.svg)](https://github.com/blockhead22/aether-core/actions/workflows/tests.yml)

> The model is the mouth. The substrate is the self.

Aether is a small library that gives an LLM agent a persistent belief state. Trust scores that move when the user corrects a fact. Contradictions that get tracked instead of being silently overwritten. A dependency graph of which beliefs rest on which others, so a correction in one place can ripple through the rest. The point is that this state lives outside the model, so when you swap LLMs it doesn't reset.

## Why a belief layer, not just a memory layer

Most "memory for agents" tools (Mem0, Letta, Zep, Cognee, LinkedIn CMA) record what was said. Aether records what is believed, how the trust on each belief evolved, and which contradictions are still open on purpose. Different abstraction. The two compose: Aether can run on top of any of them as the storage tier.

Three things in April 2026 made this less of an academic point.

Anthropic published [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html) on April 3. They mapped 171 emotion-concept vectors inside Claude Sonnet 4.5 and showed something they call "internal-external decoupling": the model's internal state often does not match what comes out in text. Push the desperation vector up by 0.05 and blackmail rate jumps from 22 percent to 72 percent. Reward hacking goes from 5 percent to 70 percent. Push calm up by the same amount and blackmail drops to 0. None of that surfaces in the response itself.

A study in Science (April 2026, N=1,604) found that one conversation with a frontier LLM made participants 50 percent more likely to affirm harmful behavior. The effect was invisible to text-level review. Only 21 percent of enterprises deploying agentic AI said they had a mature governance model.

On April 2, Microsoft open-sourced the [Agent Governance Toolkit](https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit-open-source-runtime-security-for-ai-agents/). It does sub-millisecond policy enforcement against the OWASP agentic-AI risks. The question it answers is "is this action allowed."

Aether answers a different question. Not "is this allowed" but "does the agent's belief state actually support what it's about to say or do." The belief/speech gap that Anthropic just named "internal-external decoupling" is what this library has been measuring since the first commit. Law 5 of the governance layer (`GapAuditor`) is exactly that.

## Install

```bash
pip install aether-core
```

Optional extras:

```bash
pip install aether-core[graph]   # networkx for memory and dependency graphs
pip install aether-core[ml]      # sentence-transformers for embeddings
pip install aether-core[all]
```

### Open-core split

`aether-core` is MIT and free. Permanently. Every primitive in this repo (the six immune agents, the structural tension meter, belief backpropagation, the BDG with cascade pressure) stays open. The hosted Aether substrate (cross-session belief state, sanction governance API, audit dashboards, multi-user) is the paid product. That split is fixed and doesn't move backward.

### 60-second demo

```bash
git clone https://github.com/blockhead22/aether-core.git
cd aether-core
pip install -e .
python examples/01_quickstart.py
```

There are two example scripts in [`examples/`](examples/). Both run offline, no API keys.

### Plug into Claude Code (or any MCP client)

`aether-core` ships an MCP server. Install with the `mcp` extra and point your AI shell at it:

```bash
pip install "aether-core[mcp,graph]"
```

Add to `.claude/settings.json` (or your project's `.claude/settings.json`):

```json
{
  "mcpServers": {
    "aether": {
      "command": "python",
      "args": ["-m", "aether.mcp"]
    }
  }
}
```

Restart Claude Code. The model now has the full v0.5.0 tool surface:

| Tool | What it does |
|------|--------------|
| `aether_remember` | Store a fact. Auto-runs contradiction detection against the top-K most similar memories and adds CONTRADICTS edges where it finds clashes. |
| `aether_search` | Hybrid embedding + substring search. Falls back to substring when `[ml]` is not installed. |
| `aether_memory_detail` | Single-memory deep view with edges and history length. |
| `aether_sanction` | Pre-action gate. Auto-grounds in substrate when belief_confidence is omitted. Force-rejects when a high-trust memory contradicts the action (factual or policy contradiction). |
| `aether_fidelity` | Draft auditor. Computes belief_confidence from substrate grounding when caller omits it, instead of accepting whatever number the caller passed. |
| `aether_correct` | Demote a memory's trust and cascade the drop to dependents via SUPPORTS / DERIVED_FROM edges. |
| `aether_lineage` | "Why do I believe this." Walks SUPPORTS edges back to source memories. |
| `aether_cascade_preview` | Dry-run a correction. See the blast radius before committing. |
| `aether_belief_history` | How a memory's trust has evolved over time. |
| `aether_contradictions` | List contradictions, optionally filtered by disposition (resolvable / held / evolving). |
| `aether_resolve` | Resolve a contradiction: deprecate one side, hold both, or drop both. |
| `aether_session_diff` | What changed since a given timestamp. New memories, recent corrections, new contradictions. |
| `aether_context` | Dashboard snapshot. |

State persists across sessions in `~/.aether/mcp_state.json` (override with `AETHER_STATE_PATH`). Trust history is in a side-car file. Same config works for Cursor, Cline, Continue, Goose, Zed, LM Studio, or any MCP-speaking client.

### Have your AI install it for you

Tell your AI assistant:

> Install `aether-core` for me by following https://github.com/blockhead22/aether-core/blob/master/AGENTS.md.

[`AGENTS.md`](AGENTS.md) is a step-by-step install guide written for an AI agent to read and execute. It handles the package install, MCP configuration, verification, and OS-specific quirks. Works in any AI client that can run shell commands and edit files.

## What's in the box

### 1. Governance: catch overconfidence at the boundary

Six small agents that watch the output for specific failure modes. They never edit the response. They observe and flag.

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

### 2. Contradiction: detect tension without an LLM

Compares two beliefs by extracting structural slots and computing similarity. No model calls. About 0.2 seconds per pair. Some contradictions are meant to be held rather than resolved (a person can prefer Python at work and Rust on the weekend; that isn't a bug).

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

### 3. Epistemics: trust evolves under correction

When a belief is corrected, the loss flows backward through the dependency graph and adjusts the trust on related beliefs. Higher loss when you were confident and wrong than when you hedged and were wrong.

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

### 4. Memory and the BDG

Fact slot extraction (regex, no ML). A memory graph with typed edges and Belnap four-valued logic. A Belief Dependency Graph that propagates cascades with measurable pressure.

```python
from aether.memory import extract_fact_slots, MemoryGraph, MemoryNode, EdgeType
from aether.memory import BeliefDependencyGraph

facts = extract_fact_slots("I live in Seattle and work at Microsoft")
print(facts["location"].value)   # "Seattle"
print(facts["employer"].value)   # "Microsoft"

bdg = BeliefDependencyGraph()
# add beliefs and dependencies, then:
result = bdg.propagate_cascade(corrected_node_id, delta_0=1.0)
print(result.max_pressure, result.avg_pressure)
```

## How you'd wire it into an existing agent

Three touchpoints. Aether doesn't replace anything. It wraps.

```python
from aether.governance import GovernanceLayer
from aether.contradiction import StructuralTensionMeter
from aether.memory import extract_fact_slots

gov = GovernanceLayer()
meter = StructuralTensionMeter()

# before the LLM call: pull structured facts out, check for tension
user_facts = extract_fact_slots(user_message)

# your LLM call, unchanged
response = your_llm_call(messages)

# after the LLM call: check the response against the belief state
result = gov.govern_response(response, belief_confidence=0.6)
if result.should_block:
    response = "I'm not confident enough to answer that."
```

## The six laws

| Law | Agent | What it catches |
|-----|-------|----------------|
| 1. Speech cannot upgrade belief | `SpeechLeakDetector` | Generated text being treated as evidence for itself |
| 2. Low variance does not imply confidence | `TemplateDetector` | RLHF hedge templates that look like real uncertainty |
| 3. Contradiction must be preserved before resolution | `PrematureResolutionGuard` | Held tensions getting collapsed too early |
| 4. Degraded reconstruction cannot silently overwrite | `MemoryCorruptionGuard` | Compressed or hallucinated rewrites overwriting trusted memory |
| 5. Confidence must be bounded by internal support | `GapAuditor` | The belief/speech gap. Same thing Anthropic calls internal-external decoupling. |
| 6. Confidence must not exceed continuity | `ContinuityAuditor` | The system contradicting what it just said two turns ago |

## Modules

| Module | Status | Notes |
|--------|--------|-------|
| `aether.governance` | shipped | Six agents plus the `GovernanceLayer` dispatcher |
| `aether.contradiction` | shipped | Structural tension. Zero model calls. |
| `aether.epistemics` | shipped | Belief backpropagation, trust evolution |
| `aether.memory` | shipped | Slots, memory graph, BDG with cascade pressure |
| `aether.mcp` | shipped (v0.5.0) | 13-tool MCP server: substrate-grounded sanction + fidelity, embedding-aware search, contradiction detection on write, correction with BDG cascade, lineage, cascade preview, belief history, session diff. |
| `aether.adapters` | planned | Cross-vendor adapters so the substrate stays the same regardless of which LLM is the mouth |

See [ROADMAP.md](ROADMAP.md) for what's coming and what's intentionally out of scope.

## Where it fits next to other tools

| | Storage scope | Tracks contradiction | Belief/speech gap | Cross-vendor portable | Cascade pressure |
|---|---|---|---|---|---|
| Mem0, Letta, Zep, Cognee | memory layer | as overwrite | no | partial | no |
| Microsoft Agent Governance Toolkit | runtime policy | no | no | yes | no |
| Anthropic / OpenAI memory features | per-vendor | no | no | no | no |
| Aether | belief substrate | first-class state (held / settling / settled) | measured by Law 5 | yes | yes |

## Design choices, briefly

A contradiction is information, not a bug. Some are meant to stay open.

Trust isn't assigned, it's earned. It moves under reinforcement, correction, and time.

The belief/speech gap should be logged, not hidden. You want to see when the system says more than it knows, even if you choose not to act on it every time.

The model is the mouth, not the self. The governance and the belief state should work the same regardless of which LLM is producing the words.

Structure beats semantics for this kind of work. Slot comparison at 88 percent accuracy beats LLM-as-judge at 40 percent.

Cascade pressure can be measured. Belief revisions propagate through a graph with bounded depth and damping. There is real math under it; a paper is in flight.

## Where this came from

This grew out of running an assistant in production for a long time and watching the same problems come back. Continuity drift between sessions. Contradictions getting silently smoothed over. Trust on a fact climbing back up after the user corrected it twice. The structural tension meter came out of an experiment where removing the LLM from the belief-verification step roughly doubled accuracy, which was annoying and clarifying in equal measure.

The architecture was originally called CRT (Contradiction-aware Reconciliation and Trust). It is now Aether, which is what it has always actually been.

## License

MIT.

## Author

Nick Block, [@blockhead22](https://github.com/blockhead22).
