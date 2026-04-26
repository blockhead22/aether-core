# Roadmap

What is shipped, what is next, and what is intentionally not in this repo.

## Shipped (v0.3.0)

- `aether.governance` — six immune agents + 4-tier dispatcher (`GovernanceLayer`)
- `aether.contradiction` — `StructuralTensionMeter`, zero-LLM tension detection
- `aether.epistemics` — `EpistemicLoss`, belief backpropagation, `DomainVolatility`
- `aether.memory` — fact slot extraction, `MemoryGraph`, `BeliefDependencyGraph` with cascade pressure (`propagate_cascade`, `propagate_backward`, held-node firewalling)
- 89 tests, GitHub Actions CI, MIT license, Python 3.10+

## Near term (next 1-2 minor releases)

- **PyPI publish** — package builds clean; upload pending. `pip install aether-core` should work end-to-end.
- **`aether.mcp`** — MCP server exposing the differentiator tools as primitives:
  - `aether_sanction` — pre-action governance gate (APPROVE / HOLD / REJECT) based on belief state and cascade pressure
  - `aether_lineage` — "why do I believe this" via BDG edges
  - `aether_cascade_preview` — dry-run a trust change to see blast radius
  - `aether_fidelity` — grade a draft response's grounding in belief state
  - `aether_done_check` — grade a response against declared success criteria
- **Variance probe** — per-model fragility characterization. Same prompt, different LLMs, measure where belief/speech gap diverges.
- **Held contradiction lifecycle** — full Active → Settling → Settled → Archived state machine on contradictions, with policies for when each transition fires.

## Medium term

- **`aether.adapters`** — cross-vendor adapters (Anthropic, OpenAI, Ollama, local models). The portability claim of "the model is the mouth, the substrate is the self" needs runnable proof across vendors.
- **arXiv preprints**: cascade complexity (depth bound, NP-hardness conjecture, damping convergence) and belief backpropagation as AGM-style iterated revision. Math moat documented in citable form.
- **Benchmark suite** — `aether-bench` runner against [LongMemEval](https://arxiv.org/abs/2410.10813) and a held-contradiction benchmark. Numbers in the README, comparable to Mem0 / Letta / Zep.

## Long term

- **`aether.compaction`** — belief-aware context compaction with trust-tiered compression (currently lives in private codebase; extraction pending)
- **`aether.session_state`** — running belief state with away/resume diff
- **Reference assistant** — minimal demo agent showing the full integration end-to-end. Not a product; a proof.

## Out of scope (intentionally)

- **A full assistant product / chatbot UI.** Aether is middleware. Building Aether-the-assistant as OSS would compete with frontier consumer AI and dilute the substrate framing.
- **Vendor lock-in** to any single LLM provider. The point is portability across mouths.
- **LLM calls inside the core library.** Optional adapters can use them; the core is structural.
- **Hosted multi-user features** (dashboards, SOC 2, audit storage). These are the paid Aether tier.

## Philosophy

This roadmap is what's likely; not what's promised. Solo project. Cinema and photography sometimes win.

If something in "near term" is overdue and you care about it, open an issue or a PR. Working code beats roadmap entries.
