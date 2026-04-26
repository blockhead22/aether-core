# Roadmap

What is shipped today, what is coming next, and what is intentionally not in this repo.

## Shipped (v0.3.0)

- `aether.governance`. Six immune agents and the four-tier `GovernanceLayer` dispatcher.
- `aether.contradiction`. The `StructuralTensionMeter`. Zero-LLM tension detection between two beliefs at about 0.2 seconds per pair.
- `aether.epistemics`. `EpistemicLoss`, belief backpropagation, `DomainVolatility`.
- `aether.memory`. Fact slot extraction, `MemoryGraph`, and a `BeliefDependencyGraph` that propagates cascades with measurable pressure (`propagate_cascade`, `propagate_backward`, held-node firewalling).
- 89 tests, GitHub Actions CI, MIT license, Python 3.10 and up.

## Near term (next one or two minor releases)

PyPI publish. The package builds clean and `twine check` passes; the upload itself is pending. Once it lands, `pip install aether-core` works end to end.

`aether.mcp`. An MCP server exposing the differentiator tools as primitives any AI engineer can wire into Claude Desktop or Cursor:

- `aether_sanction`. A pre-action governance gate that returns APPROVE / HOLD / REJECT based on belief state and cascade pressure.
- `aether_lineage`. Answers "why do I believe this" by walking BDG edges.
- `aether_cascade_preview`. Dry-runs a trust change so you can see the blast radius before committing.
- `aether_fidelity`. Grades a draft response against the current belief state.
- `aether_done_check`. Grades a response against declared success criteria.

Variance probe. Per-model fragility characterization. Same prompt, several LLMs, measure where the belief/speech gap diverges. Useful both as a diagnostic and as evidence that the substrate is what gives you portability.

Held-contradiction lifecycle. Today there is a `Disposition.HELD` enum and the right primitives. The full state machine (Active to Settling to Settled to Archived, with policies for each transition) is not yet in.

## Medium term

`aether.adapters`. Cross-vendor adapters for Anthropic, OpenAI, Ollama, and local models. The portability claim ("the model is the mouth, the substrate is the self") needs runnable proof across vendors.

arXiv preprints. The cascade complexity paper (depth bound, NP-hardness conjecture, damping convergence) and the belief-backpropagation paper as AGM-style iterated revision. Both date-stamp the math.

Benchmark suite. An `aether-bench` runner against [LongMemEval](https://arxiv.org/abs/2410.10813) plus a held-contradiction benchmark, with numbers in the README that compare to Mem0, Letta, and Zep.

## Long term

`aether.compaction`. Belief-aware context compaction with trust-tiered compression. Currently lives in the private codebase; extraction is pending.

`aether.session_state`. Running belief state with an away/resume diff so a returning agent knows what it missed.

A reference assistant. A minimal demo agent showing the full integration end to end. Not a product. A proof.

## Out of scope (intentionally)

A full assistant or chatbot UI. Aether is middleware. Building Aether-the-assistant as open source competes with frontier consumer AI and dilutes the substrate framing. Not a fight worth picking.

Vendor lock-in to any single LLM provider. The point is portability across mouths.

LLM calls inside the core library. Optional adapters can use them; the core stays structural.

Hosted multi-user features (dashboards, SOC 2, audit storage). These are the paid Aether tier, not the open-source library.

## Philosophy

This roadmap is what is likely, not what is promised. Solo project. Cinema and photography sometimes win.

If something in "near term" is overdue and you care about it, open an issue or a pull request. Working code beats roadmap entries.
