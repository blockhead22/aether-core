# Roadmap

What is shipped today, what is coming next, and what is intentionally not in this repo.

## Shipped (v0.9.0)

### Shortest-path retrieval — the RollerCoaster Tycoon idea, now real

`aether_path(query, max_tokens=2000, max_hops=8)` runs Dijkstra backward over the BDG from the top-1 cosine match. Edge weights = `(1 - trust) * token_estimate(text)` — high-trust memories are cheap, low-trust ones are expensive. CONTRADICTS edges are skipped entirely (held contradictions are closed paths). Returns the cheapest dependency chain that fits in `max_tokens`, ordered by Dijkstra distance from target.

Three semantic-preserving guarantees the test suite locks in:

1. Empty substrate → `method: "no_substrate"`, no errors.
2. Target-only (no ancestors) → path is just the target.
3. CONTRADICTS edges count toward `closed_paths` but never enter the path.

163 tests pass.

Plus a `/aether-path` slash command and the `aether_path` MCP tool exposed in v0.9.0 of the Claude Code plugin.

## Up next

In RCT, park guests without a map wander — they scan local intersections and pick paths heuristically. Guests *with* a map have a precomputed shortest route to their goal. Buying a map trades cash for search cost. Chris Sawyer wrote that pathfinder in assembly.

The substrate today does the equivalent of "wandering" — `aether_search` returns top-K by cosine similarity, which is greedy local search over the embedding space. It can pull five memories that all overlap with each other and miss the one upstream memory that everything else depends on.

What's missing: weighted shortest-path retrieval over the BDG.

```
aether_path(target_query, max_tokens=2000) -> [memory_id chain]
```

Implementation sketch:
1. `target = aether_search(target_query, limit=1)` — pick the destination
2. Run Dijkstra backward over BDG from `target`, edge weights = `(1 - trust) * token_estimate(memory)`
3. Return the path that fits in `max_tokens`, ranked by total grounding gain

Why this matters:
- **Cost-weighted preload.** Compute the cheapest set of memories that grounds the query, then preload that set. Beats top-K because it follows dependency structure instead of just topical similarity.
- **Map-purchase as meta-decision.** Cheap Dijkstra estimates "how much context do I need." If small, preload. If huge, fall back to wander mode (search-on-demand).
- **Held contradictions = closed paths.** A memory in Belnap state `B` is a dead-end edge — Dijkstra naturally routes around it. That's exactly how a careful reasoner should behave.
- **Per-task maps.** Each session keeps a preloaded subgraph for its current task. Task switches drop the old map, load a new one. Same pattern as RCT's per-guest map state.

Adds <100 lines. Substrate has the graph already. New tool, new slash command, new chapter in the README that explains memory-as-map without any AI jargon.

### Other v0.9 candidates

- **Mutual-exclusion class registry expansion.** Today: 10 classes (cloud_provider, package_manager_*, database, etc.). Add: language runtimes, cache layers, queue systems, monitoring vendors, IaC tools.
- **Held-contradiction state machine.** Active → Settling → Settled → Archived with policies for each transition.
- **`aether_done_check` / `aether_done_shape`.** Declared success criteria, then grade a response against them.

## Shipped (v0.8.0)

- **Claude Code plugin packaging.** Install with one command:

  ```
  claude plugin install github.com/blockhead22/aether-core
  ```

  Wires up the MCP server registration, the auto-ingest Stop hook, a SessionStart hook that pip-installs `aether-core[mcp,graph,ml]` if it isn't already present, and seven slash commands:

  - `/aether-status`
  - `/aether-search <query>`
  - `/aether-contradictions [disposition]`
  - `/aether-check <draft>`
  - `/aether-init`
  - `/aether-ingest`
  - `/aether-correct <memory_id> [reason]`

  Plugin layout follows the canonical structure: `.claude-plugin/plugin.json`, `.mcp.json` at the repo root, `hooks/hooks.json`, and `commands/*.md`. Manual MCP install still works for non-Claude-Code clients (Cursor, Cline, Continue, Goose, Zed, LM Studio).

## Shipped (v0.7.0)

- **Repo-aware substrate discovery.** `StateStore` walks up from cwd looking for `.aether/state.json` and uses it when found. Falls back to user-global `~/.aether/mcp_state.json`. Override with `$AETHER_STATE_PATH`. Disable discovery with `$AETHER_NO_REPO_DISCOVERY=1`. The substrate becomes a per-repo team artifact, not just per-developer state.
- **`aether` CLI**. Four subcommands:
  - `aether init` — scaffolds `.aether/` with empty state, README, and `.gitignore` (exclude `state_embeddings.npz`).
  - `aether status` — substrate stats.
  - `aether contradictions [--disposition]` — lists current contradictions.
  - `aether check` — runs substrate-grounded fidelity on text from `--message`, `--message-file`, `--diff`, or stdin. Returns non-zero at `--fail-severity` (default CRITICAL). Designed for pre-commit and CI.
- **Pre-commit hook** at `examples/git-hooks/pre-commit`. Reads commit message + staged diff, runs `aether check`, blocks at CRITICAL.
- **GitHub Action** at `examples/github-actions/aether-check.yml`. Same logic on PRs, posts a comment with the grounding report and fails the check at CRITICAL.
- 142 tests pass.

## Shipped (v0.6.0)

- **Mutual-exclusion contradiction detection** (`aether.contradiction.detect_mutex_conflict`). A registry of canonical class-valued facts (cloud providers, package managers, databases, frontend frameworks, backend runtimes, container orchestrators, auth providers, payment processors, VCS hosts) catches the cases the structural meter misses — "we deploy to AWS" vs "we deploy to GCP" lands a CONTRADICTS edge with `kind: mutex`. Adding a new class is one entry in `DEFAULT_CLASSES`.
- **Auto-ingest extractor** (`aether.memory.auto_ingest`). Pure-regex heuristic that pulls high-signal facts from a conversation turn — preferences, identity, project facts, decisions, constraints, corrections. Conservative on purpose. Ships with a sample Claude Code Stop hook (`examples/claude-code-hooks/auto_ingest_hook.py`) so the substrate fills automatically without users calling `aether_remember` by hand.
- **`aether_ingest_turn` MCP tool**. Same extractor, exposed for direct invocation. Dedupes against substrate before writing.
- **Belnap-state visibility on search**. Search results now carry a `warnings` field — "contested: held contradiction", "deprecated: superseded", "uncertain: insufficient evidence" — so the LLM knows when a memory is in a non-T state.
- 133 tests pass.

## Shipped (v0.5.0)

- `aether.governance`. Six immune agents and the four-tier `GovernanceLayer` dispatcher.
- `aether.contradiction`. The `StructuralTensionMeter`. Zero-LLM tension detection between two beliefs at about 0.2 seconds per pair.
- `aether.epistemics`. `EpistemicLoss`, belief backpropagation, `DomainVolatility`.
- `aether.memory`. Fact slot extraction, `MemoryGraph`, and a `BeliefDependencyGraph` that propagates cascades with measurable pressure (`propagate_cascade`, `propagate_backward`, held-node firewalling).
- `aether.mcp`. Standalone MCP server with 13 tools. Persists state to JSON plus a side-car trust-history log.
  - **Memory:** `aether_remember` (with auto contradiction detection on write), `aether_search` (embedding + substring hybrid), `aether_memory_detail`.
  - **Governance:** `aether_sanction` and `aether_fidelity` — both substrate-grounded. When the caller omits `belief_confidence`, the tool searches the substrate and computes a real grounding score from supporting and contradicting memories. Sanction includes a policy-contradiction check that catches command-vs-prohibition cases the structural tension meter misses.
  - **Substrate ops:** `aether_correct` (with BDG cascade through SUPPORTS edges), `aether_lineage`, `aether_cascade_preview` (dry-run, no commit), `aether_belief_history`, `aether_contradictions`, `aether_resolve`, `aether_session_diff`.
- PyPI release with auto-publish via Trusted Publishers (OIDC).
- 113 tests, GitHub Actions CI, MIT license, Python 3.10 and up.

## Near term (next one or two minor releases)

`.aether/` repo artifact. A directory checked into a project's git so the substrate becomes a team artifact, not just per-developer state. Onboard a new dev and they inherit the repo's accumulated decisions. CI hook that runs `aether_fidelity` on PR description and diff.

Claude Code plugin packaging. So `claude plugin add aether-core` works without manual `.claude/settings.json` editing.

Auto-ingest hook. Stop-event hook that scans the last turn for high-signal facts and writes them with `aether_remember`. The substrate fills without the user having to remember.

Variance probe. Per-model fragility characterization. Same prompt, several LLMs, measure where the belief/speech gap diverges. Useful both as a diagnostic and as evidence that the substrate is what gives you portability.

Held-contradiction lifecycle. Today there is a `Disposition.HELD` enum and the right primitives. The full state machine (Active to Settling to Settled to Archived, with policies for each transition) is not yet in.

`aether_done_check` / `aether_done_shape`. Declared success criteria, then grade a response against them.

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
