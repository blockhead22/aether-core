# Aether — next session handoff (after 2026-04-27 build marathon)

## Where we left off

`aether-core` is on PyPI at **v0.8.2**. Tonight we built five releases:

| Version | What it shipped |
|---|---|
| v0.5.0 | Substrate-grounded sanction + fidelity; 8 new MCP tools (correct, lineage, cascade-preview, belief-history, contradictions, resolve, session-diff, memory-detail); BDG cascade |
| v0.6.0 | Mutex contradictions (cloud providers, package managers, etc.); auto-ingest extractor; Belnap warnings |
| v0.7.0 | Repo-aware substrate discovery (`.aether/state.json`); `aether` CLI (init/status/contradictions/check); pre-commit + GitHub Action |
| v0.8.0 | Claude Code plugin packaging (`claude plugin install github.com/blockhead22/aether-core`); 7 slash commands |
| v0.8.1 | Fix: `aether_context` no longer blocks on encoder cold-load |
| v0.8.2 | Fix: `aether_search` non-blocking via background warmup + process-wide cache; substring fallback during warmup |

152 tests pass. CI green on master. PyPI: `aether-core==0.8.2`.

## Goal of next session

**Implement v0.9 — Dijkstra shortest-path retrieval.** This is the "park map" idea (RollerCoaster Tycoon analogy in `ROADMAP.md`). Right now `aether_search` is greedy local search — top-K by cosine. v0.9 adds `aether_path` which runs Dijkstra backward over the BDG to return the cheapest dependency chain that grounds a query. This is what makes "preload context" actually correct instead of just topically similar.

Then test it end-to-end: drive a real coding task in a real repo with the substrate active, the auto-ingest hook firing, sanction gating risky actions, and `aether_path` preloading context. Measure the felt difference vs. baseline (substrate disabled).

## What's already done in v0.9 prep

- ROADMAP entry written (`ROADMAP.md` "Up next" section)
- The BDG already supports backward propagation in `aether/memory/graph.py` (`BeliefDependencyGraph.propagate_backward`) — Dijkstra can reuse the edge-walking machinery
- `aether_lineage` already walks SUPPORTS edges back; weight + budget logic is the new piece

## Substrate state

User-global: `~/.aether/mcp_state.json`. Currently has 4 memories from this session (test memories about the OSS surface). One was corrected to reflect the v0.8.2 14-tool surface. Trust history side-car logs all changes.

## Hooks active in this project

`D:/AI_round2/.claude/settings.local.json` registers three project-level hooks:

- **SessionStart** (`async`): runs `pip install -U --no-deps aether-core` so this session always has the latest. Silent unless an upgrade happens.
- **Stop**: extracts high-signal facts from each completed turn via `aether.memory.ingest_turn`. Logs writes to stderr.
- **PreCompact**: builds a substrate brief (counts, Belnap states, 24h diff, recent memories) and injects as `additionalContext` so the post-compact session starts substrate-aware.

Scripts at `D:/AI_round2/.aether-hooks/`. All three pipe-tested green.

## Allowlist

All 14 aether-oss MCP tools are on the allow list — no permission prompts during testing. The new `aether_path` tool (v0.9) will need to be added once it ships.

## First moves for next session

1. **Verify hooks fired** — open `/hooks` to confirm. SessionStart should have run on startup.
2. **Verify substrate surface** — call `aether_context`. Should return instantly with `embeddings_loaded: true` (warmup done by now).
3. **Build `aether_path`** — Dijkstra retrieval over BDG. ~100 lines in `aether/mcp/state.py`, plus the MCP tool wrapper, plus a `/aether-path` slash command, plus tests.
4. **Ship v0.9.0** — version bump, commit, GitHub release, PyPI publish via OIDC.
5. **Real-codebase test** — pick a repo, run a coding task that benefits from preloaded context (e.g. "add retry to API client" in a codebase with prior retry decisions). Measure: did sanction block anything wrong, did fidelity catch overclaims, did `aether_path` preload the right ancestors?

## Open follow-ups (not blocking v0.9)

- **Mutex registry expansion** — language runtimes, cache layers, queue systems, monitoring vendors, IaC tools.
- **Held-contradiction state machine** — Active → Settling → Settled → Archived.
- **`aether_done_check` / `aether_done_shape`** — declared success criteria, grade against them.
- **Deeper Stop hook payload handling** — current `stop_auto_ingest.py` reads several payload shapes defensively; would benefit from a real Claude Code Stop payload dump to refine.

## Files of interest

- `aether/mcp/state.py` — StateStore, all substrate ops live here
- `aether/mcp/server.py` — MCP tool surface (14 tools)
- `aether/memory/graph.py` — MemoryGraph + BeliefDependencyGraph + cascade propagation
- `aether/contradiction/mutex.py` — class-based contradiction detection
- `aether/memory/auto_ingest.py` — heuristic fact extractor
- `aether/cli.py` — `aether init / status / check` CLI
- `tests/test_mcp_v05.py`, `test_mcp_v06.py`, `test_cli_v07.py`, `test_encoder_nonblocking.py`, `test_state_stats_lazy.py` — covers the full surface

## Useful commands

```powershell
# Run full test suite (152 tests)
cd D:/AI_round2/aether-core
python -m pytest tests/ -q

# Check substrate state from CLI
aether status

# Check what the substrate believes about a draft
aether check --message "Some claim text" --format json

# List contradictions
aether contradictions
```

---

*This doc is the warm-start brief for the next session. Read it first, then start building v0.9.*
