# Aether — next session handoff (after 2026-04-30 evening)

## Where we are

`aether-core` is at **v0.12.5** on origin master, commit `ad27f1b`, **published live on PyPI as the latest version**. **361+ tests pass**, zero xfail, CI green on master.

Today (2026-04-30) shipped **five PyPI releases** in one day, each fixing or hardening a real finding the substrate-assisted dev loop surfaced:

| Version | What | Findings closed |
|---|---|---|
| 0.12.1 | E2E harness scaffold + v0.9.6 carry-overs (link helper, version stamp, [mcp] networkx, release.yml trigger) | F#1, F#2 |
| 0.12.2 | Extended policy contradiction detection — IMPERATIVE_CUES covers real CLI forms; strong-trust override bypasses sim gate when belief trust ≥ 0.85 | F#4 |
| 0.12.3 | F#7 fix — StateStore syncs from disk on external writes; auto-ingest hook writes survive server saves | F#7 |
| 0.12.4 | `aether doctor` diagnostic command — six checks (imports, state file, activity, encoder, hook, mcp registration) that surface silent breakage in seconds | (catches future F#1-style bugs) |
| 0.12.5 | F#3 fix — `MemoryGraph.get_memory()` catches deserialization errors; read tools degrade to "unknown memory_id" instead of crashing on corrupt nodes | F#3 |

E2E harness expanded from turns 1-2 → turns 1-10 (full scripted scenario). 9 full_loop tests + 4 smoke tests, all passing.

The auto-ingest Stop hook was fixed to actually read `transcript_path` (had been a silent no-op for 3 days because Claude Code sends a path, not inline messages). Hook now fires every turn and writes high-trust facts to the substrate. The substrate grew from 38 → 41+ memories during the session as a direct consequence.

### Findings status (all 7 closed)

| ID | Description | Disposition |
|---|---|---|
| F#1 | `[mcp]` install missing networkx → server crash on first tool call | Fixed v0.12.1 |
| F#2 | release.yml triggered on `release: published` not tag push → no PyPI publishes since 0.9.0 | Fixed v0.12.1 |
| F#3 | Read tools crash with TypeError on corrupt nodes (zombie `backfill`-shape) | Fixed v0.12.5 |
| F#4 | `aether_sanction` approves `git push --force` against "Never force-push" belief | Fixed v0.12.2 |
| F#5 | `aether_link("supports")` doesn't override existing `related_to` | Closed — test-author error in turn-8 e2e, not a bug |
| F#6 | Auto-link sim threshold leaves dissimilar facts disconnected | Closed — expected behavior, threshold is tunable |
| F#7 | Stop hook + MCP server share state file with no coordination → server clobbers hook writes | Fixed v0.12.3 |

## Strategic state

The substrate is now **structurally complete** for in-session use:
- Auto-ingest fires after every turn → substrate grows on its own.
- `_sync_first` decorator ensures the server picks up external writes on the next tool call.
- `aether doctor` diagnoses install / hook / MCP / state-file issues in seconds.
- All read tools degrade gracefully on corrupt nodes.
- All write tools sync from disk before mutating, so hook writes survive.

The remaining gap is **richness, not architecture**. With 41 memories the substrate can't ground much; the value compounds at 1000+. The fix for that is just time + sessions — which the auto-ingest hook now does for free as long as you're working.

OSS remains the main focus. Today's marathon proves the substrate-assisted dev loop is real: every shipped release was validated by Aether's own tools (sanction APPROVE for cleanup, search for prior context, fidelity for grounding, doctor for health). The README's "I built a substrate that caught itself shipping bugs" thesis got 5 fresh data points today.

## What we're working on next

Three plausible threads, in priority order:

### 1. README rewrite + onboarding (highest external leverage)

The handoff has called for this for two sessions. With the architecture now stable and all findings closed, this is the bottleneck for adoption. Today's work proves the value-prop concretely; the README still reads like a feature list, not a "what this changes about your assistant" pitch.

Concrete deliverables:
- Top section: 3 sentences explaining why this exists. "The model is the mouth, the substrate is the self." cite something concrete from today (e.g. "F#7 was caught by the substrate auditing the substrate's own writes").
- Quickstart: `pip install aether-core[mcp,graph,ml]` → claude plugin install → `aether doctor` → first remember → first sanction.
- "What this catches that other tools don't": cross-session belief continuity, contradiction-as-signal, governance gate. Avoid "memory MCP" framing.
- Drop or move: long lists of tools, internal architecture sections. Move to docs/.

### 2. Validation chapter

Per prior handoffs: 4 tests — N>1 user, cross-vendor, fresh-session-no-context, scale (1k/10k memories). The substrate is now stable enough to start collecting external evidence rather than dogfood-only.

### 3. Slot extractor name patterns

Production CRT layers have `user.name` slots; OSS extractor returns empty for "My name is Nick" / "I am Nick" patterns. Adding a name pattern bank (~30 lines) closes the gap and lets the slot-equality detector catch the canonical Nick<>Aether case the v0.12 audit cited.

## Other parking-lot items

- **Backport audit to main repo** (`D:/AI_round2/personal_agent/`). Probably has the same slot-equality + local-context blind spots OSS just fixed. ~2-3 hours mechanical port.
- **Cascade complexity paper to arxiv** (drafted in `papers/cascade_complexity/`).
- **Blog post:** *"I built a substrate that caught itself shipping bugs"* — today's 5-release marathon is the new climax (was 38-hour story).
- **Production aether bug:** `crt_search_sessions` returns 0 sessions for common topics despite 79k memories indexed. Session indexer out of sync. Not OSS-blocking.
- **F#7 deeper architecture:** the disk-sync fix is correct but pessimistic — every tool call does an `os.stat`. Long-term, having the hook talk to the running MCP server via stdio would be cleaner (no file polling). Not urgent; current fix is solid.

## Useful commands

```powershell
cd D:/AI_round2/aether-core

# Tests (348 unit + 13 e2e = 361)
python -m pytest tests/ -q                         # everything (~3 min, e2e builds a venv)
python -m pytest tests/ --ignore=tests/e2e -q      # just unit (~25s)
python -m pytest tests/e2e/ -v                     # just e2e (~2 min)

# Diagnostic
aether doctor                                      # check install/hook/mcp/state health
aether doctor --format json                        # for scripts

# Calibration bench
python bench/run_fidelity_bench.py            # warm 29/29
python bench/run_fidelity_bench.py --cold     # cold 21/29

# Substrate state
aether status

# Verify latest publish
pip install --upgrade aether-core
python -c "import aether; print(aether.__version__)"   # should print 0.12.5
```

## First moves for next session

1. **Run `aether doctor`** to confirm everything is wired correctly. If anything is FAIL or WARN, address before continuing.
2. **Pick one of the three threads above.** README rewrite is the highest external leverage.
3. **Check `aether_search`** for memories the auto-ingest captured since last connect. The substrate should be richer than it was — that's a material data point for the README pitch.

## What stays closed

- `electron/` — agent product
- `dnnt/` — training pipeline (model-agnostic conflict)
- Cloud features, multi-user, billing
- Specific Claude Code dispatch / action execution

---

*This doc replaces the v0.12.0 / v0.12.1 handoffs. Strategic context (OSS-as-main-focus) unchanged. Today shipped five PyPI releases, closed all seven open findings, fixed the auto-ingest hook (3-day silent bug), and added a diagnostic command (`aether doctor`) that prevents the same class of silent bug from happening again. The substrate is now structurally complete; growth and onboarding are the remaining work.*
