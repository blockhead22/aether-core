# Aether — next session handoff (after 2026-04-29 evening)

## Where we are

`aether-core` is at **v0.12.1** on origin master, commit `64d5595`, tag `v0.12.1` pushed and **published live on PyPI as the latest version** (`pip install aether-core` resolves to 0.12.1; first PyPI publish since 0.9.0). **328 tests pass** plus **8 e2e tests** (7 passing, 1 xfail capturing a real finding).

Today's session was the e2e harness pivot from the prior handoff. Goal was: scaffold the harness, verify install path, get 2 turns working, capture findings. We ended up shipping considerably more than that.

### v0.12.1 — what just shipped

E2E harness scaffold (new `tests/e2e/`):
- `conftest.py` — session-scoped fresh-venv fixture, per-test state-path isolation, async `mcp_session` helper using the `mcp` SDK's `stdio_client`. No pytest-asyncio dependency.
- `test_install_smoke.py` — 4 tests: import works, version is right, server builds, console scripts land. Includes regression pin for `[mcp]`-only installs.
- `test_full_loop.py` — turns 1-5 of the scripted scenario over real MCP stdio (4 passing, 1 xfail).

Carry-overs from the v0.9.6 cleanup list (pre-existing on disk this session, all committed in v0.12.1):
- `[mcp]` extra now declares `networkx` (F#1 fix — was the first finding the harness caught).
- `release.yml` triggers on `push: tags: ['v*']` instead of `release: published` (F#2 fix — caught when checking PyPI publish status; tags v0.10.0 → v0.12.0 had been pushed but no GitHub Releases were ever created so OIDC publish never ran).
- `aether/mcp/state.py`: `_link_threshold(mode)` helper unifies the auto-link similarity decision used by both `add_memory` and `backfill_edges` (B3).
- `aether/memory/graph.py`: `save()` stamps `aether_version` into the state file so future loads can detect old state and apply migrations (B4).

Substrate cleanup (per-user runtime data, not in git): pre-v0.10.1 metadata-collision zombie node `id=backfill` + 2 RELATED_TO edges hard-deleted from `~/.aether/mcp_state.json` after `aether_sanction` APPROVE (action_id `ccf7e316`). Audit memory `m1777477400408_v096cleanup` written. Backup at `~/.aether/mcp_state.pre_v096_cleanup_1777477345.json`.

## What we're working on next

**Continue the e2e harness — turns 6-10.** Remaining test bodies, each its own test in `test_full_loop.py` so failures localize:

  6. `aether_fidelity` on a draft — grade a written response against the substrate, expect non-zero score and methodological_concerns when the draft makes an unsupported inference.
  7. `aether_correct` cascade — drop trust on a memory, assert the correction propagates through SUPPORTS edges to dependents (cascade complexity paper's headline behavior).
  8. `aether_lineage` walks the BDG — seed a SUPPORTS chain via `aether_link`, assert lineage returns the ancestors.
  9. `aether_path` returns weighted route — Dijkstra retrieval over the BDG. **High regression value** because v0.10.1 had a 12-hour silent-crash bug here.
 10. `aether_session_diff` briefs returning agent — write some memories, call session_diff with a past timestamp, assert the new memories appear.

After turns 6-10:
- `tests/e2e/test_cold_warm_modes.py` — same scenario in cold mode (no embeddings) and warm mode (sentence-transformers loaded). Pins the v0.9.5 cold-encoder fix.
- `tests/e2e/test_plugin_install.py` — Claude Code plugin install path if feasible. Likely surfaces the most onboarding findings.

## Open findings (caught by the harness)

- **F#3 (open):** `aether_memory_detail`, `aether_lineage`, `aether_cascade_preview` all crash with `TypeError: MemoryNode.__init__() missing 3 required positional arguments` when fed a corrupt node id. Should return a graceful "node not found / not deserializable" response. One-line type guard in each tool plus a deserialization-failure unit test.
- **F#4 (xfail in `test_sanction_non_approves_action_contradicting_substrate`):** `aether_sanction` approves an action that contradicts a high-trust prohibition belief, because `IMPERATIVE_CUES` requires substrings like `force push` or `push to main` — natural CLI form `git push --force` matches none of them. Plus cold-encoder Jaccard similarity falls below the 0.45 sim gate. Fix candidates: extend `IMPERATIVE_CUES` to cover real CLI forms (`--force`, `-f origin`, etc.); add a cue-only fallback when sim is below threshold but trust is above `POLICY_CONTRA_MIN_TRUST`; or lower the sim gate when cues fire on both sides. xfail flips green automatically when fixed.
- **Slot extractor name gap (open, observed during turn 3-4 work):** `extract_fact_slots` returns empty for `My name is X`, `I am X`, `Nick lives in X`. Production `user.name` contradictions must come from a higher CRT layer setting slots explicitly. If the OSS extractor should handle names natively, that's separate ~30-min work.

## Parking lot (unchanged from prior handoff unless noted)

- **Backport audit to main repo.** Main almost certainly has the same slot-equality + local-context blind spots OSS just fixed. ~2-3 hours mechanical port.
- **Dijkstra `aether_path` v2** — the RCT-map idea. Cost-weighted shortest-path retrieval over BDG. <100 lines. Lab A v2's 42 production node pairs are the input set.
- **Blog post:** *"I built a substrate that caught itself shipping bugs"* — 38-hour story; v0.12.1 makes a stronger close (substrate audit found bugs in the audit tools themselves).
- **Cascade complexity paper to arxiv** (drafted in `papers/cascade_complexity/`).
- **Validation chapter (4 tests):** N>1 user, cross-vendor, fresh-session-no-context (clean), scale (1k/10k memories).
- **Production aether bug:** `crt_search_sessions` returns 0 sessions for common topics despite 79k memories indexed. Session indexer out of sync. Not OSS-blocking.
- **Older PyPI tags:** v0.10.0, v0.10.1, v0.11.0, v0.12.0 are pushed as git tags but were never published to PyPI (the old `release: published` trigger required a GitHub Release UI step that never happened). v0.12.1 leapfrogs them on PyPI. If you want to backfill, create GitHub Releases retroactively or `gh workflow run release.yml --ref vX.Y.Z` per tag — but probably not worth it; the publish pipeline is fixed going forward.

## Useful commands

```powershell
cd D:/AI_round2/aether-core

# Tests (323 unit + 8 e2e = 331)
python -m pytest tests/ -q                         # everything (~3 min, e2e builds a venv)
python -m pytest tests/ --ignore=tests/e2e -q      # just unit (~30s)
python -m pytest tests/e2e/ -v                     # just e2e (~2 min)

# Calibration bench
python bench/run_fidelity_bench.py            # warm 29/29
python bench/run_fidelity_bench.py --cold     # cold 21/29

# Verify v0.12.1 publish
pip install --upgrade aether-core
python -c "import aether; print(aether.__version__)"

# Substrate state
aether status
```

## First moves for next session

1. **Pick a turn from 6-10 and write its test.** Lowest-friction continuation. Turn 9 (`aether_path`) is the highest-value test because of its prior production bug history.
2. **Or address F#4 first.** It's a real governance bug the harness caught; fix is bounded (extend `IMPERATIVE_CUES`, possibly relax sim gate). xfail will auto-flip when shipped. Could be combined with a v0.12.2 patch.
3. **Or pivot to `aether doctor`.** With the harness at 5/10 it's already proven valuable; the diagnostic command is a natural follow-up that leverages everything the harness exposed (which extras matter, what state-path env vars exist, etc.).

## What stays closed

- `electron/` — agent product
- `dnnt/` — training pipeline (model-agnostic conflict)
- Cloud features, multi-user, billing
- Specific Claude Code dispatch / action execution

---

*This doc replaces the v0.12.0 handoff. Strategic context (OSS-as-main-focus) unchanged. Today's session published v0.12.1 to PyPI, scaffolded the e2e harness through turn 5, and surfaced 4 findings via the harness (2 fixed in the same release, 2 open). Three concerns from the prior handoff — e2e gap, install ergonomics, onboarding — are now: e2e gap **partly closed**, install ergonomics **first cut shipped**, onboarding **not yet started**.*
