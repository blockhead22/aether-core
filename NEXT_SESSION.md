# Aether — next session handoff (after 2026-04-29 morning)

## Where we are

`aether-core` is at **v0.12.0** on origin master, commit `bf1cb7b`, tag `v0.12.0` pushed. **323 tests pass.** Calibration bench: warm **29/29 (100%)**, cold **21/29 (72.4%)** — both up from v0.11 baseline.

Marathon since 2026-04-27 has shipped eight versions: v0.9.1 → v0.9.2 → v0.9.3 → v0.9.4 → v0.9.5 → v0.10.0 → v0.10.1 → v0.11.0 → v0.12.0. Tests grew 163 → 323.

### v0.12.0 — what just shipped

Two parallel fixes from Lab A v2's production-substrate audit:

- **Slot-equality detector** (`patterns.slot_equality`). Closes the 0/42 detection gap on production's real categorical contradictions (Nick<>Aether on `user.name`, blue<>orange on `user.favorite_color`, etc.). Wired into both write path (`_detect_and_record_tensions`, `kind=slot_value_conflict`, `nli_score=0.9`) and read path (`compute_grounding` extracts draft slots on the fly). Slot-keyed candidate pre-screen bypasses the `sim<0.2` gate so low-textual-similarity slot conflicts still reach detection.
- **Shape local-context gate.** `LOCAL_CONTEXT_TOKENS=3`, `LOCAL_CONTEXT_MIN_OVERLAP=0.30`. Suppresses co-topical false positives. Caught by the substrate-assisted dev loop's own dogfooding — the v0.12 design's first sanction REJECTED on a real v0.11 production bug. Same release closes the gap the audit found AND the bug the audit triggered.
- **Auto slot extraction in `add_memory`** when `slots=None`. Memories now carry slot tags by default.

Audit trail: `bench/lab_a_v2_production_substrate_findings.md`. Audit-close memory recorded: `m1777365692026_1` in OSS substrate (trust 0.95).

### Strategic state (unchanged from prior handoff)

OSS is the main focus. Main repo (`D:/AI_round2/personal_agent/`) is the workshop. OSS is what gets cited / installed externally / built on by the validation chapter.

## What we're working on next

**E2E harness.** Decided this morning. The 323 tests are all unit + slice — no test covers the full loop as a real user touches it. The harness is ~1 day of work and pays back as test infrastructure AND surfaces every install / onboarding gap as concrete findings instead of guesses.

### E2E harness scope

Not Python-import integration tests — those exist already. The point is to drive Aether **via the same surface a real user does**: pip-install in a fresh venv, spawn the MCP server as a subprocess, talk to it over the MCP wire protocol, run a scripted scenario hitting every tool.

Skeleton:

```
tests/e2e/
  conftest.py            # fixtures: fresh venv, subprocess MCP server
  test_install_smoke.py  # pip install [mcp,graph,ml] in tmpdir venv works
  test_full_loop.py      # 10-turn scripted scenario:
                         #   1. aether_remember (seed 3 facts)
                         #   2. aether_search verifies retrieval
                         #   3. aether_remember a contradicting fact
                         #   4. tension_findings surfaces it
                         #   5. aether_sanction on a related action
                         #   6. aether_fidelity on a draft
                         #   7. aether_correct cascade
                         #   8. aether_lineage walks the BDG
                         #   9. aether_path returns weighted route
                         #  10. aether_session_diff briefs returning agent
  test_cold_warm_modes.py  # same scenario, both encoder modes
  test_plugin_install.py   # claude plugin install path (if feasible)
```

Implementation notes:
- `subprocess.Popen` with `python -m aether.mcp.server`
- MCP protocol uses JSON-RPC over stdio; write a thin client wrapper
- Cleanup: `~/.aether/` mocked to tmp dir via `AETHER_STATE_PATH`
- Cold mode: don't wait for embedding warmup; assert tools still work
- Warm mode: wait for `is_loaded`; assert improved scores

What it should catch:
- The `aether_path` no-op bug (caught in v0.9.1 by hand) would have failed `test_full_loop.py` step 9
- The cold-encoder crash (v0.9.5) would have failed `test_cold_warm_modes.py`
- The save/load metadata collision (v0.10.1) would have failed any second-call test
- The shape false positive (v0.12) would have shown up in step 4 as a spurious contradiction

If the harness existed before this marathon, several of those bugs ship-to-fix cycles would have been compressed.

### After the harness

Two threads tee themselves up from harness findings:

- **Install ergonomics.** `aether doctor` diagnostic command (the harness defines what it should check). Lighter default install — `[ml]` is heavy. Progress UI on cold-start model download. State-path docs cleanup (currently three docs explain `~/.aether/` vs `.aether/` vs `$AETHER_STATE_PATH` three different ways).
- **Onboarding.** `/aether-tour` slash command (60-sec walkthrough demoing contradiction + resolution + receipt). First-run substrate seed memories so the LLM has context to answer "what is this?" accurately. README rewrite — top section explains "what this changes about your assistant" in 3 sentences, not feature list. One-time SessionStart nudge on empty substrate.

## Background — three concerns that surfaced this morning

1. **End-to-end testing is missing.** Confirmed above. We picked this for next session.
2. **Better installation for OSS users.** Friction points already enumerated: heavy `[ml]` extra, silent SessionStart pip install, ambiguous state path docs, no progress UI on first model download, no `aether doctor`, empty-substrate cold start has no "what do I do" hint.
3. **Explicit onboarding.** Plugin model means we don't own the shell, but the value prop *requires* a mental model the user has to grasp. Without onboarding users reduce Aether to "a memory MCP" and never touch sanction / fidelity / contradictions. Solution shape: minimal + opt-in (tour + first-run seed + README rewrite + one-time nudge).

## Parking lot (not blocking)

- **PyPI publish v0.12.0.** Tag is pushed (`v0.12.0`); verify the OIDC workflow ran and `pip install aether-core==0.12.0` resolves.
- **Backport audit to main repo.** Main almost certainly has the same slot-equality + local-context blind spots OSS just fixed. ~2-3 hours mechanical port.
- **Dijkstra `aether_path` v2** — the RCT-map idea. Cost-weighted shortest-path retrieval over BDG. <100 lines. Lab A v2's 42 production node pairs are the input set.
- **v0.9.6 cleanup** that never shipped: unify auto-link similarity formula (~10 lines), substrate writes its own version on release (~5 lines), correct 5 polluting test memories.
- **Blog post:** *"I built a substrate that caught itself shipping bugs"* — 38-hour story with v0.12 meta-finding as climax.
- **Cascade complexity paper to arxiv** (drafted in `papers/cascade_complexity/`).
- **Validation chapter (4 tests):** N>1 user, cross-vendor, fresh-session-no-context (clean), scale (1k/10k memories).
- **Production aether bug:** `crt_search_sessions` returns 0 sessions for common topics despite 79k memories indexed. Session indexer out of sync. Not OSS-blocking.

## Useful commands

```powershell
cd D:/AI_round2/aether-core

# Tests (323)
python -m pytest tests/ -q

# Calibration bench
python bench/run_fidelity_bench.py            # warm 29/29
python bench/run_fidelity_bench.py --cold     # cold 21/29

# Verify v0.12.0 publish
pip install --upgrade aether-core
python -c "import aether; print(aether.__version__)"

# Substrate state
aether status
```

## First moves for next session

1. **Verify v0.12.0 published to PyPI.** Check the GitHub Actions OIDC workflow for tag `v0.12.0`. If green, `pip install -U aether-core` should resolve to 0.12.0.
2. **Scaffold `tests/e2e/`.** Start with `test_install_smoke.py` — fresh venv, pip install, import check. Smallest possible thing that exercises the install path.
3. **Subprocess MCP harness.** `test_full_loop.py` skeleton. Don't write all 10 turns yet — get 2 turns working end-to-end (remember + search), then expand.
4. **Capture findings as they surface.** Every install bug or rough edge the harness exposes goes in a list. That list IS the install-ergonomics + onboarding spec.

## What stays closed

- `electron/` — agent product
- `dnnt/` — training pipeline (model-agnostic conflict)
- Cloud features, multi-user, billing
- Specific Claude Code dispatch / action execution

---

*This doc replaces the v0.9.5 handoff. Strategic context (OSS-as-main-focus) unchanged. Three concerns from this morning's session — e2e gap, install ergonomics, onboarding — are the active threads. We picked the e2e harness as the lever that surfaces the other two.*
