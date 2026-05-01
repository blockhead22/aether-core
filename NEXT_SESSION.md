# Aether — next session handoff (after 2026-04-30 marathon)

## Where we are

`aether-core` is at **v0.12.12** on origin master, commit `4d27af3`. **413 unit tests pass**, 1 unrelated HF-offline failure, CI green on master.

Today (2026-04-30) shipped **twelve PyPI releases** total. The first eight closed all original findings (F#1–F#10) and made the substrate structurally complete. The last four (v0.12.9–v0.12.12, after the prod-readiness audit) closed five hygiene gaps required for "ready for real users":

| Version | What | Findings / gaps closed |
|---|---|---|
| 0.12.1 | E2E harness scaffold + v0.9.6 carry-overs (link helper, version stamp, [mcp] networkx, release.yml trigger) | F#1, F#2 |
| 0.12.2 | Extended policy contradiction detection — IMPERATIVE_CUES covers real CLI forms; strong-trust override bypasses sim gate when belief trust ≥ 0.85 | F#4 |
| 0.12.3 | F#7 fix — StateStore syncs from disk on external writes; auto-ingest hook writes survive server saves | F#7 |
| 0.12.4 | `aether doctor` diagnostic command — six checks (imports, state file, activity, encoder, hook, mcp registration) that surface silent breakage in seconds | (catches future F#1-style bugs) |
| 0.12.5 | F#3 fix — `MemoryGraph.get_memory()` catches deserialization errors; read tools degrade to "unknown memory_id" instead of crashing on corrupt nodes | F#3 |
| 0.12.6 | F#8 fix — encoder warmup no longer hangs in MCP subprocess (HF env vars + redirect_stdout/stderr around SentenceTransformer init + diagnostic log) | F#8 |
| 0.12.7 | F#8 layer 1.5 — root cause was HF online check, not stdout; force offline mode when model is cached | (F#8 hardening) |
| 0.12.8 | F#9 + F#10 fix — search() weights combined score by trust; inject_substrate_context filters trust=0 entries | F#9, F#10 |
| 0.12.9 | Hygiene gap #1 + #2 — `AETHER_DISABLE_AUTOINGEST` env var + `redact_secrets()` regex layer (sk-, AKIA, ghp_, xox*, Stripe, bearer, PEM, password=…) before fact extraction | gap: opt-out + redaction |
| 0.12.10 | Hygiene gap #3 — rotating backups (`{state_dir}/backups/{stem}.{timestamp}.json`) + atomic write (`.tmp` + os.replace). `AETHER_BACKUP_KEEP=N`, `AETHER_DISABLE_BACKUPS=1`. New `_doctor_backups` check. | gap: backup safety |
| 0.12.11 | Hygiene gap #4 — `aether doctor --report` outputs a self-contained markdown bundle (env + checks + log tails) for one-paste GitHub issues. New `.github/ISSUE_TEMPLATE/bug_report.yml` form requires the bundle. | gap: useful bug reports |
| 0.12.12 | Hygiene gap #5 — `aether warmup` CLI eagerly pulls the encoder model with clear remediation messaging (HF Hub guidance, `[ml]` extra hint, cold-mode-fallback reassurance). Run after `pip install` to surface install issues immediately. | gap: install-resilience messaging |

E2E harness expanded from turns 1-2 → turns 1-10 (full scripted scenario). 9 full_loop tests + 4 smoke tests, all passing.

The auto-ingest Stop hook was fixed to actually read `transcript_path` (had been a silent no-op for 3 days because Claude Code sends a path, not inline messages). Hook now fires every turn and writes high-trust facts to the substrate. The substrate grew from 38 → 41+ memories during the session as a direct consequence.

### Findings status (all 10 closed)

| ID | Description | Disposition |
|---|---|---|
| F#1 | `[mcp]` install missing networkx → server crash on first tool call | Fixed v0.12.1 |
| F#2 | release.yml triggered on `release: published` not tag push → no PyPI publishes since 0.9.0 | Fixed v0.12.1 |
| F#3 | Read tools crash with TypeError on corrupt nodes (zombie `backfill`-shape) | Fixed v0.12.5 |
| F#4 | `aether_sanction` approves `git push --force` against "Never force-push" belief | Fixed v0.12.2 |
| F#5 | `aether_link("supports")` doesn't override existing `related_to` | Closed — test-author error in turn-8 e2e, not a bug |
| F#6 | Auto-link sim threshold leaves dissimilar facts disconnected | Closed — expected behavior, threshold is tunable |
| F#7 | Stop hook + MCP server share state file with no coordination → server clobbers hook writes | Fixed v0.12.3 |
| F#8 | `_LazyEncoder` warmup hangs in MCP subprocess (HF online check, stdio bleed) | Fixed v0.12.6 + v0.12.7 |
| F#9 | search() doesn't tiebreak by trust; inject hook doesn't filter trust=0 | Fixed v0.12.8 (subsumed by F#10) |
| F#10 | Cosine actively anti-ranks high-trust truths when annotation suffixes dilute their embedding (no trust term in score) | Fixed v0.12.8 |

## Strategic state

The substrate is now **structurally complete AND prod-ready for first-stranger trial**:

Architecture (closed by v0.12.0–v0.12.8):
- Auto-ingest fires after every turn → substrate grows on its own.
- `_sync_first` decorator ensures the server picks up external writes on the next tool call.
- `aether doctor` (now 7 checks) diagnoses install / hook / MCP / state-file / backup issues in seconds.
- All read tools degrade gracefully on corrupt nodes.
- All write tools sync from disk before mutating, so hook writes survive.
- Search ranking weights by trust so demoted entries don't outrank canonical truths.

Hygiene (closed by v0.12.9–v0.12.12):
- `AETHER_DISABLE_AUTOINGEST=1` pauses the hook without uninstalling.
- Common secrets (API keys, bearer tokens, PEM blocks, password=… forms) are redacted before reaching the substrate.
- Every save snapshots the prior state to `~/.aether/backups/` (5 most recent by default), atomic-writes via `.tmp` + `os.replace`.
- `aether doctor --report` produces a one-paste markdown bundle for GitHub issues, with a form-mode template that requires it.
- `aether warmup` surfaces install-time encoder failures with clear remediation messaging.

The only remaining gap is **N>1 user validation** — only solvable by getting an actual stranger to try this. Backup safety + opt-out + redaction + a one-paste bug-report path now make that meaningfully less scary as a first ask.

The remaining technical gap is **richness, not architecture**. With 127 memories the substrate can't ground much; the value compounds at 1000+. The fix for that is just time + sessions — which the auto-ingest hook now does for free as long as you're working.

### Update 2026-04-30 evening — substrate now at 127 memories

**One-time import from AI_round2 production substrate landed.** Pulled
from `D:/AI_round2/personal_agent/crt_facts.db` + `crt_episodic.db`:
- 46 unique slot/value beliefs deduped from 324 fact rows
- 4 active user preferences
- 17 concept entities (people, orgs, projects)
- 15 top behavioral patterns

OSS substrate: 45 → 127 nodes. Native = 45, imported = 82. Backups
written before each touch (`mcp_state.pre_ai_round2_import_*`,
`mcp_state.pre_embed_repair_*`, `mcp_state.pre_underscore_fix_*`).

**Slot-equality scan over the merged substrate revealed real
contradictions** — the v0.12 detector's first encounter with production
data:

| Slot | Distinct values | Notable |
|---|---|---|
| `user.occupation` | 15 | Mix of LLM hallucinations (`stocking`, `filmmaker`, `research engineer`) + user-trolling (`dork`, `fucking dork`) + truth (`freelance dev, sole crt builder`, `web developer`) |
| `entity.organization` | 10 | Includes garbage extractions (`like`, `both`, `myself`) — entity_extraction needs cleanup |
| `user.favorite_color` | 9 | All distinct (blue/brown/cyan/green/magenta/orange/purple/red/yellow). Most user_stated trust 0.95 — real evolution, not error |
| `user.name` | 9 | Nick (truth) + Nick Block + LLM hallucinations (Aether, Claude, Jake, Marcus, October Baby, Turbo, World) |
| `user.employer` | 6 | Amazon, Anthropic, CRT/Aether, Google, "left:a design studio", Walmart |
| `user.location` | 4 | Milwaukee + Seattle + Portland — multi-residence reality |
| `entity.person` | 4 | Includes garbage (`at`, `just`) |
| `entity.project` | 3 | All garbage extractions ("project this/new/approach") |
| `user.age` | 2 | 34 and 4 — clear LLM hallucination |

This is the v0.12 slot-equality detector's strongest empirical
demonstration to date. All these conflicts are *latent* in the
substrate; running the detector against the merged state surfaces them
as actionable.

OSS remains the main focus. Today's marathon proves the substrate-assisted dev loop is real: every shipped release was validated by Aether's own tools (sanction APPROVE for cleanup, search for prior context, fidelity for grounding, doctor for health). The README's "I built a substrate that caught itself shipping bugs" thesis got 5 fresh data points today.

## What we're working on next

The README was rewritten in commit `a84774c` with the concrete-hook + quickstart-cookbook structure. That priority is closed.

Three plausible next threads, in priority order:

### 1. Validation chapter, test #1: fresh-session-no-context (highest external leverage)

The substrate is now stable enough to collect external evidence rather than dogfood-only. Test #1 is the cheapest of the four NEXT_SESSION calls out: spin up a clean Claude session with no context injection, ask 5–10 standard questions, see what the substrate surfaces / blocks vs the no-substrate baseline. Concrete data → blog material → first non-dogfood validation point.

The other three tests (N>1 user, cross-vendor, scale to 10k) are more involved. Test #1 unblocks them by establishing a methodology.

### 2. Slot extractor name patterns

OSS extractor returns empty for "My name is Nick" / "I am Nick" patterns; production CRT layers have `user.name` slots. Adding a name pattern bank (~30 lines) closes the gap and lets the slot-equality detector catch the canonical Nick↔Aether case the v0.12 audit cited. The merged substrate already has 9 distinct `user.name` values waiting for it to fire.

### 3. Resolve imported slot conflicts (substrate hygiene)

The merged substrate has 15 `user.occupation` values, 9 names, 9 favorite colors with the LLM hallucinations mixed in. Now that F#10 ranks them correctly, an `aether_correct` sweep is straightforward — demote the obvious garbage, promote the real ones. Cleaner demo substrate for blog screenshots / validation chapter. Could do it from the CLI in one pass.

### 4. Blog post: "I built a substrate that caught itself shipping bugs"

Today's twelve-release marathon adds 5 fresh dogfood data points to the existing list. README rewrite + the prod-readiness gap closure narrative is writable now. High external leverage.

## Open findings

### Open (surfaced 2026-04-30 by validation chapter test #1, deferred)

The validation harness (`bench/validation_test1.py`) snapshotted substrate behavior against a 10-question battery. The first warm run surfaced three real findings:

- **F#11 — sanction has no default policy beliefs.** `aether_sanction("git push --force origin main")` returns APPROVE on a fresh substrate because F#4's policy contradiction detection requires an explicit "never force-push" belief to be present. Same for "drop the production database." Fix candidates: (a) `aether init` seeds default policy beliefs (force-push, prod-db-drops, --no-verify, etc.), (b) sanction adds structural detection of high-risk language even without belief presence. Option (a) is simpler; option (b) is more general. Visible in the warm reference snapshot at `bench/results/validation_test1_2026-04-30_warm.md` categories C1 + C2.
- **F#12 — trust-vs-verbosity ranking gap.** F#10's `SEARCH_TRUST_WEIGHT=0.7` keeps trust=0.67 demoted entries below trust=0.95 truths, but a trust=0.90 *short-text* entry can outrank a trust=0.95 *verbose-text* entry because cosine similarity to the bare query favors shorter text. Visible in category B2 of the warm snapshot: `user name: Jake` (trust=0.90) ranked above `user name: Nick (observed 65x in production)` (trust=0.95). Fix candidates: (a) re-embed imported nodes after stripping the "(observed Nx)" annotation suffix, (b) raise the trust weight slope. (a) is a one-time data migration; (b) has knock-on effects.
- **F#13 — cold-query inject leak.** Library-level `compute_grounding` is correct (returns empty support+contradict for "capital of France" via the embedding floor in the meter), but the AI_round2 project-level inject hook (`.aether-hooks/inject_substrate_context.py`) uses a 0.15 score threshold that lets weak-similarity matches through (e.g. "user name: Claude" at score 0.158 for query "capital of France"). **Attempted v0.12.13 fix at the OSS layer: reverted** — adding a `GROUNDING_MIN_SIMILARITY=0.30` floor regressed the v0.9.3 methodological-overclaim test, because methodological signals also live at sim ~0.25. The right fix is at the inject layer (project-level), not the library layer. Approach: bump inject's min_score to 0.20 OR add `min_similarity=0.30` AND `min_score=0.15` in the inject filter only.

### Closed earlier this session

- **F#8 (FIXED v0.12.6 + v0.12.7):** `_LazyEncoder` warmup thread silently hung in MCP-subprocess context. Three-layer fix in v0.12.6 (HF env vars, redirect_stdout/stderr, widened except + diagnostic log). v0.12.7 found the deeper root cause was HF Hub's online connectivity check, not stdout — added force-offline-when-cached. Verified live 2026-04-30 evening: encoder warmed in ~18s, 127/129 substrate nodes have embeddings, doctor reports `[OK] encoder`.
- **F#9 (FIXED v0.12.8):** trust didn't break ties in search ranking; trust=0 demoted entries leaked into the inject hook's LLM context. Subsumed by the F#10 fix below — both parts addressed.
- **F#10 (FIXED v0.12.8):** discovered while verifying F#8 against the 127-memory merged substrate. `aether_search "user favorite color"` ranked the demoted `red` (trust=0.67, sim=0.904) *above* all 7 trust=0.95 truths because their `(observed Nx in production)` annotation suffixes diluted their embeddings. Score function `0.7*sim + 0.3*substring` had no trust term — even tiebreak fixes (F#9 part a) wouldn't have helped because the scores weren't tied, cosine was actively anti-ranking the truths. Fix: new `SEARCH_TRUST_WEIGHT = 0.7` constant; `combined *= (1-w) + w*trust` so trust=0 drops to 0.3× score (still surfaces) and trust=1 leaves it unchanged. Also F#9(b): inject_substrate_context.py drops trust≤0 entries from the prompt. After fix: red drops from #1 to #9; all trust=0.95 truths rank above it. Four regression tests in `test_v127_search_trust_weight.py`.

## Other parking-lot items

- **Resolve the imported slot conflicts.** The 9 distinct `user.favorite_color` values, 15 occupations, etc. are now in the substrate as live slot-tagged memories. A pass through `aether_resolve` (or batch via `aether_correct` to demote LLM-stated low-trust ones) cleans up the production noise that was inherited. Could also be the validation-chapter material — substrate doesn't just store contradictions, it *resolves* them.
- **Entity-extraction cleanup.** The imported `entity.person` slot has `at` and `just` as values; `entity.project` has all garbage; `entity.organization` has `like`/`both`/`myself`. The original CRT entity extraction needs a quality pass before any future merge. Could be filtered out post-hoc with a stop-word list.
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

# Pre-flight encoder check (v0.12.12)
aether warmup                                      # eager model pull, surfaces install issues

# Verify latest publish
pip install --upgrade aether-core
python -c "import aether; print(aether.__version__)"   # should print 0.12.12

# One-paste bug-report bundle (v0.12.11)
aether doctor --report                             # markdown for GitHub issues
```

## First moves for next session

1. **Run `aether doctor`** to confirm all 7 checks are OK. If anything is FAIL or WARN, address before continuing — `aether doctor --report` produces a one-paste bundle if you need to file an issue.
2. **Pick one of the four threads above.** Validation chapter test #1 is the highest external leverage now that the README + prod-readiness work is shipped.
3. **Check `aether_search`** for memories the auto-ingest captured since last connect. The substrate should be richer than it was — every working session contributes for free.

## What stays closed

- `electron/` — agent product
- `dnnt/` — training pipeline (model-agnostic conflict)
- Cloud features, multi-user, billing
- Specific Claude Code dispatch / action execution

---

*Today shipped twelve PyPI releases. The first eight closed all original findings (F#1–F#10) and made the substrate structurally complete. The last four (v0.12.9–v0.12.12) closed five prod-readiness hygiene gaps surfaced by an explicit "ready for real users" audit: opt-out env var, secret redaction, rotating backups + atomic write, one-paste bug-report bundle + GitHub issue template, and an `aether warmup` CLI for install-time encoder failures. After that, validation chapter test #1 shipped (`bench/validation_test1.py`) and surfaced three new findings on first contact (F#11, F#12, F#13) — exactly the substrate-caught-itself loop the README promises. Strategic context (OSS-as-main-focus) unchanged. The remaining work: close F#11–F#13 next session and ship N>1 user validation — the only gap pure code cannot solve.*
