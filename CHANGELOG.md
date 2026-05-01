# Changelog

What's shipped, in reverse chronological order. For what's coming next, see [ROADMAP.md](ROADMAP.md).

## Shipped (v0.12.14) — Track 1: layman onboarding

`claude plugin install github.com/blockhead22/aether-core` is now genuinely one command. The SessionStart hook auto-installs aether-core if missing, auto-upgrades it if it's older than the floor the hook needs, kicks off encoder warmup in the background, ensures `~/.aether/mcp_state.json` exists with default policy beliefs seeded, and emits a first-run welcome message via the `additionalContext` protocol so the user sees aether is active in their first turn.

Plus:
- **`aether status` and `aether doctor` flag version drift** — when the installed `aether-core` is behind the latest on PyPI, both commands print a one-line "update available" notice with the upgrade command. Cached for 24h at `~/.aether/.pypi_version_cache.json`. Skip with `AETHER_NO_UPDATE_CHECK=1`.
- **`aether uninstall-cleanup`** — explicit, dry-run-by-default way to remove `~/.aether/` after `claude plugin uninstall`. Pass `--yes` to actually remove. Pass `--keep-substrate` to preserve `mcp_state.json` + backups while clearing logs / markers / caches.
- **New `hooks/session_start.py`** replaces the inline pip-install one-liner. Fail-soft: errors go to `~/.aether/session_start.log`, the hook never blocks Claude even if aether install fails.
- 15 regression tests in `test_v1214_layman_onboarding.py`.

Track 1 #3 (visible aether activity) and #5 (landing page) deferred to a future release.

## Shipped (v0.12.13) — F#11 fix

`aether init` now seeds 7 default policy beliefs (force-push, --no-verify, production data safety, rm -rf) so `aether_sanction` has something to gate against from day one. Skip with `--no-defaults`. F#4's policy contradiction detection now fires on a fresh substrate instead of returning APPROVE for everything.

Implementation: the seed function blocks for up to 60s on encoder warmup before writing, so the seeded memories get real embeddings (not None). 6 regression tests cover seeding, idempotency, the no-defaults flag, source-tag stamping, and end-to-end sanction blocking.

## Shipped (v0.12.12) — Hygiene gap #5: install resilience messaging

`aether warmup` CLI eagerly pulls the encoder model and reports install failures with clear remediation messaging (HF Hub guidance, `[ml]` extra hint, cold-mode-fallback reassurance). Run after `pip install` to surface install issues immediately instead of silently leaving the substrate in cold mode.

7 regression tests covering success path, unavailable-encoder path, timeout path, and CLI wiring.

## Shipped (v0.12.11) — Hygiene gap #4: useful bug reports

`aether doctor --report` outputs a self-contained markdown bundle (env + checks + log tails) for one-paste GitHub issue submissions. New `.github/ISSUE_TEMPLATE/bug_report.yml` form requires the bundle. Privacy: state file contents are NEVER included; only metadata (size, node count, paths) reaches the bundle.

10 regression tests covering bundle shape, redaction contract, log-tail handling, and end-to-end inclusion of available logs.

## Shipped (v0.12.10) — Hygiene gap #3: backup safety

Every `MemoryGraph.save()` snapshots the previous state file to `{state_dir}/backups/{stem}.{timestamp}.json` before overwriting, then atomic-writes via `.tmp` + `os.replace`. Default keep depth is 5 most recent rotations; override with `AETHER_BACKUP_KEEP=N` (`=0` disables). `AETHER_DISABLE_BACKUPS=1` skips rotation entirely. New `_doctor_backups` health check.

11 regression tests covering rotation, atomic write, env var handling, and helper invariants.

## Shipped (v0.12.9) — Hygiene gaps #1 + #2: opt-out + secret redaction

`AETHER_DISABLE_AUTOINGEST=1` short-circuits `extract_facts` and `ingest_turn` immediately. Hook stays installed; nothing is written. Honored both at hook entry and inside the library function (defense in depth).

`redact_secrets()` runs before pattern matching. Replaces with `[REDACTED]`: API-key shapes (`sk-`, `AKIA`, `ghp_`, `github_pat_`, `xox[abprs]-`, Stripe live/test), bearer tokens, PEM private-key blocks, and explicit `password=` / `token=` / `api_key=` / `secret=` key-value forms. Conservative on purpose — emails and phone numbers are not matched because they're usually legitimate context.

27 regression tests covering each pattern, idempotency, env var truthy/falsy values, and the end-to-end invariant that secrets in input never reach candidate fact text.

## Shipped (v0.12.8) — F#9 + F#10 fix

`search()` now weights combined score by trust. New `SEARCH_TRUST_WEIGHT = 0.7` constant; `combined *= (1 - w) + w * trust` so trust=0 drops to 0.3× score (still surfaces) and trust=1 leaves it unchanged. Closes the case where a corrected-down memory was outranking the canonical truth because the score function had no trust term — the substrate's own search caught the substrate's own scoring bug.

`inject_substrate_context.py` (project-level hook) now drops trust ≤ 0 entries from the LLM prompt so demoted memories don't leak.

Four regression tests in `test_v127_search_trust_weight.py`.

## Shipped (v0.12.7) — F#8 hardening

`_LazyEncoder` warmup hang root cause was HuggingFace Hub's online connectivity check, not stdout. Force offline mode when the model is cached so subsequent loads skip the check entirely.

## Shipped (v0.12.6) — F#8 fix (initial)

Encoder warmup no longer hangs in MCP-subprocess context. Three-layer fix: HF env vars at module load, `redirect_stdout/stderr` around the SentenceTransformer construction, widened except to BaseException. Diagnostic log at `~/.aether/encoder_warmup.log`.

## Shipped (v0.12.5) — F#3 fix

`MemoryGraph.get_memory()` catches deserialization errors. Read tools (`aether_memory_detail`, `aether_lineage`, `aether_cascade_preview`) degrade to "unknown memory_id" instead of crashing on corrupt nodes.

## Shipped (v0.12.4) — `aether doctor`

New diagnostic command. Six checks (imports, state file, activity, encoder, hook, MCP registration) that surface silent breakage in seconds. `aether doctor --format json` for scripts.

## Shipped (v0.12.3) — F#7 fix

`StateStore` syncs from disk on external writes via the new `_sync_first` decorator. The auto-ingest Stop hook's writes survive server saves instead of being silently clobbered.

## Shipped (v0.12.2) — F#4 fix

Extended policy contradiction detection — `IMPERATIVE_CUES` covers real CLI forms (`git push --force`, `rm -rf`, etc.). Strong-trust override bypasses the similarity gate when belief trust ≥ 0.85.

## Shipped (v0.12.1) — release pipeline + carry-overs

E2E harness scaffold + v0.9.6 carry-overs: link helper, version stamp on save, `[mcp]` extra includes networkx, `release.yml` triggers on tag push (was `release: published`, blocking PyPI publishes since v0.9.0).

## Shipped (v0.12.0)

### Slot-equality detector + shape local-context gate

Two parallel fixes that came directly out of Lab A v2's production-substrate
audit (`bench/lab_a_v2_production_substrate_findings.md`). The audit walked
324 facts and 94 supersessions in the production CRT substrate and found
that **v0.11's detection layer caught 0 of 42 real categorical
contradictions** on slots like `user.name`, `user.favorite_color`,
`user.location`. Production had been doing the contradiction work manually
via supersession because the detection layer was blind to slot-equality
conflicts.

#### Fix A — slot-equality detector

New primitive in `aether/patterns.py`:

```python
slot_equality(tags_a, tags_b) -> MatchResult
```

The simplest possible contradiction class: two memories tag the same
`slot:KEY=VALUE` with different categorical values. Score 1.0 on
disagreement, 0.5 on agreement, 0.0 on no shared slots. Cold-mode safe,
case-insensitive, whitespace-stripped equality. The slot taxonomy is
**substrate-resident** — every new slot the extractor recognizes becomes
a contradiction-detection point automatically (no hardcoded registry).

Wired into both contradiction paths:

- `_detect_and_record_tensions` (write): runs as a 6th independent signal
  alongside slot_conflict, asymm_neg, policy, mutex, shape. Fires with
  `kind="slot_value_conflict"` and `nli_score=0.9` (high confidence —
  categorical conflict on a known slot is decisive).
- `compute_grounding` (read): extracts slots from draft text on the fly
  via `extract_fact_slots`, compares against candidate memory tags.

**Slot-keyed candidate pre-screen**: production texts often have low
textual overlap (e.g. "remember my name is Nick" vs "your name is Aether"
~ 0.14 Jaccard, below the 0.2 candidate gate). v0.12 augments the top-K
candidate set with any memory sharing a slot key with the new memory, and
bypasses the sim gate for slot-shared candidates. Without this the
detector never sees the pair.

#### Fix B — shape primitive local-context gate

The substrate-assisted dev loop caught a real v0.11 production bug. When
the v0.12 design first went up for sanction, the call REJECTED — and the
reason wasn't a process failure, it was a true positive: the action text
mentioned "v0.12" and a Lab A v2 memory mentioned "v0.11", so v0.11's
shape primitive registered them as a version conflict. Same shape,
unrelated topics, false positive.

Fix: shape conflicts now require the **immediate surrounding tokens** to
overlap before treating differing typed values as contradicting:

- `LOCAL_CONTEXT_TOKENS = 3` — 3 tokens before and after the typed value
- `LOCAL_CONTEXT_MIN_OVERLAP = 0.30` — Jaccard threshold

"Python 3.10 or higher" vs "Python 3.8" share `python` in local context →
real conflict, fires at score 1.0. "ship v0.12 today" vs "v0.11 was last
week" don't share local context around the version → suppressed. Rejected
conflicts are still visible in `evidence["suppressed"]` for debugging.

#### Auto slot extraction in add_memory

When `slots=None` (the default), `add_memory` now runs `extract_fact_slots`
on the text and tags the memory with whatever the extractor recognizes
(`employer`, `location`, etc.). Without this, memories written through
normal channels carry no slot tags and the detector has nothing to compare.

#### Bench impact

| Mode | v0.11.0 | v0.12.0 |
|---|---|---|
| Warm | 29/29 blocker pass (100%) | **29/29 blocker pass (100%)** |
| Cold | 19/29 (65.5%) | **21/29 (72.4%)** |

No warm regressions. Cold gained +2 categories. The Lab A v2 production
case (Nick→Aether, blue→orange, etc.) is now caught — verified by the
6 write-path slot-equality tests.

**323 tests pass** (was 300). 23 new in
`tests/test_v120_slot_equality_and_local_context.py` covering the
primitive in isolation, write-path integration across the textual-
similarity gradient, read-path integration via compute_grounding, the
sanction-self-reject regression, and the both-fixes-compose case.

#### Meta-finding

The substrate caught its own bug. v0.11.0 shipped with a shape false
positive that nobody saw. The substrate-assisted dev loop surfaced it
the first time the new detection touched a real audit memory — the
sanction REJECT verdict WAS the bug report. v0.12 closes both the gap
the audit found AND the bug the audit triggered.

## Shipped (v0.11.0)

### Composable pattern primitives + the quantitative known-gap closes

Up through v0.10.x every contradiction detector was hand-coded regex.
That worked for narrow stable patterns but left visible gaps:

- The `known_gap_quantitative` category in the v0.9.4 calibration bench
  documented three failing cases (Python 3.10 vs 3.8, 222 vs 99 tests,
  2026-04-27 vs 2025-01-15) the slot extractor couldn't catch because
  they're not categorical conflicts.
- Cold-mode policy / negation_asymmetry stuck at 0% because the gates
  needed embedding similarity.
- New pattern types required code changes; no composability.

v0.11 adds a **composable primitives module** (`aether/patterns.py`)
that supplements regex with four building blocks:

1. **`token_overlap`** — Jaccard on token sets. Symmetric, bounded,
   cold-mode safe. Useful as a topical-relevance gate.
2. **`shape`** — typed-pattern detection. Recognizes versions
   (`3.10`, `3.8`), dates (`2026-04-27`), integers (`222`, `99`),
   floats. Ships with comparators per type:
   `numeric_tuple`, `chronological`, `magnitude`, `categorical`. Returns
   `score=1.0` on conflict, `0.5` on agreement, `0.0` on no shared
   shape. **This is what closes `known_gap_quantitative`** — and it
   works without embeddings.
3. **`substring_window`** — multiple substrings within N tokens.
   Useful for catching multi-clause patterns like inference + claim
   in the same sentence.
4. **`ncd`** — normalized compression distance via gzip. The
   "gzip beats BERT" approach as a substrate-resident similarity
   primitive. No model, no training.

Each primitive returns a `MatchResult` with `score` and `evidence`.
A `combined_score()` helper ensembles results with weighted average.

**Wiring**: shape detection is now a fifth contradiction signal
alongside slot conflict, asymmetric negation, policy, and mutex.
Fires in both `_detect_and_record_tensions` (write path) and
`compute_grounding` (read path). Conflicting `kind="quantitative"`
edges are added with rule trace `shape:version:3.10<>3.8` etc.

**Calibration bench impact**:

| Mode | Before (v0.10.1) | After (v0.11.0) |
|---|---|---|
| Warm | 26/26 blocker pass (3 known_gap fail) | **29/29 blocker pass, no remaining known gaps** |
| Cold | 16/26 (61.5%) | **19/29 (65.5%)**; quantitative now passes cold too |

**factual_contradiction in cold mode** went 0% → 60% (the three
quantitative cases now fire without needing embeddings). Other
cold-mode gaps (policy, negation_asymmetry — which still need
embedding-similarity gates) deferred to a future release that
similarly replaces those gates with regex-free primitives.

**300 tests pass** (was 268). 32 new in
`tests/test_v110_pattern_primitives.py` covering each primitive
in isolation, comparator behavior per shape type, integration on
both write and read paths, cold-mode safety, and false-positive
guards (no shape → no firing; same value → agreement, not conflict).

### Lab framing recorded for v0.11+ research

Tonight's research direction synthesized into one thesis: **meaning
emerges from the substrate's contradiction structure and lived
trajectory, not from labels or training.** Five interlocking
ideas saved as substrate memory `m1777362414276_1` (trust 0.9):

1. **Slot induction from contradiction structure** — slots aren't
   pre-defined; they emerge as positions where memories disagree.
   Walk CONTRADICTS edges, extract `(frame, filler_pair)`, cluster
   frames, induce comparison functions per cluster.
2. **Semantic gravity per token** — tokens have evolving multi-
   dimensional state (mass, charge, position vector, velocity).
   Meaning is the *shape* of a token's gravity field, not its
   position.
3. **Gravity wells** — depressions in semantic space where many
   other tokens get pulled in. Distinct from mass.
4. **Contradictions clustering into meta-beliefs** — when
   contradictions concentrate around a topic, the fact-of-
   contradiction becomes a belief: "this is contested territory."
5. **Self-constitution as trajectory** — substrate identity isn't
   its current state, it's the gravity field's trajectory through
   semantic space. *You're not who you remember; you're how your
   meanings have moved.*

These are the v0.11+ research direction. Each maps to a small
experiment that uses substrate-resident state in ways no other
mainstream pattern matcher can. The composable primitives in
`aether/patterns.py` are the foundation layer; learned-pattern
primitives mined from contradictions are the next layer up.

## Shipped (v0.10.1)

### `aether_path` was crashing in production -- the substrate caught it on first use

The first end-to-end test under v0.10.0 (substrate-assisted dev loop on
v0.10.1 design work) called `aether_path` to preload context. The tool
returned:

```
Error: MemoryNode.__init__() missing 3 required positional arguments:
'memory_id', 'text', 'created_at'
```

Root cause: a save/load asymmetry in `aether/memory/graph.py`. In `save()`,
edges were serialized as:

```python
data["edges"].append({
    "source": source_id,    # endpoint, set first
    "target": target_id,
    **clean_data,           # <-- shadow-overrides "source" if metadata had it
})
```

`backfill_edges` (v0.9.5) included `"source": "backfill"` in its auto-link
metadata. The `**clean_data` spread overrode the endpoint `"source"` key.
The JSON wrote `source="backfill"` instead of the real `memory_id`. On
`load()`, networkx's `add_edge("backfill", target_id, ...)` auto-created a
stub node with id `"backfill"` and no required fields. Subsequent
`get_memory()` on that stub crashed with the AttributeError. The stub
sat in production for ~12 hours; aether_path was effectively broken on
any substrate that ran `backfill_edges` -- which is the documented
recovery tool from v0.9.1.

The unit tests didn't catch it because every persistence test used the
public API to build edges and never round-tripped metadata containing
keys that shadow the JSON edge schema.

Fix (three-part defense):

1. **`save()` puts endpoints AFTER `**clean_data`** so endpoints always win
   (the colliding metadata field is silently dropped on save -- documented
   contract).
2. **`load()` skips edges with unknown endpoints** -- defense in depth for
   any pre-v0.10.1 substrates loaded by v0.10.1+ runtimes; prevents stub
   nodes from forming.
3. **`backfill_edges` renames metadata key `source` -> `origin`** to avoid
   the collision entirely going forward.

Production substrate at `~/.aether/mcp_state.json` was repaired in
parallel: backed up to `mcp_state.broken_pre_v0101_<ts>.json`, then
the orphan `"backfill"` stub node and the two corrupt edges referencing
it were removed (27 -> 26 nodes, 6 -> 4 edges).

**Tests: 268 pass (was 260). 8 new tests in
`tests/test_v101_save_load_metadata_collision.py`** covering: edge
metadata key collision (source/target), absence of stub nodes after
round-trip, `get_memory` working on every node, `load()` defense in
depth on corrupted substrates, and `backfill_edges` using the renamed
`origin` key.

### Meta-finding: the substrate-assisted dev loop demonstrated itself

This bug was caught by the substrate, not by tests. Specifically: the
fresh-session agent investigating the v0.10.1 design called `aether_path`
to preload context, the tool crashed, the agent flagged it correctly,
fell back to source-of-truth grep, and the diagnosis surfaced before any
code change. The "query the source of truth, don't infer" rule from
session 1's feedback file propagated across five+ session restarts and
caught a production bug in the marquee tool. This is the validation
chapter's strongest data point yet -- the loop catches what synthetic
tests miss, and the engineering response to its findings produces real
patches with regression discipline.

Deferred to v0.10.2: the auto-link substring formula divergence
(v0.9.6 carry-over). Separable concern, cleaner commit narrative.

## Shipped (v0.10.0)

### Action receipts — the audit half of the governance loop

The first port from the main repo (`personal_agent/action_receipts.py`)
to OSS under the new OSS-focus pivot. Up through v0.9.5, OSS had the
*gate* (aether_sanction returns APPROVE / HOLD / REJECT) but no
*audit trail* of what actually executed and what its outcome was. That
left the governance loop half-open: pre-action verdicts but no
post-action verification.

**Closes the loop**:

1. `aether_sanction` now opens an `ActionReceipt` and returns its
   `action_id` in the response. The receipt is created with the
   sanction verdict, the supporting / contradicting / methodological
   memory IDs that informed it, but no outcome yet.
2. The caller (agent / tool / human) cites that `action_id` in
   `aether_receipt(action_id, result, ...)` after executing or
   skipping the action. The receipt fills in `tool_name`, `target`,
   `result` (success / error / partial / skipped), `details`,
   verification status, and optional model attribution.
3. `aether_receipts` lists receipts (filterable by result, verdict,
   or only-open) newest-first. `aether_receipt_detail` returns one
   full record. `aether_receipt_summary` aggregates counts of
   verdicts, outcomes, open receipts, and verification pass rate.

**Persistence**: JSON side-car at `<state_path>_receipts.json`,
consistent with the existing `_trust_history.json` pattern. Round-trips
across StateStore restarts.

**Open / closed signal**: `open_receipts > 0` in the summary means the
agent is sanctioning actions but not closing the loop — a real signal
worth surfacing in the next session-start brief.

**Ported from main, tightened for OSS**:

- Dropped: SQLite (use the existing JSON state pattern instead),
  thread_id, agent_name, orchestration_id, run_step_id,
  expectation_keywords (all personal_agent-specific).
- Kept: receipt_id, timestamp, action, sanction_verdict, tool_name,
  target, result, reversible, reverse_action, details,
  verification_passed, verification_reason, model_attribution,
  completed_at, sanction_memory_ids.

22 new tests in `tests/test_v100_action_receipts.py`. **260 tests
total** (was 238). MCP surface grew from 16 tools to 20:
- `aether_sanction` — gains `action_id` in response (backward-compatible
  additive)
- `aether_receipt` — record outcome after execution (NEW)
- `aether_receipts` — list / filter receipts (NEW)
- `aether_receipt_detail` — one full record (NEW)
- `aether_receipt_summary` — aggregate stats (NEW)

**This is the first ship under the OSS-focus pivot.** The strategic
context: aether-core is now the primary track for thesis-foundational
mechanisms; main repo continues as research and Electron-app workshop.
v0.10 is the start of porting the parts of main that belong in OSS:
runtime governance loop (action receipts here), runtime belief/speech
gap detector (belief_speech_engine, future v0.10.x), lightweight
self-model + reflection primitives (future). Closed-side stays closed:
DNNT training pipeline, Electron app, multi-user / hosted features.

## Shipped (v0.9.5)

### Governance works in cold-encoder mode (the v0.9.4 production miss)

The first end-to-end re-run of the calibration rubric (2026-04-28
late-night) confirmed v0.9.2 (no-hang) but flagged that v0.9.3
(methodological detection) and v0.9.1 (auto-link RELATED_TO) BOTH
failed in production despite passing 100% in the v0.9.4 bench.

Root cause: `aether/contradiction/tension.py:_encode` handled
`self._encoder is None` (no encoder injected) but NOT the case where
the encoder is present but its `encode()` method returns `None`
(LazyEncoder's contract when not yet warm). When that happened,
`_compute_similarity` did `vector.size` on `None` →
`AttributeError` → the `try/except: continue` in BOTH
`compute_grounding` and `_detect_and_record_tensions` swallowed the
raise and silently skipped the entire loop body, including the
methodological-overclaim check (v0.9.3), the auto-link RELATED_TO
logic (v0.9.1), the contradiction-on-write logic (v0.5+), and even
mutex contradiction routing.

The bench passed in v0.9.4 because every test fixture forces
synchronous `s._encoder._load()` before running. Same meta-pattern
as v0.9.0: synthetic tests bypassed the production code path.

Fixes:

1. **`tension.py:_encode` now handles `None` returns from a non-blocking
   encoder.** Five-line patch: convert `None` to `np.array([])` so
   downstream size checks work without raising.

2. **Adaptive `AUTO_LINK_THRESHOLD`.** The original 0.7 was tuned for
   embedding cosine; Jaccard rarely hits 0.7 even on clearly-related
   text. v0.9.5 introduces `AUTO_LINK_THRESHOLD_SUBSTRING` (default
   0.4) used when the encoder is cold, picked via
   `self._encoder.is_loaded` rather than the unreliable
   `embedding_similarity is not None` check (the meter sets
   `embedding_similarity: 0.0` even cold). Both `_detect_and_record_tensions`
   and `backfill_edges` use the adaptive threshold.

3. **Adaptive `GROUNDING_MIN_SCORE`.** The 0.15 floor was tuned for
   embedding combined-score; substring scores are typically lower.
   v0.9.5 introduces `GROUNDING_MIN_SCORE_SUBSTRING` (0.10) used in
   cold mode so clearly-related memories at substring score 0.10-0.15
   still surface for the methodological check.

4. **Bench runs in cold mode too.** New `--cold-encoder` flag on
   `run_fidelity_bench` swaps in a never-warmed `LazyEncoder`. New
   pytest class `TestFidelityCalibrationColdMode` enforces the cold-mode
   baseline as part of the regular suite. Future regressions in cold-
   mode behavior will fail CI.

**Cold-mode baseline numbers** established by this release:

| Category | Warm | Cold |
|---|---|---|
| mutex_contradiction | 100% | 100% |
| methodological_overclaim | 100% | 80% (4/5) |
| false_positive_guard | 100% | 100% |
| no_issue_unrelated | 100% | 100% |
| factual_contradiction | 100% | 0% — needs slot extraction (embedding-gated) |
| policy_violation | 100% | 0% — `embedding_similarity >= 0.45` gate |
| negation_asymmetry | 100% | 0% — same gate |
| no_issue_grounded | 100% | 25% — DUPLICATE/REFINEMENT classification needs embeddings |

Some cold-mode gaps are inherent (slot extraction without embedding
fallback can't classify paraphrases as DUPLICATE). Some are fixable
in future work (lower the policy / negation embedding-similarity
gate, add a Jaccard-based pathway). The bench tracks both modes so
future improvements show up as numbers.

Tests: 238 pass (was 227). 8 new tests in
`tests/test_v095_cold_encoder_path.py` directly exercising the
production cold-start code path. 3 new bench assertions in
`tests/test_v094_fidelity_calibration.py::TestFidelityCalibrationColdMode`.

This closes the v0.9.4 production miss the agent's re-run found.

## Shipped (v0.9.4)

### Fidelity calibration bench — measurable governance quality

Layer 3 of the governance work. Until now, "fidelity catches the right
things" was an anecdote: the v0.9.1 end-to-end test missed a
methodological overclaim, v0.9.3 fixed it, and that was the only data
point. v0.9.4 turns this into a measurable property.

**Corpus** (`bench/fidelity_corpus.json`): 29 hand-curated cases across
9 categories — factual contradictions, mutex contradictions, methodological
overclaims, policy violations, negation-asymmetry, no-issue grounded
claims, no-issue unrelated claims, false-positive guards, and known-gap
quantitative cases. Each case is a `(substrate, claim, expected)` triple
with constraints on which channels should fire and what verdict
sanction should return.

**Runner** (`bench/run_fidelity_bench.py`): builds a fresh substrate per
case, seeds memories, runs `compute_grounding` and (when expected)
`aether_sanction`, grades against the constraints. Outputs a
markdown report with per-category pass rates plus a list of failing
cases with diagnostic detail. Pre-warms the encoder once so per-case
overhead is sub-100ms.

**Pytest integration** (`tests/test_v094_fidelity_calibration.py`): five
assertions enforced as part of the regular suite — every blocker
category passes, blocker rate is exactly 100%, methodological recall is
100% (Layer 2 regression guard), false-positive guards hold, corpus has
coverage of every required category. So `pytest tests/` re-runs the
bench every time. Regressions surface with per-case breakdowns.

**Known-gap workflow**: categories prefixed `known_gap_` are tracked but
don't trigger non-zero exit. This lets the bench be honest (visible
failures, no hiding) and useful as CI (won't break for known
limitations). When a fix lands, the case moves out of `known_gap_` and
becomes a regression test.

**Baseline numbers** (this release):

| Category | Rate |
|---|---|
| factual_contradiction | 100% (2/2) |
| mutex_contradiction | 100% (3/3) |
| negation_asymmetry | 100% (2/2) |
| policy_violation | 100% (2/2) |
| methodological_overclaim | 100% (5/5) |
| no_issue_grounded | 100% (4/4) |
| no_issue_unrelated | 100% (3/3) |
| false_positive_guard | 100% (5/5) |
| known_gap_quantitative | 0% (0/3) — tracked, not blocking |

Blocker pass rate: **26/26 (100%)**. Three known-gap cases document
that the StructuralTensionMeter is built for categorical slot conflicts
(Seattle vs Portland) and does NOT catch quantitative / version /
date conflicts (Python 3.10 vs 3.8, 222 vs 99 tests, dates). Future
work can close that gap; the bench will report when it does.

227 tests pass (was 222). 5 new tests in
`tests/test_v094_fidelity_calibration.py`. 29-case corpus + runner in
`bench/`.

## Shipped (v0.9.3)

### Fidelity catches methodological overclaims, not just factual contradictions

The first end-to-end substrate-assisted dev test (2026-04-28) seeded a
high-trust memory: "the v1-vs-v3 CogniMap conclusion is confounded; the
'CALIC is bad' takeaway is unsupported." The agent then drafted "v3 was
worse than v1, so CALIC is bad" and called `aether_fidelity`. The tool
returned `gap_score: 0.0, action: PASS, supporting_memories: [],
contradicting_memories: []`. It missed entirely.

Diagnosis: the existing contradiction detection (StructuralTensionMeter
+ asymmetric-negation + policy + mutex) is wired around *factual* slot
clashes — Seattle vs Portland, AWS vs GCP. The methodological-gap
memory has different slots than the draft (no slot conflict), so the
meter classified them as unrelated and the memory dropped out. Same
class of miss as the v0.9.0 `aether_path` no-op: the substrate had the
right knowledge; the verdict-producing path didn't ask the right question.

Fix:

1. **Two new helpers in `aether/mcp/state.py`**:
   - `_has_inference_marker(text)` — recognizes draft inference markers
     (`so X`, `therefore Y`, `thus`, `means that`, `proves`, `implies`,
     `because`, `since`, etc.). Conservative — leading whitespace
     required so substrings inside larger words don't false-match.
   - `_has_methodological_signal(memory_text, source)` — recognizes
     methodological-warning language (`unsupported`, `doesn't follow`,
     `missing cell`, `confounded`, `non-causal`, `lazy reading`,
     `methodological gap`) OR the explicit `source:methodological_gap`
     tag at write time.

2. **`compute_grounding` adds a `methodological_concerns` channel.**
   Fires when the draft has an inference marker AND a topically-similar
   memory carries methodological-warning language or the
   methodological-gap source tag. Surfaces in the output as a separate
   list from `contradict` so downstream callers can show "the substrate
   flagged this as a methodological overclaim" distinctly. Reduces
   `belief_confidence` the same way factual contradictions do, so
   `gap_score` and `severity` reflect the concern automatically.

3. **Methodological check runs BEFORE factual contradiction check.**
   When both fire on the same memory (the methodological-gap memory
   often *also* contains negation cues like "unsupported"), the
   methodological framing wins. It's more informative — it tells the
   user *why* the inference is flawed, not just *that* a memory
   disagrees.

4. **`aether_fidelity` and `aether_sanction` expose the new field.**
   `aether_sanction` adds a HOLD-when-APPROVE rule: if the baseline
   verdict would have been APPROVE but a high-trust methodological
   concern fires, the verdict downgrades to HOLD. Methodological
   overclaims are about the form of a claim, not the action itself —
   the right response is "review this methodology before proceeding,"
   not blanket REJECT.

Critical regression test: `test_grounding_surfaces_methodological_concern`
seeds the exact memory and queries the exact draft from the v0.9.1
test report. The methodological concern surfaces; `belief_confidence`
drops below the 0.4 neutral baseline. Plus 24 supporting tests covering
the helpers, false-positive guards (dissimilar topics, drafts without
inference markers), and the MCP tool surface.

222 tests pass (was 197). 25 new tests in
`test_v093_methodological_overclaim.py`.

## Shipped (v0.9.2)

### Governance tier no longer wedges on cold encoder

The first end-to-end substrate-assisted dev test (2026-04-28) found that
`aether_sanction` hung >10s on cold start and got killed twice. Diagnosis:
the call path is `aether_sanction → govern_response → template_detector
.detect("rm test_image.jpg") → regex finds no hedges → falls into
_scan_hedges_by_embedding → _get_embedder()`. That last call did a
*synchronous* `from sentence_transformers import SentenceTransformer` +
`SentenceTransformer(...)` instantiation — the same wedge v0.8.x fixed
for `aether_search`, hidden inside the governance layer with its own
private embedder. Two more identical patterns in `speech_leak_detector`
and `continuity_auditor`. None integrated with the StateStore's
`_LazyEncoder` warmup machinery.

Fix:

1. **Extract `LazyEncoder` to `aether/_lazy_encoder.py`** as a shared,
   process-wide non-blocking encoder. Cache (`_MODEL_CACHE`) means
   multiple `LazyEncoder` instances for the same model share one
   underlying load. Model name normalization (`all-MiniLM-L6-v2` ↔
   `sentence-transformers/all-MiniLM-L6-v2`) keeps the cache key
   consistent across modules.

2. **Each governance immune agent** (`TemplateDetector`,
   `SpeechLeakDetector`, `ContinuityAuditor`) now uses the shared
   `LazyEncoder`. First access kicks off background warmup; until warm,
   `_get_*()` returns `None` instead of blocking. Each callsite handles
   `None` gracefully:
   - `TemplateDetector._scan_hedges_by_embedding` returns `[]` (regex
     pass remains authoritative).
   - `TemplateDetector._compute_variance` returns `0.0`.
   - `SpeechLeakDetector.detect` returns a conservative fallback verdict
     — `BLOCK` for high-trust writes, `DOWNGRADE` for low-trust — with
     reason "encoder still warming up; cannot verify grounding."
   - `ContinuityAuditor.check` returns `PASS` (no continuity check
     possible without embeddings).

3. **Critical regression test** (`test_sanction_on_hedge_free_imperative_returns_under_5s`):
   the exact input that wedged in production now returns within 5s on
   cold encoder. Plus 7 unit tests covering each module's non-blocking
   contract and shared-cache behavior.

197 tests pass (was 189). 8 new tests in `test_v092_governance_nonblocking.py`.

### Deferred to v0.9.3

Tool-level timeout wrapper as defense-in-depth. The internal blocking is
gone, so this is insurance for future tools rather than a needed fix.
Implementation requires careful Python threading work (you can't kill
threads cleanly; signal-based timeouts don't work on Windows).

## Shipped (v0.9.1)

### aether_path was a no-op on substrates built through MCP — now fixed

v0.9.0 shipped Dijkstra retrieval over the BDG, but the MCP write
surface (`aether_remember`, `aether_ingest_turn`, `add_memory`)
only ever produced CONTRADICTS edges. SUPPORTS / DERIVED_FROM /
RELATED_TO had no creation path from MCP. So `aether_path` walked
backward from the target, found no edges to traverse, and returned
just the target alone — a no-op in production. The tests passed
because every multi-node test in `test_path_v09.py` manually called
`store.graph.add_edge(...)` to construct a chain. Public-API
behavior was never asserted.

Fix:

1. **Auto-link RELATED_TO on write.** In the same top-K candidate
   scan that detects contradictions, any candidate above
   `AUTO_LINK_THRESHOLD` (default 0.7, override with
   `$AETHER_AUTO_LINK_THRESHOLD`) that did NOT trigger a
   contradiction gets a bidirectional RELATED_TO edge. Reuses the
   existing candidate set — no second scan, no second cost.
   Contradicting pairs are excluded by design (no both-edges case).
2. **`aether_link` MCP tool.** Explicit edge creation when the
   similarity heuristic won't catch a relationship —
   `aether_link(source_id, target_id, edge_type, weight, reason)`.
   `edge_type` is validated against the EdgeType enum;
   CONTRADICTS / SUPERSEDES are rejected (those have their own
   detection / resolution paths). SUPPORTS / DERIVED_FROM are
   directional; RELATED_TO is bidirectional.
3. **`aether backfill-edges` CLI.** For substrates built on v0.9.0
   that have orphan nodes — retroactively walks all pairs and
   wires RELATED_TO edges for those above threshold. Idempotent
   (skips pairs that already have any edge). Supports `--dry-run`.

Critical regression test added: writes two memories via the public
`aether_remember` API only (no manual `graph.add_edge`) and asserts
`aether_path` returns a path with more than one node. That test
should have existed in v0.9.0.

189 tests pass (was 163). 26 new tests in `test_v091_auto_link.py`.

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
