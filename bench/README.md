# Aether Fidelity Calibration Bench

Turns "fidelity catches the right things" from anecdote into a measurable
property. Curated corpus of `(substrate, claim, expected_verdict)` cases
across the categories aether's governance tier is supposed to handle.

## Run

```bash
# Full corpus, markdown report
python -m bench.run_fidelity_bench

# JSON output for CI scraping
python -m bench.run_fidelity_bench --format json

# Run a single case (debug a failing one)
python -m bench.run_fidelity_bench --only methodological_001_calic_canonical
```

The runner exits non-zero if any **blocker category** regresses.
Known-gap categories (prefixed `known_gap_`) are tracked in the report
but do not trigger non-zero exit.

## Categories

Two columns: **warm** = encoder loaded (`pip install aether-core[ml]` + warmup completed), **cold** = no encoder (substring + Jaccard fallback only). The numbers below are **`bench/run_fidelity_bench.py`** results on a clean substrate, not the live `~/.aether/` substrate.

| Category | What it tests | Warm | Cold |
|---|---|---:|---:|
| `factual_contradiction` | StructuralTensionMeter slot conflicts (Seattle vs Portland) | 100% | 100% |
| `mutex_contradiction` | Class-based mutex (AWS vs GCP, Postgres vs MySQL) — v0.6 | 100% | 100% |
| `false_positive_guard` | Specificity tests — must NOT fire on superficially-similar text | 100% | 100% |
| `no_issue_unrelated` | Claim about topic substrate doesn't cover; no spurious flags | 100% | 100% |
| `methodological_overclaim` | Inference draft + methodological-gap memory — v0.9.3 | 100% | 80% |
| `policy_violation` | Prohibition + imperative (force push, --no-verify) — v0.5 | 100% | 50% |
| `no_issue_grounded` | Claim aligns with substrate; should surface support | 100% | 25% |
| `negation_asymmetry` | "We use X not Y" vs imperative to use Y — v0.5 | 100% | 0% |
| `known_gap_quantitative` | Numeric / version / date conflicts (NOT YET caught) | 0% | 0% |
| **Blocker pass rate** |  | **100% (26/26)** | **~76% (22/29)** |

**Why cold rates degrade.** The categories that survive cold mode (`factual_contradiction`, `mutex_contradiction`, `false_positive_guard`, `no_issue_unrelated`) are the ones doing pure slot extraction or class-based mutex — structural primitives that need no embeddings. The categories that fall (`negation_asymmetry`, `no_issue_grounded`, parts of `policy_violation`) all gate on `embedding_similarity >= 0.45` for meter dispatch, and Jaccard on bare tokens rarely clears that gate. Cold mode is a documented degraded-functionality state, not a regression — see `_v0.9.5_` release notes for the original baseline.

**What this means for installs.** On a fresh Mac install without `[ml]` (PyTorch + sentence-transformers, ~2GB), the substrate runs in cold mode by default. `aether warmup` after `pip install aether-core[ml]` is the path to warm-mode numbers. `aether doctor` reports the encoder state explicitly so you can tell which mode you're in.

## Adding a case

Edit `bench/fidelity_corpus.json`. Each case looks like:

```json
{
  "id": "unique_snake_case_id",
  "category": "one_of_the_categories_above",
  "description": "what this tests, in one sentence",
  "substrate": [
    {"text": "memory text to seed", "trust": 0.85, "source": "user"}
  ],
  "claim": "the draft to grade",
  "expected": {
    "supporting_min": 1,
    "supporting_max": 3,
    "contradicting_min": 0,
    "contradicting_max": 0,
    "methodological_min": 0,
    "methodological_max": 0,
    "sanction_verdict_in": ["APPROVE", "HOLD", "REJECT"]
  }
}
```

All `expected` fields are optional. Use only the constraints that matter
for the case. `sanction_verdict_in` is only checked when present.

## Known-gap workflow

When you find a real fidelity miss in production:

1. Add a case under `category: "known_gap_<short_label>"` documenting
   the limitation in the description.
2. The bench tracks the failing case but doesn't block. The blocker
   pass rate in the report stays at 100%.
3. When you ship a fix, move the case to the appropriate non-`known_gap_`
   category. The bench will then enforce the new behavior as a
   regression test.

This pattern lets the bench be both **honest** (visible failures, no
hiding) and **useful as CI** (won't break the build for known
limitations).

## Pytest integration

`tests/test_v094_fidelity_calibration.py` runs the corpus inside the main
test suite. Five assertions:

1. Every blocker case passes
2. Blocker pass rate is exactly 100%
3. Methodological recall is 100% (Layer 2 regression guard)
4. False-positive guards hold (specificity guard)
5. Corpus has at least one case per blocker category (coverage guard)

So `pytest tests/` re-runs the bench every time. A regression in
fidelity will fail the suite with a per-case breakdown.

---

# Validation Chapter, Test #1 — observational substrate snapshot

`run_fidelity_bench.py` is unit-test style: synthetic substrates, expected
verdicts, pass/fail. `validation_test1.py` is observational: runs a fixed
question battery against the **live** substrate (your actual
`~/.aether/mcp_state.json`) and produces a markdown report of what the
substrate's read tools actually return.

## Run

```bash
# Default: warm-up encoder, run against live substrate, save to bench/results/
python -m bench.validation_test1

# Specific state file
python -m bench.validation_test1 --state-path ./my_state.json

# JSON for diffing snapshots over time
python -m bench.validation_test1 --format json --out snapshot.json

# Skip warmup (cold-mode snapshot — what a stranger sees right after install)
python -m bench.validation_test1 --wait-warmup 0
```

## Question battery

10 questions across 5 categories in `validation_test1_questions.json`:

| Category | What it exercises | Tools |
|---|---|---|
| A. Memory recall | Substrate surfaces user-stated facts | `aether_search` |
| B. Contradiction handling | Substrate exposes held tensions | `aether_search` |
| C. Sanction gate | Substrate blocks unsafe actions | `aether_sanction` |
| D. Fidelity grounding | Substrate scores draft responses | `aether_fidelity` |
| E. Cold queries | Substrate doesn't pretend to know | `aether_search`, `aether_fidelity` |

The questions are deliberately fixed across runs so two snapshots can be
diffed to show how substrate behavior evolved. Edit the JSON to add or
swap questions; existing snapshots become non-comparable but the format
stays the same.

## What the first run surfaced (2026-04-30)

Even running test #1 the first time produced three real findings — exactly
the substrate-caught-itself loop the README promises:

- **Sanction gate has no default policy beliefs.**
  `aether_sanction("git push --force origin main")` returns APPROVE on a
  fresh substrate because F#4's policy contradiction detection requires
  an explicit "never force-push" belief. Either `aether init` should seed
  a default belief set, or sanction needs structural detection of
  high-risk language without belief presence.

- **Trust-vs-verbosity ranking gap.**
  F#10's `SEARCH_TRUST_WEIGHT=0.7` keeps a trust=0.67 demoted memory below
  trust=0.95 truths, but doesn't keep trust=0.90 short-text entries from
  outranking trust=0.95 verbose-text entries. Visible in category B
  (name search): "user name: Jake" (trust=0.90) outranked
  "user name: Nick (observed 65x in production)" (trust=0.95) because
  Jake's text aligns better with the bare query.

- **Inject threshold leaks cold-query noise.**
  "capital of France" search returned `user name: Claude` at score 0.158
  — just above the 0.15 inject threshold, so unrelated memories would
  reach the LLM. Either raise the threshold to ~0.20 or add a per-query
  semantic relevance gate.

These belong in NEXT_SESSION's open-findings list once investigated /
prioritized. The methodology proved its purpose on first contact.

## Future work

- **Pair with a no-substrate baseline.** Same questions against an empty
  `StateStore`. The diff is the substrate's measurable value-add.
- **Ground-truth labels.** Add expected outcomes per question so the
  report can grade itself instead of just snapshotting.
- **Time-series.** `bench/results/` accumulates timestamped snapshots; a
  diff tool could highlight what changed when.
- **N>1 user.** Run on substrates seeded by different real users.
