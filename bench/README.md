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

| Category | What it tests | Current rate |
|---|---|---|
| `factual_contradiction` | StructuralTensionMeter slot conflicts (Seattle vs Portland) | 100% |
| `mutex_contradiction` | Class-based mutex (AWS vs GCP, Postgres vs MySQL) — v0.6 | 100% |
| `negation_asymmetry` | "We use X not Y" vs imperative to use Y — v0.5 | 100% |
| `policy_violation` | Prohibition + imperative (force push, --no-verify) — v0.5 | 100% |
| `methodological_overclaim` | Inference draft + methodological-gap memory — v0.9.3 | 100% |
| `no_issue_grounded` | Claim aligns with substrate; should surface support | 100% |
| `no_issue_unrelated` | Claim about topic substrate doesn't cover; no spurious flags | 100% |
| `false_positive_guard` | Specificity tests — must NOT fire on superficially-similar text | 100% |
| `known_gap_quantitative` | Numeric / version / date conflicts (NOT YET caught) | 0% (tracked) |

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
