# Contradiction-detection bench: structural meter vs LLM-as-judge

_Generated: 2026-05-01 00:58:49._

**Sample.** 24 cases from `bench/fidelity_corpus.json` covering factual_contradiction, mutex_contradiction, negation_asymmetry, policy_violation (positives) and false_positive_guard, no_issue_grounded, no_issue_unrelated (negatives). 5 cases excluded (methodological_overclaim is a different channel; known_gap_* is tracked separately).

## Headline

- **Structural meter:** 24/24 = **100.0%** accuracy.
- **LLM as judge:** SKIPPED — no `ANTHROPIC_API_KEY` in environment, or the `anthropic` SDK isn't installed. Re-run with the key set to populate this row.

## By category

| Category | n | Structural | LLM judge |
|---|---:|---:|---:|
| factual_contradiction | 5 | 100.0% (5/5) | — |
| false_positive_guard | 5 | 100.0% (5/5) | — |
| mutex_contradiction | 3 | 100.0% (3/3) | — |
| negation_asymmetry | 2 | 100.0% (2/2) | — |
| no_issue_grounded | 4 | 100.0% (4/4) | — |
| no_issue_unrelated | 3 | 100.0% (3/3) | — |
| policy_violation | 2 | 100.0% (2/2) | — |

## Methodology caveats

- The corpus is a hand-curated set the structural meter was tuned for. A held-out corpus is the next iteration.
- LLM-as-judge prompt is a single-shot YES/NO with a fixed prefix; a different prompt could change the LLM's accuracy meaningfully.
- Multi-memory cases are reduced to the FIRST memory for the LLM judge; the structural meter sees all memories. This is a deliberate asymmetry — it gives the LLM a simpler task, not a harder one.

---

_See `bench/run_contradiction_bench.py` for the runner. Re-run with `python -m bench.run_contradiction_bench` (set `ANTHROPIC_API_KEY` for the LLM column)._