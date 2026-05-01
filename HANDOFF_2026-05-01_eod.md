# End-of-day handoff: 2026-05-01

Closes the day. Picks up from `HANDOFF_2026-05-01_to_windows.md` (morning) and `HANDOFF_2026-05-01_to_windows_late.md` (afternoon).

## What shipped today (5 releases)

| Tag | Where | What |
|---|---|---|
| `v0.12.19` | M2 | Tier 1 — doctor plugin scan, co-policy guard, version-drift direction |
| `v0.12.20` | M2 | extract_facts clause-trim + question-skip |
| `v0.12.21` | M2 | Polarity-aware contradiction detection (asymm_neg threshold raise, co-rejection guard, polarity-flip guard in compute_grounding) |
| `v0.13.0` | M2 | **Phase A** — slot extractors for third-person prose + project facts |
| `v0.13.1` | Windows | **Phase B** — write-time slot conflicts via `project_chosen_option` |
| `v0.13.2` | Windows | CLI silent-output regression fix (`_lazy_encoder` redirect_stdout on bg thread) |

Plus a Phase D doc-honesty pass (Windows, `d5e9c60`) — README has a "What's measurably true today" smoke-probe table and an explicit "What aether does and doesn't do (yet)" section. The 88%/40% claim is hedged in two places now.

## Verified end-to-end

The cross-platform CI loop closed for the first time today: regression caught on M2 → fixed on Windows → re-verified on M2.

**M2 validation gauntlet results (post-pull of `d5e9c60`):**
- `aether status` produces output (was 0 bytes in v0.12.6 → v0.13.1) ✅
- `python -m bench.smoke_v131` → 8/8 PASS ✅
- `pytest tests/ --ignore=tests/e2e -q` → 539 pass, 1 fail (pre-existing) ✅
- Phase B done criterion #2 — DECISIONS.md ingest: **0 FPs, 0 asymm_neg traces** (vs 17 baseline) ✅
- Positive control on Option A/B canonical phrasing: 1 contradicts edge, tension=0.90 ✅

**Honest caveat on the DECISIONS.md result:** the 0-FP score is partly selectivity and partly extractor template-rigidity. We don't know the false-negative rate without a labeled corpus. That's the recall question Phase C is for.

## State at EOD

- Repo: master at `d5e9c60`, in sync with origin on both machines
- Tags through `v0.13.2` pushed to GitHub
- Plugin marketplace `autoUpdate: true` will propagate v0.13.2 to user plugin caches on next session start
- PyPI still on `v0.12.17`. `dist/aether_core-0.13.2-{whl,tar.gz}` is built + `twine check` clean (Windows). Marketplace install path doesn't depend on PyPI; this is only for `pip install aether-core` users.
- `~/.aether-venv/` editable installs on both machines, `aether.__version__ == "0.13.2"`
- Substrate state: M2 has 15 memories, Windows has 155 memories — both clean (no false-positive contradictions remaining post-v0.12.21 + Phase B)

## Three open items, in priority order

1. **Phase C** — paraphrase corpus (`bench/paraphrase_corpus.jsonl`, 50 cases) + CI integration + perf. The honest answer to the recall question. ~2-3 hours of focused work next session.
2. **PyPI upload** of v0.13.2. `cd ~/Documents/ai_round2/aether-core && twine upload dist/aether_core-0.13.2*` from either machine, with the PyPI token. Skipped today by user choice.
3. **Pre-existing test_v126 HF-network test fixture** — fails on both M2 and Windows because the test creates a tmp `HF_HOME` that can't reach huggingface.co. Real fix: skip-when-offline-and-uncached, or point at the real cache, or mock the SentenceTransformer load. ~30 min cleanup, low priority.

## Roadmap reminder

From `ROADMAP.md`, where we are vs the published plan:

| Phase | Status |
|---|---|
| Phase 0 — Archaeology | ✅ done in `SESSION_2026-05-01_archaeology_and_tier1.md` |
| Phase A — MVP slot canonicalization | ✅ shipped v0.13.0 |
| Phase B — Write-time + vocab expansion | ✅ shipped v0.13.1 (smaller than scoped — handoff hypothesis confirmed) |
| Phase C — Bench corpus + perf | ⏳ next |
| Phase D — Honest reframe of public docs | ✅ ~80% (paper-in-flight outstanding) |

## Notable technical patterns established today

Worth keeping for future sessions:

1. **Round-trip every fix through the surface that caught the bug.** Today's CLI silent-output bug was found on M2, fixed on Windows, verified back on M2. That's the cross-platform CI we don't otherwise have.
2. **Smoke probe is README's load-bearing evidence layer.** `bench/smoke_v131.py` has 8 probes that map 1:1 to README claims. Future claim-drift breaks the probe; future probe-fail forces the claim to be pulled back.
3. **Subprocess-based regression tests for shell-facing bugs.** `tests/test_v132_cli_no_silent_output.py` runs the actual `aether` console binary because pytest's stdout-capture masks the redirect-leak class of bug. Add to the toolbox for "did this CLI command produce output?" questions.
4. **Phase A → Phase B handoff hypothesis was right.** "The slot tags already flow through to write time naturally because `add_memory` already calls `extract_fact_slots`" — confirmed by `bench/phase_b_verify.py` (2/3 cases passed pre-extractor; only Case B needed the new `project_chosen_option`). Future "is this big-or-small?" questions: write a verify probe first, then scope.
