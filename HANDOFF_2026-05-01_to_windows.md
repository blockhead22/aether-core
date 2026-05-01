# 2026-05-01 Handoff: Mac → Windows

End-of-session checkpoint. Everything below is what a fresh Claude Code session on Windows needs to pick up where this Mac session left off, since the auto-memory at `~/.claude/projects/-Users-nickblock/memory/` is Mac-local and won't follow the move.

## What got done in this session (compressed)

**Four releases shipped, all on `master` and pushed to GitHub:**

| Tag | Commit | What |
|---|---|---|
| `v0.12.19` | `5999ac0` | Doctor plugin-tree scan, co-policy guard, version-drift direction (carrier of `72cb562` fix) |
| `v0.12.20` | `4c65887` | extract_facts clause-trim + question-skip (carrier of `c60052a` fix) |
| `v0.12.21` | `7e7c417` | Polarity-aware contradiction detection — `_NEGATION_CUES` cleanup, asymm_neg threshold raise to 0.75, co-rejection guard, polarity-flip guard in `compute_grounding` with content-token-overlap floor at 0.25 |
| `v0.13.0` | `addf463` | **Phase A — slot canonicalization for fidelity grading.** Third-person prose extractors (occupation/location/employer), editor preference, project_vector_store / project_framework / project_embedding_dim. Done criterion met: `aether_fidelity("Nick is a chef in Paris.")` now returns 0.00 (was 0.95). |

**Test count:** 524 pass, 1 unrelated pre-existing HF-network failure in `test_v126_encoder_warmup_in_subprocess.py` (subprocess can't reach HuggingFace, fails without my changes too).

**Distros built (Mac-local, NOT uploaded to PyPI):** `dist/aether_core-0.12.20.{whl,tar.gz}`, `dist/aether_core-0.12.21.{whl,tar.gz}`. v0.13.0 distro NOT yet built — rebuild on whichever machine handles the upload so it carries the latest git state.

**PyPI:** still on `0.12.17`. Upload deferred — needs interactive token entry.

## Key context that was in Mac auto-memory (won't carry to Windows)

**About Nick (the user):**
- Authored `github.com/blockhead22/aether-core` (~1+ year of work across CRT → CRT-GroundCheck-SSE → aether-core lineage).
- Engage as the maintainer. He prefers honest critique grounded in prior art over vague praise. Cite sources, name unsourced claims, skip "you're amazing" framing.
- This was the first session in the project's history that ran an honest end-to-end bench against natural prose. Findings were real and surfaced in real time.

**About the lineage:**
- **OG CRT** (`~/Documents/ai_round2/CRT/`) — original concept lab, started ~Dec 2024. `THEORY.md` is the conceptual doc (memory as Gaussian splat, geometric contradiction via Bhattacharyya overlap, inverse entrenchment thesis, context-dependent covariance, predictive contradiction, TDA, etc.). `core/contradiction_manager.py` is a strip-equality + confidence-drop detector. Different paradigm from aether's cue-based.
- **CRT-GroundCheck-SSE** (`/Volumes/ex_video/ai/CRT-GroundCheck-SSE/` — external drive on Mac, Nick will need to confirm path on Windows) — bridge architecture, 4 months of life. `personal_agent/contradiction/ml_detector.py` uses TfIdf + gradient-boosted scorer with explicit `negation_in_new`/`negation_in_old`/`negation_delta` features and `RETRACTION_PATTERNS`. Avoids aether-core's specific bugs by virtue of different architecture; has its own untested blind spots.
- **aether-core** — current. Started 2026-04-01 with `crt/contradiction/__init__.py` empty. Cue-based detector machinery shipped 2026-04-27 (v0.5.0). The bugs we patched tonight existed in mainline for 4 days.

The headline pitch (`aether_fidelity` distinguishes supported / merely-compatible / hallucinated drafts) was broken on non-prohibition memories until v0.13.0 Phase A landed. Phase B ships the write-time integration.

**About the substrate state:**
- Mac substrate at `~/.aether/mcp_state.json` is clean: 7 default policy beliefs in Belnap=`T`, 0 contradiction edges, no auto-ingested user memories.
- Pre-cleanup corrupt copy at `~/.aether/mcp_state.before-cleanup.json` (Mac only — has the 9 false-positive contradictions from the original buggy seeding, kept as a regression-test fixture).
- Windows will start with no substrate; SessionStart hook will seed the 7 defaults on first invocation.

## Windows setup steps (the plumbing)

1. `git clone https://github.com/blockhead22/aether-core.git ~/Documents/ai_round2/aether-core` (or wherever you keep code on Windows).
2. Create venv: `python -m venv %USERPROFILE%\.aether-venv` (the launcher looks for `~\.aether-venv\Scripts\python.exe` on Windows per the platform check in `hooks/aether_launcher.py`).
3. Editable install: `%USERPROFILE%\.aether-venv\Scripts\pip install -e <repo>`.
4. Verify: `%USERPROFILE%\.aether-venv\Scripts\python -c "import aether; print(aether.__version__)"` should print `0.13.0`.
5. Run doctor: `%USERPROFILE%\.aether-venv\Scripts\aether doctor`. Expect 7 OK on first run AFTER SessionStart fires, since SessionStart seeds the substrate.
6. Plugin install in Claude Code: `claude plugin install github.com/blockhead22/aether-core` (the marketplace points at the same GitHub repo).
7. First Claude Code session in any directory: SessionStart fires, seeds `%USERPROFILE%\.aether\mcp_state.json` with the 7 default policy beliefs, encoder warmup begins in background.

The launcher (`hooks/aether_launcher.py`) discovers the venv automatically. If discovery fails, set `AETHER_PYTHON=%USERPROFILE%\.aether-venv\Scripts\python.exe` as an env var.

## Where to read for context

In the repo (travels with git pull):
- `ROADMAP.md` — Track 0 phase plan with done criteria. **Phase A shipped tonight; Phase B is next.**
- `SESSION_2026-05-01_archaeology_and_tier1.md` — full session arc through v0.12.21. Captures bench findings, CRT/GroundCheck/aether archaeology, the Tier 1/2/3/4 plan.
- `CHANGELOG.md` — release notes (last updated v0.12.14, recent versions documented in commit messages).
- `tests/test_v1221_polarity_and_overlap.py` (22 tests) and `tests/test_v130_slot_canon_fidelity.py` (16 tests) — regression tests for tonight's fixes. Run with `pytest tests/test_v130_slot_canon_fidelity.py tests/test_v1221_polarity_and_overlap.py -v` to confirm everything still works on Windows.
- `tests/test_extractor_clause_and_question.py` (12 tests) — v0.12.20 extractor regressions.

## What's still open (Track 0 Phase B and beyond)

From `ROADMAP.md` — pasted here for handoff completeness:

**Phase B (next, ~1 week → v0.13.1)**
- Wire slot-canonical detector into the write-time contradiction cascade (currently read-side only via compute_grounding).
- Expand slot vocab from ~10 to ~30 categories. Mine from `~/Documents/ai_round2/CRT/DECISIONS.md` and `ROADMAP.md` for project-specific slots.
- Add typed-value parsers (integer/float/date/version/categorical).
- Done criterion: Test #3 reflexive bench Cases A and B fire as contradictions at write time. DECISIONS.md ingest false-positive count drops to ≤1.

**Phase B might be smaller than originally scoped** — Phase A turned out to be vocab expansion, not new logic. Phase B should be similarly thin: the slot tags already flow through to write time naturally because `add_memory` already calls `extract_fact_slots`. Need to verify the existing `_slot_equality_match` detector picks them up, then expand vocab and add typed parsers.

**Phase C (~1 week → v0.13.2):** bench corpus + perf. Build `bench/paraphrase_corpus.jsonl` with 50 cases. CI integration.

**Phase D (~1 week → v0.13.3, parallelizable with C):** README/bench/paper reframe. Hedge the "88%-vs-40%" claim or source it. Add cold-vs-warm column to bench README.

**Tier 3 / Tier 4 deferred** until Phase D ships. See ROADMAP.md for the full list (port disclosure_policy.py, volatility, crt_critic, commitments, splats, multi-turn stance flips, Contextual disposition).

## Untested by tonight's bench (worth doing in a future session)

- 6 immune agents at the per-agent triggering level (loaded + wired confirmed; per-agent behavioral tests pending)
- Auto-link `RELATED_TO` edges at scale (one false-positive `contradicts` edge surfaced from the structural tension meter on related paraphrases)
- Belnap state automatic transition with trust (currently doesn't track trust — m1 trust 0.9 → 0.0 left belnap_state at `T` in the cascade test)
- `flag_for_review` principle-pair false positives (3 remaining in DECISIONS.md ingest, from the structural tension meter — separate code path from the asymm_neg detector we patched)

## The Discord post (drafted, not yet sent)

Final calibrated version Nick was about to send to `#soft-hard-ware` in 4Space Development Server. Saved here so it doesn't get lost in the move:

> so I guess it isn't really the problem and its more that my entire bet on this system working is the concept of detecting when a misstated fact is confidently trusted over what the user has preferred historically. or conversely if the system contradicts itself say in a long multi-agent handoff coding session. Eventually something will get lost or forgotten. the irony is I'm completely relying on regex to do the fact extraction and slot classifications because any level of ML in the hot path requires considerable slowdown, a daemon shell to babysit. Maybe that's just the tradeoff for more persistent memory in AI?

## What this session validated about the project

Storage / hooks / sanction / trust cascade / immune agents (loaded) / search / remember / context — all working. The post-generation fidelity grader (the headline GroundCheck-equivalent claim) was broken on non-prohibition memories until tonight; Phase A landed the fix on the canonical case. The architectural fix for paraphrase-blindness across all memory types is Phase B.

The honest summary: the bugs we fixed had been in mainline for 4 days. The architectural disease (cue overlap dominates polarity / slot-key awareness) is the design choice cue-based detection makes. v0.13.0 starts the slot-canonicalization arc that converts that architectural choice into a working post-generation verification surface.

The previous iterations (CRT, GroundCheck) weren't failed attempts — they were the substrate this session built on. Tonight added the discipline that was missing: paste-into-harness done criteria, regression tests for every bench failure, honest reframe of what works versus what's still open.
