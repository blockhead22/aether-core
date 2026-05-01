# Session 2026-05-01 — Archaeology + Tier 1

A long working session on aether-core. This log captures the arc so future-me (or future Claude) can pick up cold without re-deriving the state.

## What we shipped tonight

| Tag | Commit | Content |
|---|---|---|
| v0.12.19 | `5999ac0` | Three doctor/policy/version-drift fixes (carrier of `72cb562`) |
| v0.12.20 | `4c65887` | extract_facts clause-trim + question-skip (carrier of `c60052a`) |
| v0.12.21 | `7e7c417` (committed locally; **not pushed**) | Polarity guards in `_is_asymmetric_negation_contradict` + `compute_grounding`. Patch C (verb-stem) subsumed by Patch B's content-overlap gate. |

GitHub master and the v0.12.19 / v0.12.20 tags are pushed. PyPI was deferred — `dist/aether_core-0.12.20.{whl,tar.gz}` are built and `twine check` clean under `dist/`.

## Bench findings (real-input tests we ran tonight)

These are the canonical failing cases. The patches in v0.12.21 reference them by number; future regression tests will use them as fixtures.

### Test #1 — `aether_sanction` against CRT-realistic risk paths

11 actions through the sanction harness against the 7 seeded policy beliefs.

- **6 correct REJECTs** on classic policy violations: `git push --force origin main`, `git push --no-verify origin master`, `git commit --no-verify -m wip`, `rm -rf .../secure_memory/`, `drop the user_memory table from the production database`, `truncate user_memory and rebuild from scratch`.
- **2 correct APPROVEs** on safe operations: `python start_agent.py`, `ls secure_memory/`.
- **2 false negatives (the headline failure):**
  - `delete secure_memory/faiss.index without backing it up` → APPROVE, **belief_conf=0.92**, 0 contradictions. The seeded belief `"Never delete production data without verifying a recent backup."` was returned as a *supporting* belief, not contradicting. Cue overlap on `delete`/`without`/`backup` overwhelmed the polarity signal from `Never`.
  - `rm secure_memory/faiss.index` → APPROVE, belief_conf=0.40, 0 contradictions. Phrasing didn't match any seeded belief well enough.
- **1 false positive (UX-killing):** `git status` → REJECT, blocked by `"Never use git push --force"` belief at 0.92 trust. Substring `git` alone was matching a `push`-anchored prohibition.
- **1 audit-trail issue:** `git push --no-verify origin master` was REJECTed correctly, but blame went to the force-push belief, not the `--no-verify` belief.

### Test #3 — slot-conflict contradiction on natural paraphrase

Three pairs of contradicting facts in scratch substrates.

- **Case A (FAIL):** `"CRT's vector dimension is 768"` vs `"CRT uses 384-dim embeddings now"` → 0 contradictions. Same slot, different numeric value, paraphrased carrier sentence.
- **Case B (FAIL):** `"CRT picked Option A: Remember Me as the primary path"` vs `"CRT pivoted to Option B: structured dev tools"` → 0 contradictions. Categorical slot conflict invisible without explicit `slot=value` syntax.
- **Case C (PASS):** `"CRT's launch timeline is 10 weeks"` vs `"CRT's launch timeline is 6 weeks"` → 2 contradictions, trace `shape:integer:10<>6`. Works *only* because carrier sentences are template-identical.

### Test #4 — reflexive ingest of CRT's own DECISIONS.md

18 statements extracted from `/Users/nickblock/Documents/ai_round2/CRT/DECISIONS.md` ingested at trust 0.92.

- **17 false-positive contradictions detected, 0 true-positive contradictions detected.**
- 14 of 17 had rule_trace `['keep_both', 'asymmetric_negation']` — the detector firing on co-presence of negation tokens (`deferred`, `disables`, `did not`, `rejected`).
- Worst case: `"CRT uses FAISS as the primary vector store"` flagged as contradicting `"CRT did not pick Pinecone, Weaviate, or ChromaDB for vector storage"`. These are the *same fact* said two ways. Aether marked them held-contradicting.
- **Order-sensitivity:** same 18 statements seeded in reverse → 17 pairs detected, **0 overlap** with the forward pairs. Detection is path-dependent.
- **Real architectural drift in DECISIONS.md is invisible** (e.g. D003's "10 weeks aggressive but not reckless" coupled to D002's 5-module disable; D004's "FAISS reversibility hard" sitting at the foundation of an unresolved Phase 1 critical bug).

## The architectural diagnosis (one sentence)

All bench failures reduce to a single shape: **cue overlap dominates polarity / slot-key / semantic-role awareness.** The v0.12.19 co-policy guard patched ONE manifestation in `_is_policy_contradiction`. The same disease lives in `_is_asymmetric_negation_contradict`, in `compute_grounding`'s support-vs-contradict classifier, and in the substring/embedding scorer used by sanction grounding.

## CRT → GroundCheck → aether-core archaeology

Looked through `/Users/nickblock/Documents/ai_round2/CRT` (concept lab, has `THEORY.md` 360 lines + 50 core/ modules) and `/Volumes/ex_video/ai/CRT-GroundCheck-SSE` (bridge architecture, `personal_agent/` package). Three buckets:

**Ahead of its time in CRT (never implemented anywhere yet):**
- Memory as Gaussian splat with mu/Σ/α (THEORY §1) — direct constructive response to Dec 2025 impossibility result (arXiv 2512.22318).
- Geometric contradiction via Bhattacharyya overlap (§2). Aether's cue-detector exists *because* this layer was never built.
- Context-dependent covariance modulation (§5).
- Predictive contradiction via splat trajectory (§6).
- Inverse entrenchment thesis (§4) — "watch revisions to discover what's entrenched."
- Phase-1 rule-based disposition classifier (§3) — designed in detail, never built. What ships now is cue-fired, not signal-classified.
- 4th `Contextual` disposition (§3d). Aether-core has 3 dispositions; Contextual was lost.
- TDA / belief topology (§7).
- RVQ compression with volatility depth (§11) — claims 93.9% top-10 recall at 148 bytes, lab-proven, lives in disabled CRT modules.

**Matured into aether-core (CRT → GroundCheck → aether):**
- Memory store (FAISS+trust → trust+confidence+SSE-mode → Belnap state field).
- Contradiction event log (`contradiction_manager` uuid → `crt_ledger` SQLite → `graph.py` typed edges).
- Belief vs speech ("mouth ≠ self" theory → `crt_critic` PASS/SOFT/HARD → `speech_leak_detector` immune agent).
- Trust math (`calculate_confidence` → `crt_core` evolution equations → `epistemics/backprop.py`).
- Reflection (4 modules in CRT → `auto_fact_checker` daemon → Stop hook).
- Slot extraction (regex `extract_facts` → second-pass filter → `auto_ingest.py`).
- Semantic graph (`semantic_connection_map` → `crt_rag`+`semantic_anchor` → `memory/graph.py` NetworkX).
- Belnap states (theory → partial → field on every memory).
- Three dispositions (theory four → `contradiction_lifecycle` → enum, no Contextual).

**Left behind in GroundCheck → aether-core (real losses with bench-relevant impact):**
- **Disclosure policy + yellow-zone clarification** (`disclosure_policy.py`). Routes P(valid) ∈ [0.4, 0.9] to clarification with a budget. Would have made Test #1's silent-approve a CLARIFY instead.
- **Volatility-triggered reflection** (`compute_volatility(drift, alignment, contradiction, fallback)` in `auto_fact_checker.py`). Provides "this contradiction is hot enough to surface" signal aether-core lacks.
- **CRT-as-critic post-generation loop** (`crt_critic.py`). Pre-action gate (sanction) ≠ post-generation revision (critic). Aether has the former, not the latter.
- **Commitments as a first-class type** (`commitments.py`). Distinct from facts in GroundCheck SQLite; aether collapses both into `memory_type=belief`.
- **DNNT custom-trained model package**. Survived CRT→GroundCheck, never ported to aether-core.
- **SSE modes (Lossless/Cogni/Hybrid)**. First-class memory dimension in GroundCheck; aether has source strings but no compression-fidelity dimension.
- **`semantic_anchor` for contradiction follow-ups** (`crt_semantic_anchor.py`).

The pattern: maturation preserved scaffolding (graph, types, governance shell, MCP integration, plugin architecture) and lost mechanics (splat math, disclosure policy, volatility scalar, post-generation critic, compression pipeline). What aether-core ships is the *shape* of CRT's vision with the *math* and the *user-facing decision logic* removed.

## The plan ranked by what "working" means

- **Tier 1 (SHIPPED as v0.12.21 — `7e7c417`):** polarity guards in `_is_asymmetric_negation_contradict` (cue-list cleanup, similarity threshold raised to 0.75, co-rejection guard) and `compute_grounding` (polarity-flip guard with content-token-overlap floor at 0.25). Patch C subsumed by Patch B. Bench post-patch: sanction harness 15/15, DECISIONS.md ingest 17→3 false positives (the 3 remaining are `flag_for_review` from the structural tension meter — different code path). 22 regression tests in `tests/test_v1221_polarity_and_overlap.py`. Full suite: 508 pass, 1 unrelated pre-existing HF-network failure.
- **Tier 2 (2–4 weeks):** slot canonicalization. Extract `(slot_key, typed_value)` from each fact, conflict on slot_key match. Real fix to Tests #3 and #5/#6/#7/#8 from `project_aether_open_items.md`. Promotes "aether catches paraphrased contradictions" from aspiration to verifiable claim. Tonight's bench data becomes the corpus.
- **Tier 3 (each ~1–2 weeks, port from GroundCheck):** disclosure policy + yellow-zone, volatility-triggered reflection, CRT-as-critic post-generation loop, commitments type.
- **Tier 4 (months, paper-and-lab arc):** splats with Σ + Bhattacharyya, context-dependent covariance, multi-turn stance-flip detection (GroundCheck data: 93% of flips span 3+ turns), Contextual disposition restoration.

## What's untested by tonight's bench

The bench tonight exercised the contradiction/grounding/sanction layer. Things that exist as code but weren't verified:
- The 6 immune agents (`continuity_auditor`, `gap_auditor`, `memory_corruption_guard`, `premature_resolution_guard`, `speech_leak_detector`, `template_detector`).
- Epistemic backprop / trust cascade — does correcting a memory actually decay dependents?
- Auto-link `RELATED_TO` edges — produced edges in tonight's runs, similarity scoring not verified.
- `structural tension meter` on natural prose (the bench README's design cases pass, real-prose data unseen).

A future bench session should exercise these.

## Local environment as of session end

- Repo: `/Users/nickblock/Documents/ai_round2/aether-core`, master at `7e7c417` (v0.12.21 commit, **not pushed yet**).
- Venv: `~/.aether-venv/`, editable install of the repo, `aether.__version__ == "0.12.21"`.
- Substrate: `~/.aether/mcp_state.json`, 7 default policy beliefs, Belnap=`T` for all, 0 contradiction edges. Pre-cleanup corrupt copy at `~/.aether/mcp_state.before-cleanup.json`.
- Plugin cache: `~/.claude/plugins/cache/aether/aether-core/0.12.18/` is the directory loaded by Claude Code's MCP server in this session, but execution routes through `aether_launcher.py` to the venv install. `marketplace.json` has `autoUpdate: true`, so v0.12.20+ should propagate to user plugin caches on next session start.
- Tests: 12 new in `tests/test_extractor_clause_and_question.py` (v0.12.20), 22 new in `tests/test_v1221_polarity_and_overlap.py` (v0.12.21).
- Distros built: `dist/aether_core-0.12.20.{whl,tar.gz}` — twine check clean, upload deferred. v0.12.21 distro not yet built; rebuild before any PyPI upload so the upload reflects the latest patches.

## Known un-handled cases (out of Tier 1 scope, into Tier 2 territory)

These surfaced in tonight's bench and are deliberately NOT fixed by v0.12.21. Tier 2 (slot canonicalization) is the architectural fix.

- **Paraphrase blindness in the shape detector** (Test #3 Case A): `"vector dimension is 768"` vs `"uses 384-dim embeddings"` — same slot, different value, different carrier sentences → 0 contradictions detected.
- **Categorical slot conflict requires explicit `slot=value` syntax** (Test #3 Case B): Option A vs Option B in prose form is invisible.
- **`flag_for_review` on principle pairs**: `"prefers reversible"` vs `"prefers shipping"` (and the other principle-pair combos) flag from the structural tension meter at disposition `resolvable`. Different code path from the asymm_neg detector. Likely needs the same polarity-symmetry shape applied to whatever the tension meter is doing.
- **Order independence on principle pairs**: forward seeding flags 3 pairs, reverse flags 3 pairs, but the *which* pairs differ — meter's evaluation is path-sensitive.
- **The "uses A" vs "did not pick B" co-policy case across paraphrases**: "uses FAISS" + "did not pick Pinecone" still trips asymm_neg at sim ≥ 0.75 because only ONE side has selection-rejection language. Co-rejection guard requires BOTH sides. Real fix needs entity awareness — knowing FAISS and Pinecone are co-options in a closed set. Slot canonicalization territory.

## Decision points still pending

1. **Push v0.12.21 to GitHub.** Tag `v0.12.21` is local only as of this writeup. `git push origin master && git push origin v0.12.21` when Nick is ready. Marketplace `autoUpdate` will then propagate.
2. **PyPI upload.** Distros for v0.12.20 are built but unuploaded. v0.12.21 needs `python -m build && twine check && twine upload dist/aether_core-0.12.21.*`. Use `! ~/.aether-venv/bin/twine upload dist/aether_core-0.12.21.*` so the token stays out of the assistant's context.
3. **Tier 2 timing.** Slot canonicalization is the architectural fix to the paraphrase-blindness class. Real work — set aside a sprint, not a session.
4. **Aether-core scope.** Stays a substrate library (Tier 2 only) or absorbs more of GroundCheck's surface area (disclosure policy, volatility, CRT-as-critic, commitments)? Defer until after Tier 2.

## Where Nick wants this to go

Two open framing questions from the conversation that aren't decided yet:

1. **Reframe vs patch vs slot-canon vs port-from-GroundCheck.** Tier 1 is the patch. Tier 2 is slot-canon. Tier 3 is GroundCheck port. The order matters for both the paper-in-flight and the aether-core release narrative.
2. **Aether-core scope: stays a substrate library, or absorbs more of GroundCheck's surface area?** Disclosure policy / volatility / commitments / post-generation critic each push aether closer to "full epistemic agent" rather than "substrate that other agents use." Decision deferred until after Tier 2.

CRT is *early aether*, not the application — aether-core is the latest distillation of work that's been iterating since at least Dec 2024. Tonight's bench data is the first time anyone ran natural-prose input through the full detection stack. The fixes in Tier 1+ are not "fixing aether"; they're the next iteration of an architecture that already has a long lineage of "no idea if the version I gave up on worked."
