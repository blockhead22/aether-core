# Roadmap

What's coming next, when we expect it, and what is intentionally not in this repo.

For what's already shipped, see [CHANGELOG.md](CHANGELOG.md).

## Where we are (2026-05-01, late session)

`aether-core` is at **v0.12.21** on master / GitHub releases (PyPI is one cycle behind at v0.12.17 — distros for v0.12.20 and v0.12.21 are built locally, upload deferred). **508 unit tests pass.** The substrate is **structurally complete** at the storage layer — every read tool degrades gracefully on corrupt nodes, every write tool syncs from disk, the auto-ingest hook fires every turn, secrets get redacted before fact extraction, every save snapshots the prior state, `aether_sanction` has default policy beliefs to gate against, and the SessionStart hook auto-installs / auto-upgrades / auto-initializes on first run.

**The 2026-05-01 reflexive bench ([SESSION_2026-05-01_archaeology_and_tier1.md](SESSION_2026-05-01_archaeology_and_tier1.md)) revealed a real architectural gap.** When natural prose from a real document (CRT's `DECISIONS.md`) is fed through the contradiction detector, the cue-based detectors produce ~1 false-positive per fact and miss every paraphrased contradiction. The polarity-aware patches in v0.12.19 / v0.12.21 (covered below) closed three concrete leaks; the underlying disease — **cue overlap dominates polarity / slot-key / semantic-role awareness** — remains in the non-prohibition code paths. `aether_fidelity` cannot currently distinguish *supported* from *merely-compatible* from *hallucinated* on non-prohibition memories.

The work splits three ways now, in priority order:

1. **Make the contradiction layer match what natural prose looks like (Track 0 — v0.13).** The headline post-generation verification claim. See "Track 0" below.
2. **Make the substrate installable for laymen (Track 1).** Already mostly shipped. Maintenance.
3. **Validate externally (Track 2).** Replace dogfood-only evidence with numbers from benchmarks we didn't author.

Internal architecture's *scaffolding* is ahead of these (graph types, governance shell, MCP integration, plugin architecture). The *mechanics* of cross-fact reasoning are behind.

## Track 0 — Paraphrase-aware contradiction (priority, 4 weeks → v0.13.x)

Goal: `aether_fidelity` distinguishes supported / merely-compatible / hallucinated on real prose, not just template-aligned cases. The whole post-generation verification claim hinges on this.

### Phase 0 — Archaeology (1 day, before Phase A)

Don't rebuild what already exists in the lineage.

- Scan OG CRT (`~/Documents/ai_round2/CRT/core/`): `extract_facts` in `utils.py`, `semantic_substitute.py`, `semantic_connection_map.py`. May contain reusable slot-extraction primitives.
- Scan CRT-GroundCheck-SSE (`/Volumes/ex_video/ai/CRT-GroundCheck-SSE/personal_agent/`): `crt_semantic_anchor.py`, `auto_fact_checker.py` second-pass filter. The slot-canonicalization shape may already be implemented.
- Per-agent behavioral validation of the 6 immune agents (~10 min each). Tonight's bench only confirmed they load and wire into `GovernanceLayer`; per-agent triggering is unverified.

**Output:** either "we can reuse X from GroundCheck" (shaves ~3 days from Phase A) or "starting fresh is correct."

### Phase A — MVP slot canonicalization (1 week → v0.13.0)

Goal: `aether_fidelity` passes the chef-in-Paris and vim-over-emacs cases.

Scope:
- Extend `aether/memory/slots.py` (or new `aether/memory/slot_canon.py`) for ~10 common categories: `user.employer`, `user.location`, `user.role`, `user.editor`, `user.language`, `project.vector_store`, `project.embedding_dim`, `project.framework`, `project.timeline`, `project.version`. Hard-coded synonym map.
- In `compute_grounding`, after the existing detectors fire/fail, run a slot-canonicalized check. Use the existing `slot:k=v` tag infrastructure on memories.
- Read-side only. Do not wire into write-time contradiction detection yet.
- Disposition: draft slot with no substrate match → ungrounded; draft slot value ≠ substrate value → contradicts; draft slot value = substrate value → supports.

**Done criterion:**
- `aether_fidelity("Nick is a chef in Paris.")` returns `belief_confidence < 0.4` against a substrate where no `user.role` or `user.location` slot matches.
- `aether_fidelity("Nick uses emacs primarily.")` returns ≥1 contradiction against a substrate with `user.editor=vim`.
- 10+ regression tests in `tests/test_v130_slot_canon_fidelity.py` covering tonight's failing cases.

### Phase B — Write-time slot conflict + coverage expansion (1 week → v0.13.1)

Goal: Test #3's paraphrased cases (768 vs 384, Option A vs Option B in prose) fire as contradictions at write time.

Scope:
- Wire the slot-canonical detector into the contradiction-detector cascade in `add_memory`, alongside `_is_policy_contradiction`, `_is_asymmetric_negation_contradict`, `mutex`, `shape`, `slot_eq`.
- Expand slot vocab to ~30 categories, mined from tonight's bench data + a scan of CRT's `DECISIONS.md` and `ROADMAP.md`.
- Add typed-value parsers: integer / float / date / version / categorical.
- Disposition: numeric/version conflicts → resolvable or evolving (based on temporal gap); categorical → resolvable.

**Done criterion:**
- Test #3 reflexive bench: Cases A and B both produce 1 contradiction edge each at write time, with the correct disposition.
- The "uses FAISS" vs "did not pick Pinecone" co-policy case stays clean (no false positive — the v0.12.21 selection-rejection guard still applies).
- DECISIONS.md ingest: false-positive count drops from 17 → ≤1. (The 3 remaining principle-pair flags from the structural tension meter may persist; that's a separate code path.)

### Phase C — Polish, bench corpus, perf (1 week → v0.13.2)

Goal: a reusable bench corpus + the architecture holds at scale.

Scope:
- Build `bench/paraphrase_corpus.jsonl` — 50 cases. ~25 from tonight's bench, ~25 synthesized to cover edge cases (pluralization, case, multi-token values, unicode).
- Run the corpus before-and-after on every change to the contradiction detectors. Wire into CI.
- Materialize the slot index (currently linear scan) for O(1) lookup.
- Edge cases: pluralization (table/tables), capitalization, multi-token values (San Francisco), unicode, non-English fragments.

**Done criterion:**
- Bench corpus has 50 yes/no cases. Aether agrees with the human verdict on >85%.
- p95 latency on a 10k-memory substrate: `aether_fidelity` <50ms, `aether_search` <100ms.

### Phase D — Honest reframe of public docs (1 week → v0.13.3, parallelizable with C)

Goal: README / bench / paper claims match what aether actually does.

Scope:
- README: update contradiction-detection claims to reflect post-Phase-B behavior. Hedge anything still aspirational (multi-turn stance flips, splat-based geometric overlap, predictive contradiction — Tier 4 / THEORY.md territory).
- `bench/README.md`: cold-vs-warm column on every category (existing open item).
- The "88% slot vs 40% LLM-as-judge" claim — source it with a reproducible script or remove. (Existing open item — `project_aether_open_items.md` #2.)
- Add a "what aether does and doesn't do (yet)" section that names the limits clearly.
- Update the paper-in-flight to reflect what's defensible.

**Done criterion:**
- Anyone running the published bench gets results within ±5% of the README's claims.
- Every empirical claim in the paper has a `bench/` reproduction script.

### Deferred until Phase D ships (was previously called "Tier 3" / "Tier 4")

| Tier | Work | Reason for deferral |
|---|---|---|
| Tier 3a | Port `disclosure_policy.py` from GroundCheck — yellow-zone clarification + budget | Builds on a working fidelity grader. Without Phase A/B, CLARIFY routes are grounded in noise. |
| Tier 3b | Port `compute_volatility(drift, alignment, contradiction, fallback)` | Same — needs cascade and grounding to work first. |
| Tier 3c | Port `crt_critic.py` PASS/SOFT/HARD pattern (post-generation revision loop) | Same. |
| Tier 3d | Port `commitments.py` — first-class commitment type distinct from belief | Independent of fidelity. Lower priority than fixing the headline claim. |
| Tier 4a | Splats with diagonal Σ + Bhattacharyya overlap (THEORY §1, §2) | Months. Paper work. After v0.13 is real. |
| Tier 4b | Context-dependent covariance modulation (THEORY §5) | Same. |
| Tier 4c | Multi-turn stance-flip detection (GroundCheck data: 93% span 3+ turns) | Same — but the bench corpus from Phase C becomes the seed dataset. |
| Tier 4d | The 4th `Contextual` disposition (THEORY §3d) | Restoring a lost CRT concept. |

### Untested by the 2026-05-01 bench

The validation pass exercised the contradiction / grounding / sanction layer. Things that exist as code but weren't behaviorally verified:

- The 6 immune agents at the per-agent triggering level (loaded + wired confirmed).
- Auto-link `RELATED_TO` edges — produced edges in tonight's runs, similarity scoring not verified at scale, and one false-positive `contradicts` edge surfaced from the structural tension meter on related paraphrases.
- Belnap state automatic transition (currently doesn't track trust — m1 trust 0.9 → 0.0 left belnap_state at `T`).
- `structural tension meter` on natural prose at scale (the 3 `flag_for_review` principle-pair false positives in DECISIONS.md ingest are this code path's bench data).

Address in Phase 0 (immune agents) and Phase B (auto-link / tension meter cleanup as a side effect of slot-canonical integration).

## Track 1 — Layman onboarding (priority, 1-3 weeks)

Goal: a stranger from the Claude Discord can install via the README's quickstart and reach a working substrate in <10 minutes without filing an issue.

| # | Item | Status |
|---|---|---|
| 1 | One-command install (SessionStart auto-installs, auto-upgrades, auto-warmups, auto-inits) | **shipped v0.12.14** |
| 2 | First-run welcome message via SessionStart `additionalContext` | **shipped v0.12.14** |
| 4 | Encoder install resilience (auto-warmup, fail-soft to cold mode, clear remediation) | **shipped v0.12.14** |
| 6 | Update notification — `aether status` and `aether doctor` flag version drift vs PyPI | **shipped v0.12.14** |
| 7 | `aether uninstall-cleanup` (dry-run by default, `--keep-substrate` preserves data) | **shipped v0.12.14** |
| 3 | Visible aether activity (PostToolUse hook? statusline?) | **deferred** — feasibility uncertain |
| 5 | Landing page (one-pager site, GitHub Pages) | **deferred** — non-code, separate session |

#3 needs investigation: surfacing "aether held this action" or "memory injected" requires either a PostToolUse hook (might work for sanction verdicts), Claude Code statusline (good for ambient state but not events), or a feature request to Anthropic. The fallback for now is the SessionStart welcome telling users to look for aether tool calls in Claude's responses, plus running `/aether-status` periodically.

**Done criteria.** Run `claude plugin install github.com/blockhead22/aether-core` on a fresh Mac, restart Claude Code, send any prompt → see the substrate's welcome message and confirm via `aether doctor` that all 7 checks pass without manual intervention.

## Track 2 — External validation (parallel, 1-3 weeks)

Goal: aether has at least one published benchmark number that wasn't authored by us.

1. **Run [EQL-Bench](https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai/tree/main/evals) against aether.** Their `evals/compression_faithfulness.py` (~$3, n=50) tests qualifier preservation through compression. Direct overlap with aether's `GapAuditor` (Law 5). Even a mediocre score is the first non-dogfood number.
2. **Cross-reference marker libraries** with Credence (MIT). Diff their HEDGING / ANCHORS / SELF_CORRECTIONS lexicons against aether's `template_detector.py`. Either import what's missing or document why aether deliberately doesn't.
3. **Add no-substrate baseline** to `bench/validation_test1.py`. Same questions, empty StateStore. The diff is the substrate's measurable value-add.
4. **Reproduce the 88%-vs-40% claim** (or retract it). The README's "structure beats semantics" line cited this number for a long time without a reproducible experiment behind it. Build the experiment: a fixed corpus of contradiction-detection cases, run aether's structural meter, run GPT-4 / Claude as a judge on the same cases, publish the comparison. v0.12.16 hedged the claim in the README; a real measurement closes the loop.
5. **Close F#12 / F#13** only if EQL-Bench surfaces them as material; otherwise defer.
6. **Add semantic entropy** (Kuhn et al., Nature 2024). Aether is structural-only today; entropy gives a *model-internal* uncertainty signal aether can't see. ~100-200 lines.
7. **Real install smoke test in CI.** v0.12.16 shipped because no test exercised the actual `claude plugin install` flow — the marketplace.json absence and the author-field schema bug both made it through. Add a CI job that installs the plugin from a fresh marketplace add, runs SessionStart, and asserts `aether doctor` returns 7 OK. Without this, the next plugin-schema regression will silently break layman onboarding for everyone.

**Done criteria.** A README badge or numbers section pointing at a non-self-authored benchmark with aether's score on it.

## Track 3 — Content + community (2-6 weeks)

Goal: people outside the repo know aether exists and what it's for.

1. **Blog post.** "I built a substrate that caught itself shipping bugs." 14 dogfood data points (every release-day fix v0.12.1 through v0.12.14) + 3 validation-harness findings (F#11/12/13) + EQL-Bench numbers + Credence comparison. Mostly drafted across NEXT_SESSION; needs an hour of polish.
2. **arXiv preprint.** Cascade complexity paper (drafted in `papers/cascade_complexity/`). Date-stamps the math; gives the substrate a citable foundation.
3. **Comparison post.** aether vs [Credence](https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai) vs Letta vs Mem0 vs Zep vs Cognee. Honest about overlap and complementarity.
4. **Outreach to Credence author.** Explicitly asked for feedback. Cross-pollination on markers, benchmarks, complementary positioning.

**Done criteria.** Hacker News / Reddit / Discord post(s) producing >0 outside installs that actually try the system and either succeed or file an issue we can fix.

## Track 4 — Hosted product MVP (2-3 months)

Goal: the paid tier exists. Open core stays open; hosted features are commercial.

1. **Cross-account substrate.** A team can share a substrate without each developer running their own. Storage tier swappable from local JSON to a hosted backend.
2. **Audit dashboard.** Sanction verdicts, cascade trails, contradiction history visible to a team admin.
3. **Sanction governance API.** Production agents call a hosted `/sanction` endpoint instead of running aether locally.
4. **Open-core boundary documented.** Written guarantee of what stays MIT and what's commercial. Required for enterprise procurement.

**Done criteria.** At least one team paying for the hosted tier.

## Track 5 — Research depth (3-6 months)

Goal: aether's primitives have formal foundation, not just engineering taste.

1. **Semantic uncertainty integration** (Kuhn et al. 2024 paper-based). Adds entropy as a second uncertainty channel orthogonal to the structural one.
2. **AGM belief revision formalization.** aether's contradiction-state model is informally AGM-shaped; reading the actual axioms might tighten the model.
3. **Variance probe.** Per-model fragility characterization. Same prompt, several LLMs, measure where the belief/speech gap diverges. Useful as both diagnostic and portability evidence.
4. **Training-time epistemic objectives.** Speculative — modify compression objectives during fine-tuning to preserve uncertainty. Heavy lift, possibly out of scope for solo work.

**Done criteria.** At least one peer-reviewed publication or workshop accept.

## Track 6 — Library rolling work

Things that aren't deal-breakers but accumulate:

1. **`aether.adapters`.** Cross-vendor adapters for Anthropic, OpenAI, Ollama. The portability claim ("the model is the mouth, the substrate is the self") needs runnable proof across vendors.
2. **`aether.compaction`.** Belief-aware context compaction with trust-tiered compression. Currently in the private codebase.
3. **`aether.session_state`.** Running belief state with an away/resume diff so a returning agent knows what it missed.
4. **Held-contradiction lifecycle.** Active → Settling → Settled → Archived state machine. The enum exists; the policies don't.
5. **Mutex class registry expansion.** Language runtimes, cache layers, queue systems, monitoring vendors, IaC tools.

## Out of scope (intentional)

- A full assistant or chatbot UI. Aether is middleware. Building Aether-the-assistant as OSS competes with frontier consumer AI and dilutes the substrate framing.
- LLM calls inside the core library. Optional adapters can use them; the core stays structural.
- Hosted multi-user features in OSS. Those are the paid Aether tier, not the open-source library.
- Vendor lock-in to any single LLM provider. The point is portability across mouths.

## Philosophy

This roadmap is what is likely, not what is promised. Solo project. Cinema and photography sometimes win.

If something in Track 1 or 2 is overdue and you care about it, file an issue or open a PR. Working code beats roadmap entries.
