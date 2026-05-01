# Roadmap

What's coming next, when we expect it, and what is intentionally not in this repo.

For what's already shipped, see [CHANGELOG.md](CHANGELOG.md).

## Where we are (2026-05-01)

`aether-core` is at **v0.12.14** on PyPI. 434 unit tests pass on master. The substrate is **structurally complete** — every read tool degrades gracefully on corrupt nodes, every write tool syncs from disk, the auto-ingest hook fires every turn, secrets get redacted before fact extraction, every save snapshots the prior state, `aether_sanction` has default policy beliefs to gate against, and the SessionStart hook auto-installs / auto-upgrades / auto-initializes on first run with a welcome message visible to the user.

Remaining work splits two ways:

1. **Make the substrate installable for laymen.** A Claude Code or Cursor user who's never heard of MCP should reach a working substrate in under 10 minutes from a single `claude plugin install` command, with no docs needed for the happy path.
2. **Validate externally.** Replace dogfood-only evidence with numbers from benchmarks we didn't author and users who aren't us.

Internal architecture is ahead of these. Surface area and external evidence are behind.

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
4. **Close F#12 / F#13** only if EQL-Bench surfaces them as material; otherwise defer.
5. **Add semantic entropy** (Kuhn et al., Nature 2024). Aether is structural-only today; entropy gives a *model-internal* uncertainty signal aether can't see. ~100-200 lines.

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
