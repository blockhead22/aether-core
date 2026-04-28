# Lab A — Slot Induction Probe (2026-04-28 night)

**Question:** Does the substrate's existing contradiction structure carry enough signal to drive contradiction-driven slot induction?

**Method:** Walk every CONTRADICTS edge in production substrate (`~/.aether/mcp_state.json`). For each pair, extract `(frame, filler_pair)` via longest-common-subsequence. Cluster the frames; identify proto-slots.

**Result:** Negative finding, three actionable sub-findings.

## The substrate has zero CONTRADICTS edges

```
memories: 34 (29 live, trust > 0)
edges:    6 (4 supports, 2 related_to, 0 contradicts)
```

The original probe design assumed contradictions exist to walk. They don't. Despite hours tonight discussing contradictions as first-class state, the actual substrate has accumulated none.

**Why:** the contradiction detector requires *both* candidates to be present in the substrate at the same time AND topically similar enough to enter the top-K candidate scan. The substrate has been seeded with mostly-coherent compression facts and release notes; nothing in tonight's writes triggered an actual disagreement that survived to disk.

This is itself a meta-finding: **a substrate not lived-in long enough has no contradiction structure to mine.** Slot induction is gated on substrate maturity.

## Pivot probe: latent contradictions via shape + token overlap

To extract any signal from the substrate we have, ran v0.11's shape primitive against every pair of live memories (29 × 28 / 2 = 406 pairs):

```
shape conflicts found:        185 / 406 pairs
high-Jaccard pairs (>= 0.4):    4 / 406 pairs
real latent contradictions:     0
```

### Sub-finding 1: shape primitive without context gate produces massive false positives

185 shape "conflicts" were detected across the 406 pairs. All are spurious.

Example (the worst case):

> **A:** "aether-core v0.9.1 (commit 26ddf7d, 2026-04-27) ships three fixes for a v0.9.0..."
> **B:** "CogniMap lossless v1 ... beats raw zlib by 15.6%..."
>
> Reported conflicts: `version: 9.1 ↔ 15.6, integer: 189 ↔ 2031, ...`

These aren't conflicting versions. `9.1` is an aether version; `15.6` is a percentage improvement. They share zero conceptual context. The shape primitive matched them as comparable types because both look like decimal numbers. The integer `189` is a test count; `2031` is a kilobyte size. Different units, different meanings, no conflict.

**Why production isn't broken:** v0.11.0's wiring runs shape() inside `_detect_and_record_tensions` AFTER `if sim < 0.2: continue` — top-K candidate gating is the implicit context guard. The bench passed because the bench's seed memories are topically grouped.

**For slot induction (or any cross-memory comparison):** shape MUST be gated by context similarity. The blind-pairwise approach is unusable.

### Sub-finding 2: the gating combination *would* work — there's just no data

Combining Jaccard ≥ 0.4 (context gate) with shape() conflict detection yields:

| Step | Pairs |
|---|---|
| Total pairs | 406 |
| After Jaccard ≥ 0.4 (same-topic) | 4 |
| Of those, shape-conflicting | 0 |

The 4 high-overlap pairs are all *paraphrases* of the same fact (different ways of saying CogniMap v1 = 2031 KB beats zlib 15.6%). No conflicts. **Zero false positives, zero real conflicts.**

So the *design* of `Jaccard gate + shape conflict` is sound. It's a clean two-step filter: only pairs that share a topic AND disagree on a typed value get flagged. In the current substrate, that produces no signal because the substrate has no real disagreements — but it also produces no noise.

### Sub-finding 3: substrate density requirement

For slot induction to have testable signal, we'd need:

- **More memories** — current 29 live is too sparse for clustering
- **Intentional disagreements** — paired memories that genuinely conflict, not just paraphrase

Rough estimate: **50-100 memories with at least 10 genuine contradictions** before clustering has anything to cluster.

The substrate accumulates contradictions naturally as it's used over time (a developer disagreeing with their own past memory, an external source contradicting a stored fact, multi-version state of the world). Tonight's substrate skipped that lived-in phase — it's the result of seeded research notes plus session-end release memories, all written by one person across one day. Nothing here actually disagrees.

## What this tells us about v0.11+ direction

**The slot induction idea isn't wrong — it's blocked on substrate maturity.**

Three implications:

1. **The shape primitive needs a published guard rule.** Document that `shape()` is for *gated comparison* (within a topical neighborhood) and explicitly NOT for blind pairwise scanning. Add this to the docstring.

2. **Slot induction is a future-substrate experiment, not a present-substrate experiment.** Re-run this probe in 1-3 months when the substrate has lived through more disagreements. Until then, the technique can't be empirically validated on real data.

3. **The substrate's contradiction-emptiness IS data.** It tells us the current detection paths aren't catching things they could. Worth investigating: are there *implicit* disagreements between memories that the existing detectors miss? The four Jaccard-high paraphrase pairs are a reminder that paraphrase-vs-paraphrase isn't conflict — but maybe paraphrase-vs-paraphrase-with-different-numbers IS, and the substrate just hasn't encountered that case yet.

## What we did NOT find

- No emergent slot categories (no contradictions to mine)
- No comparison-function induction signal (same reason)
- No test of the v0.11 lab-framing's "contradictions clustering into meta-beliefs" idea (no clusters)

## Recommendation

**Don't ship the slot induction mechanism on this substrate.** Build it when the data is there. In the meantime:

- Document v0.11's shape primitive as gated-comparison-only
- Wait for substrate density to grow naturally through normal use (or seed it intentionally for testing)
- Keep the lab framing memory `m1777362414276_1` as the design intent for when data exists

## Next time — re-probe trigger conditions

Run this probe again when ANY of these become true:

- [ ] Substrate has ≥ 50 memories
- [ ] Substrate has ≥ 5 CONTRADICTS edges (auto-generated, not seeded)
- [ ] Substrate has ≥ 3 distinct topical clusters with ≥ 2 disagreeing memories per cluster
- [ ] User has lived with the substrate across ≥ 10 sessions

Until then: slot induction is a v0.12+ candidate. Foundation primitives (shape, token_overlap, ncd) shipped tonight; learned-pattern layer waits for data.

## Audit trail

```
sanction action_id: 69e77104-1850-4ab1-99dc-685ac7741698
verdict: APPROVE
result: success (negative finding documented)
verification_passed: true
verification_reason: probe ran successfully on production substrate; design conclusions falsifiable; recommendation grounded in observed data sparsity
```

Lab framing memory `m1777362414276_1` (v0.11+ research direction) preserved unchanged — slot induction remains the right idea, just blocked on substrate maturity.
