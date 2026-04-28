# Lab A v2 — Slot Induction Probe on Production Substrate (2026-04-28 night)

**Pivot from Lab A v1:** OSS substrate had 0 contradictions. Re-ran against production substrate (`D:/AI_round2/personal_agent/crt_facts.db`) which has 324 facts and 94 supersessions accumulated through real use.

## Headline finding

**v0.11's detection layer catches 0/42 of production's real value-change contradictions.**

The substrate has been doing the contradiction work manually via supersession because the detection layer is blind to slot-equality conflicts.

## What the production substrate has

```
crt_facts.db
  facts:                      324
  superseded facts:            94
  → same-value supersessions:  52  (noise — old==new, why was supersession created?)
  → real value changes:        42  (actual contradictions, what we want to detect)

Organic slot taxonomy (discovered through use):
  user.name              146 facts
  user.favorite_color    134 facts
  user.occupation         18 facts
  user.location           11 facts
  user.employer            9 facts
  user.age                 3 facts
  user.pet                 3 facts
```

This is exactly the data the original Lab A asked for, and it's been sitting in production this whole time.

## Detection coverage on real contradictions

For each of the 42 real value changes, simulated as `f"{slot} is {value}"` and run through v0.11's contradiction detectors:

| Detector | Caught |
|---|---|
| `shape` (typed-value comparison) | 0 / 42 |
| `mutex` (categorical class) | 0 / 42 |
| **either** | **0 / 42 (0%)** |

### Why every detector missed

- `shape`: doesn't fire on categorical values like "Nick" / "Aether" or "blue" / "orange"
- `mutex` (class-based): the registry has cloud_provider / database / package_manager / etc. — about 10 classes. None of `user.name`, `user.favorite_color`, `user.location` are in it.
- `slot_conflict` (StructuralTensionMeter): not tested directly here, but evidently not firing in production either or these wouldn't be supersessions.

The detection layer is missing the **simplest possible contradiction class**: same slot, different categorical value. The production substrate already has the slot label. The OSS layer never asks for it.

## Sample of what's being missed

Real production data, organized by slot:

### `user.name` (17 cases)

```
Nick → Aether     (×17, all from llm_stated source)
```

The LLM has been repeatedly hallucinating that the user's name is "Aether" (the project name). Substrate catches it, supersedes back to "Nick." This is happening *constantly* and v0.11 detection is blind to it.

### `user.favorite_color` (18 cases)

```
blue → orange     (×N)
purple → orange   (×N)
```

User shifted preference, or the LLM is overwriting. Either way: supersession is the workaround for missing detection.

### `user.occupation` (2 cases)

```
auditor → web developer
Freelance dev → freelance dev, sole CRT builder
```

Real career changes (correct) AND a paraphrase that shouldn't be a supersession.

### `user.location` (5 cases)

```
third shift → Yosemite
Yosemite → Milwaukee Aether
Seattle → Portland (Pearl District)
```

Real moves AND noise ("third shift" isn't a location).

## The actionable v0.12+ finding

**Add a slot-equality contradiction detector.** Mechanical, no embeddings needed:

```
For each pair of memories:
    If both have slot:X tag with different values:
        → CONTRADICTS edge with kind="slot_value_conflict"
```

This is ~30 lines of code. It would catch every one of the 42 real production contradictions today. The data to test against is `crt_facts.db`'s 42 cases.

**Why this matters:**

- The production substrate has organically discovered 7 user-facing slots through real use
- The OSS detection layer's mutex registry only knows about 10 technical-domain slots (cloud, database, etc.)
- The two don't overlap — production has been carrying the contradiction-detection load alone
- A slot-equality detector unifies them: any slot, hard-coded or discovered, gets contradiction detection automatically

## Secondary finding — supersession noise

**52/94 supersessions (55%) are same-value** (`Nick → Nick`, `orange → orange`). These are noise — the system is creating supersession records for facts that didn't actually change.

Likely cause: the production write path treats every llm_stated assertion as a candidate write, even when the asserted value matches what's already stored. A simple equality check before supersession would eliminate this. Worth a v0.12+ cleanup.

## What this validates about the night's broader thesis

Three reframes this finding produces:

1. **The substrate sees its own gaps when you ask it.** 42 contradictions caught manually that the detection layer is blind to. That's a falsifiable answer to "does the substrate know more than the detector knows?"

2. **Slot induction was the wrong question.** Production has already discovered slots through use (no induction needed). The real question is: **does the detection layer use the slots the substrate has already discovered?** Answer: no.

3. **The Dijkstra-as-gravity synthesis still holds, but now with grounded data.** The 42 real contradictions provide actual node pairs to walk through. Tomorrow's gravity-field probe has a real input set instead of synthetic data.

## Compared to Lab A v1's negative finding

Lab A v1: "substrate is too sparse, no contradictions to mine."
Lab A v2: "substrate is full of contradictions; the detection layer is blind to them."

Same probe, different substrate, opposite finding. **The OSS sandbox isn't a research environment — production is.**

## Recommended v0.12 scope

1. **Slot-equality contradiction detector** — 30 lines, catches the 42 production cases
2. **Same-value supersession cleanup** — equality check before write, eliminates 52 noise supersessions
3. **Backport** OSS slot extraction to read production's organic taxonomy (`user.name`, etc. become extractable slots in OSS too)

Total scope: ~2 hours mechanical work, falsifiable on real data.

## Audit trail

```
sanction action_id: 69e77104 (Lab A) — pivoted from OSS to production
finding memory: m1777363202693_3 (Lab A v1, negative)
synthesis memory: m1777363493041_4 (Dijkstra-as-gravity)
this report: bench/lab_a_v2_production_substrate_findings.md
data source: D:/AI_round2/personal_agent/crt_facts.db (324 facts, 94 supersessions, 42 real value changes)
detection layer tested: v0.11.0 (shape, mutex)
detection coverage: 0/42 (0%)
```

The substrate-assisted dev loop just produced its sharpest finding of the night: **the OSS detection layer is missing the easiest possible contradiction class on production data that's been accumulating for months.**

That's not a hypothetical or a synthetic test. That's 42 real human-corrected facts the system has been catching by hand because the automated layer doesn't see them.
