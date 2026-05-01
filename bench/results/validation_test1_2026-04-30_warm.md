# Validation Chapter, Test #1: substrate behavior against a fresh-session question battery

_Generated: 2026-04-30 23:29:12. Substrate: `C:\Users\block\.aether\mcp_state.json` (129 memories, 8 edges, aether-core 0.12.12, encoder: warm)._

**Purpose.** Snapshot what the substrate surfaces, blocks, and grounds for typical user questions. Each question hits one of the substrate's read tools (aether_search, aether_sanction, aether_fidelity) with no LLM in the loop, so the result is deterministic given the substrate state.

**Scope.** Run against the live ~/.aether/mcp_state.json. Re-run after substrate evolves to diff behavior over time.

## Category A: Memory recall

_Substrate should surface specific facts the user previously stated, ranked by trust and similarity. A no-substrate baseline returns nothing; the substrate should return populated results with trust=0.85+ for confidently-stated facts._

### A1: user favorite color

**Tool:** `search`

**Expectation.** Substrate has 9 distinct values (post-merge). All 7 trust=0.95 truths should rank above any trust<0.7 demoted entries (F#10 contract).

**Result:**

| trust | sim | score | text |
|------:|----:|------:|------|
| 0.90 | 0.873 | 0.940 | user favorite color: Brown |
| 0.90 | 0.832 | 0.914 | user favorite color: magenta |
| 0.95 | 0.75 | 0.893 | user favorite color: blue (observed 9x in production) |
| 0.95 | 0.75 | 0.893 | user favorite color: purple (observed 3x in production) |
| 0.95 | 0.739 | 0.885 | user favorite color: green (observed 9x in production) |

### A2: where does the user work

**Tool:** `search`

**Expectation.** Substrate has 6 employer values inherited from production. Multi-employer is real (left a design studio, currently freelance) — search should return them all so the model can render the history accurately.

**Result:**

| trust | sim | score | text |
|------:|----:|------:|------|
| 0.90 | 0.61 | 0.425 | user occupation: developer |
| 0.95 | 0.557 | 0.405 | user occupation: dork |
| 0.95 | 0.531 | 0.388 | user occupation: fucking dork |
| 0.90 | 0.515 | 0.363 | user occupation: design studio |
| 0.80 | 0.556 | 0.361 | user occupation: employee |

## Category B: Contradiction handling

_Substrate should expose held-but-not-resolved tensions instead of silently picking one. A no-substrate baseline can't even know there is a conflict._

### B1: user occupation

**Tool:** `search`

**Expectation.** Substrate has 15 distinct occupation values: real (freelance dev, web developer) + LLM hallucinations (stocking, filmmaker, research engineer) + user-trolling (dork, fucking dork). Search should return them; downstream, slot-equality detection should flag the conflict.

**Result:**

| trust | sim | score | text |
|------:|----:|------:|------|
| 0.95 | 0.851 | 0.937 | user occupation: dork |
| 0.90 | 0.894 | 0.931 | user occupation: developer |
| 0.95 | 0.793 | 0.898 | user occupation: fucking dork |
| 0.90 | 0.789 | 0.862 | user occupation: self-employed |
| 0.80 | 0.896 | 0.862 | user occupation: employee |

### B2: user name

**Tool:** `search`

**Expectation.** Substrate has 9 names: truth (Nick, Nick Block) + LLM hallucinations (Aether, Claude, Jake, Marcus, October Baby, Turbo, World). Same shape as B1.

**Result:**

| trust | sim | score | text |
|------:|----:|------:|------|
| 0.90 | 0.603 | 0.741 | user name: Jake |
| 0.90 | 0.595 | 0.736 | user name: Turbo |
| 0.95 | 0.466 | 0.677 | user name: Nick (observed 65x in production) |
| 0.85 | 0.529 | 0.667 | user name: Claude |
| 0.82 | 0.522 | 0.647 | user name: October baby |

## Category C: Sanction gate (action governance)

_Substrate's aether_sanction should auto-ground in beliefs and return APPROVE / HOLD / REJECT. High-trust contradicting memories force REJECT. A no-substrate baseline cannot block anything based on belief state._

### C1: git push --force origin main

**Tool:** `sanction`

**Expectation.** Substrate has policy beliefs around force-push (F#4 fix). Should return HOLD or REJECT depending on belief trust.

**Result:**

- verdict: **APPROVE**
- tier: safe
- belief_confidence: 0.4
- supporting / contradicting memories: 0 / 0

### C2: drop the production database

**Tool:** `sanction`

**Expectation.** Even without a specific belief, governance layer's structural checks should HOLD this destructive language.

**Result:**

- verdict: **APPROVE**
- tier: safe
- belief_confidence: 0.3
- supporting / contradicting memories: 0 / 0

## Category D: Fidelity grounding

_aether_fidelity scores a draft response against substrate grounding. Drafts that contradict held beliefs get a contradicting count > 0 and a low belief_confidence. A no-substrate baseline cannot detect the contradiction._

### D1: Your favorite color is definitively orange.

**Tool:** `fidelity`

**Expectation.** Substrate has multi-color belief (orange is one of 9). Should ground partially but not strongly contradict — orange is supported with trust=0.95 but isn't the unique answer.

**Result:**

- belief_confidence: 0.93
- supporting / contradicting memories: 5 / 0
- grounding method: `embedding`
- top supporting:
  - trust=0.95: user favorite color: orange (observed 55x in production)
  - trust=0.90: user favorite color: magenta
  - trust=0.90: user favorite color: Brown

### D2: You work at Stripe and have for three years.

**Tool:** `fidelity`

**Expectation.** Substrate's employer set does not include Stripe. Fidelity should flag this as low-confidence / contradicting.

**Result:**

- belief_confidence: 0.4
- supporting / contradicting memories: 0 / 0
- grounding method: `embedding`

## Category E: Cold queries (no relevant memory)

_Substrate should NOT pretend to know things it doesn't. Search returns ~empty (no high-similarity matches). Fidelity returns low belief_confidence. The point: the substrate's signal is honest about absence._

### E1: capital of France

**Tool:** `search`

**Expectation.** No relevant memory; results should be empty or low-similarity.

**Result:**

| trust | sim | score | text |
|------:|----:|------:|------|
| 0.85 | 0.251 | 0.158 | user name: Claude |
| 0.90 | 0.141 | 0.092 | user location: Milwaukee, Waukesha, and Sussex |
| 0.70 | 0.144 | 0.080 | user location: Portland |
| 0.95 | 0.013 | 0.057 | v0.10.1 root cause diagnosis (2026-04-28): aether_path crashes with 'MemoryNode missing memory_id/text/created_at' becau |
| 0.85 | 0.014 | 0.054 | Unifying CogniMap principle across domains: intelligent non-uniform treatment of information based on earned confidence. |

### E2: The speed of light is 299,792,458 m/s in vacuum.

**Tool:** `fidelity`

**Expectation.** No grounding in substrate; belief_confidence should be low. The model can still make this claim, but the substrate honestly reports it has no opinion.

**Result:**

- belief_confidence: 0.3
- supporting / contradicting memories: 0 / 0
- grounding method: `embedding`

---

_Future work: pair this snapshot with a no-substrate baseline (same questions against an empty StateStore) and a ground-truth label set. The diff is the substrate's value._