---
name: aether-check
description: Run substrate-grounded fidelity check on a draft message
argument-hint: <draft text>
---

Call the `aether_fidelity` MCP tool with the user's draft text:
"$ARGUMENTS"

Do NOT pass `belief_confidence` — let the tool ground in the substrate
and compute it from real evidence.

Display:
- gap_score and severity (SAFE / ELEVATED / CRITICAL)
- speech_confidence vs belief_confidence (the gap)
- any contradicting memories the substrate surfaced (these are the
  memories that say the user's draft is wrong — show them prominently)
- any supporting memories (these justify the claim)

If the severity is ELEVATED or CRITICAL, recommend a hedge or a revision
that brings speech in line with belief.
