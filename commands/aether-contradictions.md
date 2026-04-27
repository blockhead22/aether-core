---
name: aether-contradictions
description: List contradictions currently held in the Aether substrate
argument-hint: [resolvable|held|evolving|contextual]
---

Call the `aether_contradictions` MCP tool. If the user supplied an
argument ("$ARGUMENTS"), pass it as `disposition`. Otherwise list all.

Display each contradiction as a side-by-side pair with:
- Memory A: text + trust + belnap state
- Memory B: text + trust + belnap state
- Disposition (resolvable / held / evolving / contextual)
- Tension score
- Detected timestamp

For each, suggest a concrete action:
- **resolvable**: the user should pick a winner via `/aether-resolve`
- **held**: leave it — both are true at different times/contexts
- **evolving**: probably stale — investigate which is current
