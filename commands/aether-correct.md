---
name: aether-correct
description: Correct a memory and cascade the trust drop through dependents
argument-hint: <memory_id> [reason]
---

Parse "$ARGUMENTS" as `<memory_id> [optional reason text]`.

Call the `aether_correct` MCP tool with:
- `memory_id`: the parsed ID
- `reason`: any text after the ID (default: "user correction")
- `new_trust`: -1.0 (sentinel — let the tool halve current trust as a
  soft demotion)

Display:
- Old trust → new trust delta
- The list of cascade-affected nodes with their depths and trust deltas
- A reminder that the cascade walks SUPPORTS / DERIVED_FROM edges only,
  not CONTRADICTS, so contradictory memories are unaffected.

If the user wants a hard deprecation instead of a soft demotion, suggest
they re-run with `new_trust=0.0` via the MCP tool directly.
