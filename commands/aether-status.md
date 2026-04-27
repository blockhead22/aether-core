---
name: aether-status
description: Show the current Aether substrate state (memory count, edges, contradictions)
---

Call the `aether_context` MCP tool to fetch the current substrate state.

Display the result in a compact summary:
- memory count
- edge count
- which Belnap states are present and how many of each
- held vs evolving contradictions
- whether embeddings are available
- the path to the on-disk state file

If the substrate is empty (memory_count == 0), tell the user to either:
1. Use `/aether-init` to scaffold a `.aether/` directory in this repo, or
2. Just start telling the assistant facts — the auto-ingest hook will fill it.
