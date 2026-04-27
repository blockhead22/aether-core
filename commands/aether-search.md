---
name: aether-search
description: Search the Aether substrate for memories matching a query
argument-hint: <query>
---

Call the `aether_search` MCP tool with the user's query: "$ARGUMENTS"

Limit results to 8.

Display each hit with:
- The memory text
- Trust score
- Belnap state (and any warnings — contested, deprecated, uncertain)
- Source
- Similarity score (if embeddings used)

If a result has a `warnings` field, surface those prominently — the user
needs to know when a memory is contested or deprecated before relying on it.
