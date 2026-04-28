---
name: aether-path
description: Find the cheapest dependency chain in the substrate that grounds a query (Dijkstra over BDG)
argument-hint: <query>
---

Call the `aether_path` MCP tool with the user's query: "$ARGUMENTS"

Use the default `max_tokens=2000` and `max_hops=8`.

The tool runs Dijkstra backward from the top-1 cosine match over SUPPORTS / DERIVED_FROM / RELATED_TO edges, weighted by `(1 - trust) * token_estimate(text)`. High-trust memories are cheap to include; low-trust ones are expensive. CONTRADICTS edges are skipped entirely.

Display the result as a path with:
- The target (the most relevant memory to the query)
- Each ancestor in distance order: memory_id, text snippet, trust, distance, depth
- Total token cost vs. budget
- Number of closed paths skipped (CONTRADICTS edges encountered)

Frame the output as "to ground this query, the substrate says you need to know:" and list the path. If `closed_paths > 0`, mention that the substrate routed around held contradictions.

If the result has `method: "no_substrate"`, tell the user the substrate is empty — they should write some facts via `aether_remember` first.
