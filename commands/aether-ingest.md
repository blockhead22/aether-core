---
name: aether-ingest
description: Manually ingest the last conversation turn into the substrate
---

Call the `aether_ingest_turn` MCP tool with the most recent user message
and your own most recent response, both pulled from this conversation.

Pass them as `user_message` and `assistant_response`.

Display the count of facts ingested. If any were written, list each
write briefly — the signal type (preference / decision / project_fact /
constraint / correction / observation), the trust assigned, and the new
memory_id.

If nothing was ingested, that's fine — the extractor is conservative on
purpose. Tell the user that explicitly so they don't think it failed.
