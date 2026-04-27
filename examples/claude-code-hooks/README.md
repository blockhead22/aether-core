# Claude Code Hooks for Aether

Optional integrations that turn a Stop event from Claude Code into a
substrate write. None of these are required — `aether-core` is fully
usable through the MCP tools alone. These exist because the substrate
is most useful when it fills *automatically* and most users won't
remember to call `aether_remember` mid-flow.

## `auto_ingest_hook.py`

Reads the last turn from Claude Code on stdin, pulls high-signal facts
out via `aether.memory.ingest_turn`, and writes them to the local
substrate.

### What it captures

Conservative regex-based extractor. Six rule families:

| Signal | Trust | Pattern |
|---|---|---|
| `user_preference` | 0.85 | "I prefer ...", "We always use ..." |
| `user_identity` | 0.90 | "I am ...", "I work at ...", "I live in ..." |
| `project_fact` | 0.85 | "This repo uses ...", "We deploy to ..." |
| `decision` | 0.80 | "We decided to ...", "Let's go with ..." |
| `constraint` | 0.92 | "Never ...", "Don't ...", "You should never ..." |
| `correction` | 0.93 | "Actually ...", "Wait, no ...", "That's wrong, ..." |

Plus an `assistant_observation` rule at trust 0.6 for things like "I
see that this codebase uses ...".

### Install

1. Install `aether-core` with the MCP extra:

   ```bash
   pip install "aether-core[mcp,graph]"
   ```

2. Copy `auto_ingest_hook.py` somewhere stable (e.g. `~/.aether/hooks/`).

3. Wire into `.claude/settings.json`:

   ```json
   {
     "hooks": {
       "Stop": [
         {
           "matcher": "*",
           "hooks": [
             {
               "type": "command",
               "command": "python ~/.aether/hooks/auto_ingest_hook.py"
             }
           ]
         }
       ]
     }
   }
   ```

4. Restart Claude Code. The hook fires after every assistant turn
   ends. Watch stderr for ingestion logs:

   ```
   [aether auto-ingest] wrote 2 fact(s) from the last turn
     - (user_preference) trust=0.85 -> m1773456789012_1
     - (project_fact) trust=0.85 -> m1773456789013_2
   ```

### Tuning

The extractor errs conservative — you'll miss many turns but rarely
write garbage. To make it more aggressive:

- Add new rule patterns to `USER_RULES` in
  `aether/memory/auto_ingest.py`.
- Lower `_REJECT_PATTERNS` strictness.
- Raise `max_facts` (default 8) if you want more captures per turn.

To make it less aggressive:

- Raise the per-rule trust thresholds and have the calling code drop
  anything below e.g. 0.85.
- Tighten the regexes.

### Failure mode

If the hook errors, it logs to stderr and exits 0 — Claude Code
continues normally. The substrate is never required for the assistant
to function. Worst case: stale or sparse substrate.
