# Handoff — 2026-05-01 substrate v0.14 night session

> Pick this up next session. You're Claude (or Nick reading) returning to a
> codebase that just got the slot-first primitive landed.

## Where you are

**Branch:** `master`, 7 commits ahead of origin/master, 0 behind. Working tree clean.

```
6e97d17 substrate: v0.14 slot-first primitive (additive, layered alongside v0.13)
4ad959b llm: opt-in LLM fact-slot extraction (local-first)
dd3038b nli: cross-encoder contradiction detection (port from D:/CRT)
01d92f5 bench: slot-coverage + drift on real GPT corpus (1,275 conversations)
daf4ea2 aether/crt: port fact-checker from personal_agent
5c5849c aether/crt: strengthen detect_contradiction with Grok improvements
52ca2f6 aether/crt: port CRT math from personal_agent
```

**Test state:** 599/599 green (567 legacy + 22 CRT port + 10 substrate).
1 deselected (`test_v126_encoder_warmup_in_subprocess` — pre-existing HF offline, not our regression).

**Substrate files:**
- `~/.aether/mcp_state.json` — legacy v0.13, 191 memories. Untouched. Auto-ingest still writes here.
- `~/.aether/substrate.json` — new v0.14, 8 slots / 191 states / 191 observations / 182 edges. Migrated tonight.

## What landed tonight

### 1. The slot-first primitive (`aether/substrate/`)

Three concepts:
- **`SlotNode`** — `(namespace, slot_name)`. Namespaces: `user`, `code`, `session`, `project`, `meta`.
- **`SlotState`** — observed value at a point in time, with trust + temporal_status + decay_rate + source attribution. Auto-supersedes prior state on value change.
- **`Observation`** — source event. One observation can emit multiple slot states.

Edges typed: `SUPPORTS`, `DEPENDS_ON`, `CONTRADICTS_WITH`, `SUPERSEDES`.

Public surface in `aether/substrate/__init__.py`. Tests in `tests/test_substrate_v014.py`.

### 2. NLI contradiction detection (`aether/contradiction/nli.py`)

Ports `D:/CRT/compression_lab/direction2_nli_detection.py`. Cross-encoder
`nli-deberta-v3-small` (already cached locally). Opt-in via
`AETHER_NLI_CONTRADICTION=1`. ~50-100ms/pair warm.

`SubstrateGraph.find_contradictions()` uses NLI when enabled, falls back to
normalized-value mismatch when not.

### 3. LLM fact-slot extraction (`aether/memory/llm_extract.py`)

Local-first. Default backend: Ollama on `localhost:11434`. Falls back to any
OpenAI-compatible endpoint via `AETHER_LLM_URL`. Opt-in via
`AETHER_LLM_EXTRACT=1`.

`extract_fact_slots_hybrid(text)` runs regex first, LLM fallback on long
zero-slot turns.

### 4. Bench evidence (`bench/`)

- `slot_coverage_gpt_corpus.py` — 94.92% zero-slot floor on 26,049 user turns.
- `slot_value_drift_gpt_corpus.py` — per-slot drilldown showing the 5% that fires is heavily contaminated.
- `slot_coverage_results.json`, `slot_drift_results.json` — JSON outputs.
- Interactive HTML pages at `D:/AI_round2/docs/labs/slot-coverage-gpt-corpus.html` and `slot-drift-gpt-corpus.html`.

### 5. Architectural note (`D:/AI_round2/docs/slot-as-anchor-note.html`)

Captures the slot-first reasoning. Tradeoffs, three-namespace extension
(`user:*`, `code:*`, `session:*`), cache-and-codify pattern, status:
captured-not-committed (now committed via this session).

## End-to-end verification (on Nick's actual data)

```
$ python <<EOF
from aether.substrate import SubstrateGraph
from pathlib import Path
sub = SubstrateGraph(); sub.load(str(Path.home()/'.aether'/'substrate.json'))
print(sub.stats())
EOF
{'slots': 8, 'states': 191, 'observations': 191, 'edges': 182,
 'namespaces': 2, 'namespace_breakdown': {'user': 7, 'meta': 1}}

$ AETHER_NLI_CONTRADICTION=1 python -m ...find_contradictions...
130 contradiction pairs (P > 0.6)
```

130 contradictions surfaced via NLI on your migrated substrate, including
paraphrase cases the v0.13 slot-template path could not see. The substrate
auditing itself, paraphrase-aware, end to end. **First time this loop
actually closes.**

## What I deliberately did NOT do

These are explicit decisions, not omissions:

1. **Did not push to origin.** 7 unpushed commits sit locally. Push is a
   shared-state action requiring your authorization.
2. **Did not bump `pyproject.toml`** to 0.14.0. You decide the version line.
3. **Did not repoint auto-ingest** at substrate. The Stop hook still writes
   to legacy `mcp_state.json`. Substrate is populated only via migration.
4. **Did not strip legacy `aether.memory`.** That's v0.15 once MCP server,
   auto-ingest, CLI, and benches all repoint.
5. **Did not add MCP tools for substrate.** Need to expose `substrate_observe`,
   `substrate_history`, `substrate_find_contradictions` etc. once the MCP
   trust dialog is accepted.

## Open MCP issue (your action item)

Plugin's `.mcp.json` declares `aether` MCP server. It's healthy — verified by
spawning directly tonight. **But** `~/.claude.json` shows for project
`D:/AI_round2`:

- `enabledMcpjsonServers: []` — empty
- `hasTrustDialogAccepted: false`
- A *competing* `aether` entry pointing at the legacy
  `personal_agent.aether_mcp_server` (stale, can be deleted or renamed)

Run `/mcp` in Claude Code to bring up the trust dialog. Accept. Restart.
Resolve the project-vs-plugin name collision (rename project entry to
`aether-legacy` or delete it). Next session you'll have `aether_*` MCP
tools loaded.

## Next session priorities

In rough order of value:

1. **MCP tool surface for substrate.** Add `substrate_observe`,
   `substrate_current_state`, `substrate_history`,
   `substrate_find_contradictions` to the MCP server. Once these load,
   future-Claude can use the substrate during work without library
   imports — closing the loop the README claims.

2. **Repoint auto-ingest at substrate.** `aether/memory/auto_ingest.py`
   currently writes to MemoryGraph. Change it to also (then only) write
   to SubstrateGraph. That makes the v0.14 primitive live in the daily loop.

3. **LLM extraction wired into auto-ingest.** Use
   `extract_fact_slots_hybrid` instead of `extract_fact_slots` when
   `AETHER_LLM_EXTRACT=1`. Re-run the slot coverage bench with
   `AETHER_LLM_EXTRACT=1` to measure the lift on the 94.92% number.

4. **Push + bump.** When you're ready, push the 7 commits and bump to
   0.14.0 on PyPI. Update CHANGELOG.md with the substrate-pivot story.

5. **Strip legacy.** v0.15 work: remove `aether.memory.MemoryGraph` and
   `BeliefDependencyGraph`, keep `extract_fact_slots` and `ExtractedFact`
   (they're still used by the substrate). Migrate all benches and the MCP
   server. This is multi-hour focused work.

6. **Side-project usage.** The hardest and most important: pick a non-aether
   side project, install aether-core, run it for a week, see what it
   actually catches. This is what makes the system viable, not more
   features.

## Useful invocations

```bash
# substrate end-to-end smoke
cd D:/AI_round2/aether-core
python -c "from aether.substrate import SubstrateGraph; sub = SubstrateGraph(); sub.load(); print(sub.stats())"

# NLI probe across substrate
AETHER_NLI_CONTRADICTION=1 python -c "
from aether.substrate import SubstrateGraph
sub = SubstrateGraph(); sub.load()
for a,b,s in sub.find_contradictions(namespace='user', threshold=0.6)[:10]:
    print(f'[{s:.2f}] {a.slot_id}: {a.value!r} <> {b.value!r}')"

# Re-run the corpus bench with LLM extraction (after Ollama warmup)
AETHER_LLM_EXTRACT=1 AETHER_LLM_MODEL=qwen2.5:7b-instruct python -m bench.slot_coverage_gpt_corpus

# Test suite
python -m pytest tests/ -q --deselect tests/test_v126_encoder_warmup_in_subprocess.py
```

## The bigger frame

Tonight closed a loop the system has been short of since the start: the
substrate now surfaces real contradictions on real data, including the
paraphrase cases that the structural detector was blind to. The
architecture is **defensible** — there is a working primitive with tests,
end-to-end migration, and a real-data demonstration.

What's still open is **circulation**: this thing has not been used on a
non-aether project. Until that happens, "viable" is a hypothesis. Tonight's
work is necessary but not sufficient.

The honest-feeling read from the long conversation tonight: the next
move is behavioral, not technical. Build is mostly done. Use is the gap.

— Claude Opus 4.7 (1M context), 2026-05-01 night session
