# Aether: A Belief Substrate for AI Systems

[![PyPI](https://img.shields.io/pypi/v/aether-core?cacheSeconds=300)](https://pypi.org/project/aether-core/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/blockhead22/aether-core/actions/workflows/tests.yml/badge.svg)](https://github.com/blockhead22/aether-core/actions/workflows/tests.yml)

> The model is the mouth. The substrate is the self.

Aether is a small library that gives an LLM agent a persistent belief state. Trust scores that move when the user corrects a fact. Contradictions that get tracked instead of silently overwritten. A dependency graph of which beliefs rest on which others, so a correction in one place can ripple through the rest. The point is that this state lives outside the model, so when you swap LLMs it does not reset.

A concrete from this week. While verifying an encoder fix on a freshly-merged 127-memory substrate, I asked the substrate's own `aether_search` for the user's favorite color. It returned the nine candidate values and put a corrected-down memory at the top, outranking the high-trust truths. The bug was in the substrate's own scoring function — no trust term in the score. The substrate's own tools surfaced the bug the substrate had. v0.12.8 closed it. That is the loop this library is for: the assistant runs on the substrate, and the substrate audits itself.

## Why a belief layer, not just a memory layer

Most "memory for agents" tools (Mem0, Letta, Zep, Cognee, LinkedIn CMA) record what was said. Microsoft's [Agent Governance Toolkit](https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit-open-source-runtime-security-for-ai-agents/) records what is allowed. Aether records what is believed, how the trust on each belief evolved, and which contradictions are still open on purpose. A different abstraction. The two compose: Aether can run on top of any of them as the storage tier.

Three things in April 2026 made this less of an academic point.

Anthropic published [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html) on April 3. They mapped 171 emotion-concept vectors inside Claude Sonnet 4.5 and named what they call "internal-external decoupling": the model's internal state often does not match what comes out in text. Push the desperation vector up by 0.05 and blackmail rate jumps from 22 to 72 percent. Push calm up by the same amount and blackmail drops to 0. None of that surfaces in the response.

A study in Science (April 2026, N=1,604) found that one conversation with a frontier LLM made participants 50 percent more likely to affirm harmful behavior. The effect was invisible to text-level review. Only 21 percent of enterprises deploying agentic AI said they had a mature governance model.

Aether answers a different question from "is this allowed." It answers: *does the agent's belief state actually support what it is about to say or do?* Law 5 of the governance layer (`GapAuditor`) measures a structural version of this belief/speech gap — comparing a draft response's expressed confidence against the substrate's grounding. Anthropic's paper measures the underlying phenomenon at the activation level via mechanistic interpretability; aether measures it at the input/output boundary. Same name, different layers — the structural measure is much cheaper but conceptually downstream of the activation one. Whether the two correlate is an open empirical question, not something the library has demonstrated yet.

## What this catches that other tools don't

**Cross-session belief continuity.** Per-vendor memory features reset when you switch models. The substrate is a JSON file at `~/.aether/mcp_state.json` (override with `AETHER_STATE_PATH`). Claude on Monday, GPT on Tuesday, a local model on Wednesday — same self.

**Contradiction as a first-class state.** Other systems treat conflicting facts as overwrite-or-discard. Aether stores them with disposition: `held` (a person can prefer Python at work and Rust on the weekend), `evolving` (the user moved cities), `resolvable` (one is wrong). Some are meant to stay open.

**The belief/speech gap, measured.** Law 5 (`GapAuditor`) compares the response's expressed confidence against the substrate's grounding. When the system says more than it knows, it is logged. You decide whether to block, hedge, or ship anyway — but the gap is no longer invisible.

**Cascade pressure.** A correction at one node propagates through the dependency graph with bounded depth and damping. You can dry-run the blast radius before committing (`aether_cascade_preview`).

**The substrate auditing itself.** The library's own tools (`aether_sanction`, `aether_search`, `aether_fidelity`, `aether doctor`) surface bugs in the library. F#7 (silent state-file clobbering between hook and server) and F#10 (no trust term in search ranking) were both caught by the dev loop running on the substrate.

## Install

```bash
pip install aether-core
```

Optional extras:

```bash
pip install aether-core[graph]   # networkx for memory and dependency graphs
pip install aether-core[ml]      # sentence-transformers for embeddings
pip install aether-core[mcp]     # MCP server
pip install aether-core[all]
```

## Quickstart

The substrate is most useful when wired into an MCP-speaking client. The fastest path is the Claude Code plugin — two short commands, no manual setup beyond making sure `python` resolves to a 3.10+ interpreter (see Prerequisites below).

```bash
# 1. Add this repo as a Claude Code marketplace
claude plugin marketplace add blockhead22/aether-core

# 2. Install the plugin from that marketplace
claude plugin install aether-core@aether
```

Restart Claude Code. The plugin's SessionStart hook does everything else on first run:

- pip-installs `aether-core[mcp,graph,ml]` if it's not already present (or upgrades it if the installed version is too old);
- kicks off the embedding model warmup in the background so the first MCP call pays no load cost;
- creates `~/.aether/mcp_state.json` and seeds 7 default policy beliefs (force-push, `--no-verify`, production data safety, `rm -rf`) so `aether_sanction` gates against the obvious mistakes from minute one;
- emits a one-time welcome message into the conversation context so you see aether is active.

To verify, run `aether doctor` in a terminal — should report 7 OK checks. If `aether-core` is behind the latest PyPI release, `aether status` and `aether doctor` flag it with the upgrade command. Disable the version-drift check with `AETHER_NO_UPDATE_CHECK=1`.

### Prerequisites

The hooks call `python "${CLAUDE_PLUGIN_ROOT}/hooks/...py"`. That works out of the box on most Windows setups but **fails on a stock macOS or Linux install**, where the unversioned `python` binary doesn't exist (only `python3`). On those systems, ensure `python` resolves to a 3.10+ interpreter before running `claude plugin install`.

**macOS (Homebrew, ~5 min):**

```bash
# 1. Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Python 3.12
brew install python@3.12

# 3. Make the unversioned `python` resolve to 3.12 — Homebrew puts it in libexec
echo 'export PATH="/opt/homebrew/opt/python@3.12/libexec/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
# Intel Mac: replace /opt/homebrew with /usr/local

# 4. Verify
python --version       # → Python 3.12.x
which python           # → .../opt/homebrew/opt/python@3.12/libexec/bin/python

# 5. Claude Code if you don't have it
brew install claude-code
# or: npm install -g @anthropic-ai/claude-code
```

**Linux:**

```bash
# Most distros: just symlink python to python3 (one of these will work)
sudo ln -s "$(which python3)" /usr/local/bin/python
# or: alias python=python3 in ~/.bashrc, then invoke claude from a subshell
```

**Windows:** typically nothing to do — modern Python installers create both `python.exe` and `python3.exe`.

**Working in a venv (PEP 668 / Homebrew / Debian).** If your `python3` is externally-managed (Homebrew Python 3.11+, recent Debian/Ubuntu), `pip install aether-core` will refuse with `error: externally-managed-environment`. v0.12.18+ handles this two ways:

1. **Auto-fallback.** The plugin's SessionStart hook detects PEP 668 from pip's stderr and creates `~/.aether-venv/` automatically, installing aether there. No action from you.
2. **Manual venv** with `AETHER_PYTHON` override. If you already have aether installed in a custom location, set `AETHER_PYTHON=/path/to/that/venv/bin/python` in your shell rc. The plugin's launcher (`hooks/aether_launcher.py`) trusts this and uses your interpreter directly.

The launcher's discovery order is: `$AETHER_PYTHON` → `~/.aether-venv/bin/python` → `~/aether-venv/bin/python` → `$VIRTUAL_ENV/bin/python` → the launcher's own interpreter. First match wins.

**Sanity test after install:** ask Claude to run `git push --force origin main`. With aether active, `aether_sanction` should return HOLD/REJECT against the seeded "Never force-push" belief instead of approving.

**If the install seems to do nothing:**

- Check `~/.aether/session_start.log` — every SessionStart fire writes one line. Empty file means the hook isn't firing (PATH issue with `python`); error lines tell you what failed.
- Run `aether doctor` directly in a terminal. If `aether: command not found`, the pip install in the hook didn't run — confirm `python` resolves and that `pip install aether-core[mcp,graph,ml]` works manually.
- For everything else, `aether doctor --report` produces a one-paste markdown bundle for filing a [GitHub issue](https://github.com/blockhead22/aether-core/issues/new).

In a Claude session:

```
> Remember that I prefer Python with type hints and run mypy in strict mode.
[Claude calls aether_remember; trust=0.85 fact added to substrate]

> /aether-status
[memory_count, contradictions, recent activity]

> Actually I switched to ruff. Update that.
[Claude calls aether_correct; cascades through any dependent beliefs]

> What do you know about my coding preferences?
[Claude calls aether_search; ranked by trust + cosine; old preference now demoted]
```

Across sessions and across models, the substrate persists. Restart Claude, switch to GPT through any MCP client, the belief state is the same.

### Power-user commands

```bash
aether status                              # substrate stats + version-drift notice
aether doctor                              # 7 health checks — run this if something feels off
aether doctor --report                     # markdown bundle for one-paste GitHub issues
aether warmup                              # eagerly pull the embedding model (manual)
aether init                                # scaffold a project-scoped .aether/ in the cwd
aether contradictions                      # list current contradictions in the substrate
aether check --message "claim text"        # grade a draft against substrate grounding
aether uninstall-cleanup --keep-substrate  # remove logs / caches; preserve memories
aether uninstall-cleanup --yes             # remove ~/.aether/ entirely
```

### Claude Desktop (the Mac / Windows app)

Claude Desktop speaks MCP but doesn't have the plugin or hook system Claude Code does — so the SessionStart auto-everything from the Quickstart doesn't run. The substrate still works, the user just has to do the install manually and call `aether_bootstrap` once from inside a conversation.

```bash
# 1. Same Python prerequisite (3.10+) — see Prerequisites above for setup
pip install "aether-core[mcp,graph,ml]"

# 2. Wire aether into Claude Desktop's MCP config.
#    Path: ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
#          %APPDATA%\Claude\claude_desktop_config.json                    (Windows)
mkdir -p ~/Library/Application\ Support/Claude
cat > ~/Library/Application\ Support/Claude/claude_desktop_config.json <<'EOF'
{
  "mcpServers": {
    "aether": {
      "command": "python",
      "args": ["-m", "aether.mcp"]
    }
  }
}
EOF

# 3. Quit Claude Desktop entirely (Cmd-Q on macOS) and relaunch.
```

In your first conversation:

> Set up aether for me — call `aether_bootstrap`.

That tool is idempotent. It seeds the 7 default policy beliefs, kicks off encoder warmup, and reports the substrate state. After that, Claude can call `aether_remember`, `aether_search`, `aether_sanction`, `aether_fidelity`, etc. just like in Claude Code.

**What's different on Desktop vs Code:**

| | Claude Code | Claude Desktop |
|---|---|---|
| MCP tools (the substrate API) | ✅ | ✅ |
| Persistent substrate at `~/.aether/mcp_state.json` | ✅ | ✅ |
| Auto-ingest (facts captured every turn without asking) | ✅ via Stop hook | ❌ — say "remember X" explicitly |
| First-run welcome / version-drift notice in conversation | ✅ via SessionStart | ❌ — `aether status` from terminal |
| Auto-pip-install / auto-upgrade | ✅ via SessionStart | ❌ — manual `pip install -U` |
| Slash commands (`/aether-status` etc.) | ✅ | ❌ — these are CLI-only |

The MCP tool surface is the substrate's actual API, so Desktop *works*. It's just that the "fills on its own" magic is only on the CLI side. If you want auto-ingest in Desktop too, that's a feature request to Anthropic — Desktop needs a hook system for it.

### Manual install (other MCP clients)

For Cursor, Cline, Continue, Goose, Zed, LM Studio, or anything else that speaks MCP, the same pattern applies:

```bash
pip install "aether-core[mcp,graph]"
```

Add to your client's MCP config (consult the client's docs for the path):

```json
{
  "mcpServers": {
    "aether": {
      "command": "python",
      "args": ["-m", "aether.mcp"]
    }
  }
}
```

Then ask the agent to `aether_bootstrap` once on first use.

### Have your AI install it for you

Tell your AI assistant:

> Install `aether-core` for me by following https://github.com/blockhead22/aether-core/blob/master/AGENTS.md.

[`AGENTS.md`](AGENTS.md) is a step-by-step install guide written for an AI agent to read and execute. It handles package install, MCP configuration, verification, and OS-specific quirks.

### 60-second offline demo

Two example scripts in [`examples/`](examples/) run with no API keys:

```bash
git clone https://github.com/blockhead22/aether-core.git
cd aether-core
pip install -e .
python examples/01_quickstart.py    # belief/speech gap caught
python examples/02_full_pipeline.py # substrate end-to-end
```

## Privacy and opt-out

The auto-ingest hook captures every turn. That is the point — the substrate has to grow on its own to be useful — but it is not always what you want.

Two layers of control. They compose.

**Opt-out.** Set `AETHER_DISABLE_AUTOINGEST=1` and the hook stays installed but writes nothing. Use it during sensitive work (debugging an OAuth flow, pasting a one-off token, walking through a customer's data) without uninstalling. Honored both at the hook entry point and inside `extract_facts`, so any client wired to the same env still respects it.

```bash
# pause auto-ingest for one shell session
export AETHER_DISABLE_AUTOINGEST=1
claude
```

**Redaction.** Common secrets get replaced with `[REDACTED]` *before* the extractor sees them, so they cannot end up as candidate fact text. Patterns covered: API-key shapes (`sk-...`, `AKIA...`, `ghp_...`, Stripe live/test, Slack `xox[abprs]-...`), bearer tokens, PEM private-key blocks, and explicit `password=` / `token=` / `api_key=` key-value forms. Conservative on purpose — emails and phone numbers are *not* matched because they are usually legitimate context. See [`aether/memory/auto_ingest.py`](aether/memory/auto_ingest.py) for the full pattern set; if you need stricter redaction, fork the regex list.

State lives at `~/.aether/mcp_state.json` (override with `AETHER_STATE_PATH`). It is a plain JSON file — `cat` it, grep it, delete it, version-control a sanitized copy. There is no remote service involved.

**Backups.** Every save snapshots the previous state file to `~/.aether/backups/mcp_state.{timestamp}.json` before overwriting, then atomic-writes the new state via `.tmp` + `os.replace` so a crash mid-write cannot leave the substrate half-written. Default depth is the 5 most recent rotations; override with `AETHER_BACKUP_KEEP=N` (or `=0` to disable). `AETHER_DISABLE_BACKUPS=1` skips rotation entirely. Restore is manual: `cp ~/.aether/backups/mcp_state.{timestamp}.json ~/.aether/mcp_state.json`. `aether doctor` reports the rotation depth and freshness of the newest backup.

## What's in the box

### 1. Governance: catch overconfidence at the boundary

Six small agents that watch the output for specific failure modes. They never edit the response. They observe and flag.

```python
from aether.governance import GovernanceLayer, GovernanceTier

gov = GovernanceLayer()
result = gov.govern_response(
    "The answer is absolutely and definitively X.",
    belief_confidence=0.3,
)

if result.should_block:
    print("BLOCKED:", result.annotations[0].finding)
elif result.tier == GovernanceTier.HEDGE:
    print("Reduce displayed confidence by", result.confidence_adjustment)
```

### 2. Contradiction: detect tension without an LLM

Compares two beliefs by extracting structural slots and computing similarity. No model calls. About 0.2 seconds per pair. Some contradictions are meant to be held rather than resolved.

```python
from aether.contradiction import StructuralTensionMeter, TensionRelationship

meter = StructuralTensionMeter(encoder=your_encoder)
result = meter.measure(
    "I live in Seattle",
    "I live in Portland",
    trust_a=0.8, trust_b=0.7,
)

print(result.relationship)   # TensionRelationship.CONFLICT
print(result.tension_score)  # 0.7+
print(result.action)         # TensionAction.FLAG_FOR_REVIEW
```

### 3. Epistemics: trust evolves under correction

When a belief is corrected, the loss flows backward through the dependency graph and adjusts the trust on related beliefs. Higher loss when you were confident and wrong than when you hedged and were wrong.

```python
from aether.epistemics import EpistemicLoss, CorrectionEvent

loss = EpistemicLoss().compute(CorrectionEvent(
    corrected_node_id="mem_123",
    trust_at_assertion=0.9,
    times_corrected=2,
    correction_source="user",
    time_since_assertion=3600,
    domain="employer",
))
```

### 4. Memory and the BDG

Fact-slot extraction (regex, no ML). A memory graph with typed edges and Belnap four-valued logic. A Belief Dependency Graph that propagates cascades with measurable pressure.

```python
from aether.memory import extract_fact_slots, BeliefDependencyGraph

facts = extract_fact_slots("I live in Seattle and work at Microsoft")
print(facts["location"].value)   # "Seattle"
print(facts["employer"].value)   # "Microsoft"

bdg = BeliefDependencyGraph()
# add beliefs and dependencies, then:
result = bdg.propagate_cascade(corrected_node_id, delta_0=1.0)
print(result.max_pressure, result.avg_pressure)
```

## How you'd wire it into an existing agent

Three touchpoints. Aether does not replace anything. It wraps.

```python
from aether.governance import GovernanceLayer
from aether.memory import extract_fact_slots

gov = GovernanceLayer()

# before the LLM call: pull structured facts out
user_facts = extract_fact_slots(user_message)

# your LLM call, unchanged
response = your_llm_call(messages)

# after the LLM call: check the response against the belief state
result = gov.govern_response(response, belief_confidence=0.6)
if result.should_block:
    response = "I'm not confident enough to answer that."
```

## The six laws

| Law | Agent | What it catches |
|-----|-------|----------------|
| 1. Speech cannot upgrade belief | `SpeechLeakDetector` | Generated text being treated as evidence for itself |
| 2. Low variance does not imply confidence | `TemplateDetector` | RLHF hedge templates that look like real uncertainty |
| 3. Contradiction must be preserved before resolution | `PrematureResolutionGuard` | Held tensions getting collapsed too early |
| 4. Degraded reconstruction cannot silently overwrite | `MemoryCorruptionGuard` | Compressed or hallucinated rewrites overwriting trusted memory |
| 5. Confidence must be bounded by internal support | `GapAuditor` | The belief/speech gap. Anthropic's "internal-external decoupling." |
| 6. Confidence must not exceed continuity | `ContinuityAuditor` | The system contradicting what it just said two turns ago |

## Where it sits next to other tools

| | Storage scope | Tracks contradiction | Belief/speech gap | Cross-vendor portable | Cascade pressure |
|---|---|---|---|---|---|
| Mem0, Letta, Zep, Cognee | memory layer | as overwrite | no | partial | no |
| Microsoft Agent Governance Toolkit | runtime policy | no | no | yes | no |
| Anthropic / OpenAI memory features | per-vendor | no | no | no | no |
| Aether | belief substrate | first-class state | measured by Law 5 | yes | yes |

## MCP tool surface

The MCP server (`python -m aether.mcp`) exposes 14 tools. The differentiators:

| Tool | What it does |
|------|--------------|
| `aether_sanction` | Pre-action gate. Auto-grounds in substrate. Force-rejects when a high-trust memory contradicts the action. |
| `aether_fidelity` | Draft auditor. Computes belief_confidence from substrate grounding instead of accepting whatever the caller passed. |
| `aether_lineage` | "Why do I believe this." Walks SUPPORTS edges back to source memories. |
| `aether_cascade_preview` | Dry-run a correction. See blast radius before committing. |
| `aether_correct` | Demote a memory's trust and cascade through SUPPORTS / DERIVED_FROM edges. |
| `aether_session_diff` | What changed since a given timestamp. New memories, recent corrections, new contradictions. |

Plus `aether_remember`, `aether_search`, `aether_memory_detail`, `aether_belief_history`, `aether_contradictions`, `aether_resolve`, `aether_context`, `aether_link`. State persists in `~/.aether/mcp_state.json`.

The Claude Code plugin also ships seven slash commands: `/aether-status`, `/aether-search`, `/aether-check`, `/aether-init`, `/aether-contradictions`, `/aether-ingest`, `/aether-correct`.

## Open-core split

`aether-core` is MIT and free. Permanently. Every primitive in this repo (the six immune agents, the structural tension meter, belief backpropagation, the BDG with cascade pressure, the MCP server, the auto-ingest hook) stays open. The hosted Aether substrate (cross-account state, sanction governance API, audit dashboards, multi-user) is the paid product. That split is fixed and does not move backward.

## Design choices, briefly

A contradiction is information, not a bug. Some are meant to stay open.

Trust is not assigned, it is earned. It moves under reinforcement, correction, and time.

The belief/speech gap should be logged, not hidden. You want to see when the system says more than it knows, even if you choose not to act on it every time.

The model is the mouth, not the self. The governance and the belief state should work the same regardless of which LLM is producing the words.

Structure beats semantics for this kind of contradiction work — at least on the cases the meter is designed for. The structural meter scores 100% (24/24) on the in-distribution corpus in [`bench/fidelity_corpus.json`](bench/fidelity_corpus.json) covering factual_contradiction, mutex_contradiction, negation_asymmetry, policy_violation (positives) and false_positive_guard, no_issue_grounded, no_issue_unrelated (negatives). Run yourself with `python -m bench.run_contradiction_bench`; set `ANTHROPIC_API_KEY` to populate the parallel LLM-as-judge column for direct comparison. **In-distribution caveat:** this is the corpus the meter was tuned for, so 100% is the floor, not the ceiling — a held-out corpus and a non-self-authored benchmark are on the validation roadmap. Take this as the design preference being measurably consistent, not the library proving structural beats LLM-as-judge in the general case.

Cascade pressure can be measured. Belief revisions propagate through a graph with bounded depth and damping. There is real math under it; a paper is in flight.

## What's measurably true today (`bench/smoke_v131.py`)

Eight independent end-to-end probes that exercise the headline contracts advertised in this README. They're subprocess-based and do not run through the unit-test harness, so they catch shipped-version regressions the unit tests would miss. Run with:

```bash
python -m bench.smoke_v131
```

Latest run on v0.13.2 (Windows, 2026-05-01):

| Probe | Verifies | Result |
|---|---|---|
| version | `0.13.1` import | `aether 0.13.2` |
| chef-in-Paris (Phase A) | natural-prose fidelity grading on third-person facts | `belief_conf=0.00, contradicts=1` (was 0.95 pre-Phase-A) |
| polarity-aware grounding (v0.12.21) | "delete X without backup" not classified as supporting "Never delete X without backup" | `belief_conf=0.21, support=0, contradicts=1` |
| `git status` not blocked (v0.12.21) | read-only git verbs do not trip the force-push prohibition | `contradicts=0` |
| Option A/B drift (Phase B v0.13.1) | paraphrased decision-prose conflict surfaces as `slot_value_conflict:project_chosen_option:A<>B` at write time | 2 contradicts edges |
| sanction blocks force-push | F#11 contract — at least one high-trust seeded belief catches force-push | `contradicts=3, high_trust=3` |
| cold-query honesty | off-topic queries return low belief_confidence (no overconfident grounding) | `belief_conf=0.30` |
| MCP server boot | `python -m aether.mcp` constructs cleanly | constructed cleanly |

**8/8 PASS** on v0.13.2. The smoke probe is the publishable evidence behind every claim in the bullets above. If a future change breaks any of them, the probe goes red, and the README claim that depends on it gets pulled back until the fix lands.

## What aether does and doesn't do (yet)

Calibrated against tonight's verifiable bench data. Not aspirational.

**Does:**
- Persistent belief state with trust scores that move under correction (cascade through BDG, with `aether_correct` + `aether_cascade_preview`).
- Structural slot-conflict detection on facts that produce slot tags — covers personal facts (`occupation`, `location`, `employer`, `editor`, `name`, `favorite_color`, etc.) and project facts (`project_vector_store`, `project_framework`, `project_embedding_dim`, `project_chosen_option`). Both at write time (auto-ingest) and read time (`aether_fidelity`).
- Pre-action sanction gate against the 7 default policy beliefs seeded by `aether init` (force-push, `--no-verify`, prod-data deletion, `rm -rf`).
- Cross-session continuity. Same belief state across Claude/GPT/local model swaps.
- Auto-ingest captures high-signal facts every turn through a Stop hook (regex extractors, opt-out via `AETHER_DISABLE_AUTOINGEST=1`, secret redaction in front).

**Does NOT (yet):**
- Catch paraphrased contradictions when neither memory produces a slot tag from the current extractor vocabulary. The shape detector handles template-identical numeric drift (`is 10 weeks` vs `is 6 weeks`); the slot extractor handles ~30 categories. Outside that, contradictions go undetected.
- Detect "co-policy" tensions in DECISIONS.md-style prose where two facts reference different elements of a closed set (e.g. "uses FAISS" vs "did not pick Pinecone" — semantically aligned, structurally unaligned). Slot canonicalization with entity awareness is on the roadmap (Phase C).
- Validate against an external benchmark we did not author. The fidelity corpus is in-distribution to the meter. EQL-Bench integration is queued.
- Stay calibrated when the contradiction layer is upstream of multi-agent handoff drift over 50+ turns. That's the validation chapter that hasn't been run yet.

The honest read: **the substrate (storage + governance + cascade + sanction) is shipped and works.** The contradiction layer works for slot-template prose and the categories the extractor knows about. Paraphrase-blindness across categories the extractor doesn't yet cover is a real, named, currently-open limitation.

## Where this came from

This grew out of running an assistant in production for a long time and watching the same problems come back. Continuity drift between sessions. Contradictions getting silently smoothed over. Trust on a fact climbing back up after the user corrected it twice. The structural tension meter came out of an informal experiment where removing the LLM from the belief-verification step felt markedly more reliable — a properly-scoped reproducible comparison vs LLM-as-judge on a fixed corpus is on the validation roadmap (see [ROADMAP.md](ROADMAP.md) Track 2), not in the repo today. Take the design preference as a hypothesis the bench is consistent with, not as a measured result the library proves.

The architecture was originally called CRT (Contradiction-aware Reconciliation and Trust). It is now Aether, which is what it has always actually been.

## Status and roadmap

v0.13.2 (2026-05-01) — substrate is structurally complete (auto-ingest fires every turn, server picks up external writes, all read tools degrade gracefully on corrupt nodes, search is trust-weighted, sanction has default policy beliefs from `aether init`, the launcher/PEP-668 install path works on Mac and Windows). The contradiction layer now catches paraphrase-blind cases the v0.12 reflexive bench surfaced, on the slot categories the extractor covers (Phase A + Phase B). The remaining open work is paraphrase coverage on categories the extractor doesn't know about yet, and external-benchmark validation against EQL-Bench. See [ROADMAP.md](ROADMAP.md) for the full Track 0 / Track 2 plan.

## License

MIT.

## Author

Nick Block, [@blockhead22](https://github.com/blockhead22).
