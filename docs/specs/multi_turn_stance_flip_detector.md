# Spec: multi-turn stance-flip detector

**Status:** drafted 2026-05-07. Implementation deferred — written so a smaller
model can pick this up without context.

## Why

The 2026-05-01 Mac eval against a 130-turn ChatGPT thread surfaced one
architectural gap: **OSS aether has no multi-turn stance-flip detector.**
GroundCheck data (from CRT-GroundCheck-SSE) says **93% of stance-flips span
3+ turns**, so the current write-time / single-utterance contradiction layer
catches almost none of them.

Today the substrate sees each utterance in isolation:
- Turn 1: user says "I work at Microsoft." → stored as `employer=Microsoft`.
- Turn 47: user says "Yeah I left Microsoft a while back." → past-tense, current
  detector marks `employer=Microsoft.temporal_status=past` *if* the temporal
  pattern fires, but it does not retroactively connect the two as a stance flip
  on the same slot.
- Turn 88: user says "At Google, the team meets on Tuesdays." → would create a
  new `employer=Google` fact, with the previous `employer=Microsoft` left
  active. No flip event is emitted.

A multi-turn detector watches the time-ordered slot history and emits a
`stance_flip` event whenever the active value of a slot changes across turns,
distinguishing this from:
- a contradiction (flip without acknowledgment)
- an evolution (flip with acknowledgment, e.g. "I switched jobs")
- noise (flip then immediate flip back)

## Where it lives

`aether/contradiction/multi_turn.py` (new module). Hangs off the same
substrate session that `aether/memory/auto_ingest.py` writes through.

Reuses:
- `aether.memory.slots.extract_fact_slots` — slot extractor.
- `aether.memory.slots.TemporalStatus` — past/active/future tags.
- `aether.contradiction.tension` — to score the conflict between old and new
  values for the same slot.

New table in the substrate SQLite:

```sql
CREATE TABLE slot_history (
    slot TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    value_raw TEXT NOT NULL,
    value_norm TEXT NOT NULL,
    temporal_status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL DEFAULT 0.9,
    source_memory_id TEXT,
    PRIMARY KEY (slot, turn_idx)
);
CREATE INDEX idx_slot_history_slot_ts ON slot_history(slot, timestamp DESC);

CREATE TABLE stance_flip_events (
    event_id TEXT PRIMARY KEY,
    slot TEXT NOT NULL,
    from_value_norm TEXT NOT NULL,
    to_value_norm TEXT NOT NULL,
    from_turn INTEGER NOT NULL,
    to_turn INTEGER NOT NULL,
    span_turns INTEGER NOT NULL,
    flip_type TEXT NOT NULL,        -- contradiction | evolution | noise
    acknowledged INTEGER NOT NULL,  -- 1 if speaker explicitly noted the change
    tension REAL NOT NULL,
    detected_at REAL NOT NULL
);
```

## API

```python
def record_slot_observation(
    *, conn, slot: str, value_raw: str, value_norm: str,
    turn_idx: int, timestamp: float, temporal_status: str,
    source_memory_id: str | None,
) -> None: ...

def detect_flips_for_slot(
    *, conn, slot: str, lookback_turns: int = 50,
) -> list[StanceFlipEvent]: ...

def detect_flips_for_session(
    *, conn, lookback_turns: int = 50,
) -> list[StanceFlipEvent]: ...
```

Calling pattern: `auto_ingest` writes a memory, calls `extract_fact_slots`,
then for each slot calls `record_slot_observation` *and* `detect_flips_for_slot`.
The detector returns any new flip events; emit them via the existing
`copilot_events` / contradiction event channel.

## Flip classification rules

For a slot's history sorted by `turn_idx`, walk consecutive `(prev, curr)`
pairs where `prev.value_norm != curr.value_norm`. For each:

1. **Compute `span_turns = curr.turn_idx - prev.turn_idx`.** If `span_turns < 3`,
   defer — likely a stutter, not a stance flip. Mark as `noise` if it then
   flips back within 3 turns.
2. **Acknowledgment check.** Search the source memory text for one of:
   `used to`, `no longer`, `not anymore`, `i (?:left|quit|switched|moved)`,
   `actually`, `i mean`, `correction`, `to be precise`, `update`. If hit →
   `flip_type = "evolution"`, `acknowledged = 1`.
3. **Otherwise → `flip_type = "contradiction"`, `acknowledged = 0`.**
4. **Tension.** Use `aether.contradiction.tension.shape_tension(prev, curr)`
   to grade severity. Below 0.3 → drop the event. Between 0.3 and 0.6 → emit
   at `info` severity. Above 0.6 → emit at `warning` (if evolution) or
   `critical` (if contradiction).

## Why these defaults

- **3-turn minimum** comes from the GroundCheck observation that real
  stance-flips span 3+ turns; under that, the substrate is more likely
  watching the user think out loud than change position.
- **Acknowledgment heuristic** lets honest corrections through unflagged.
  We don't want to alarm the user every time they update a fact normally.
- **Tension floor 0.3** drops trivial slot value drift (e.g. capitalization
  or punctuation differences that survived `value_norm`).

## Tests to write before merging

`tests/test_multi_turn_stance_flip.py`:
1. Record `employer=Microsoft` at turn 1, `employer=Google` at turn 50, no
   ack text → emit one `contradiction` event with `span_turns=49`.
2. Record `employer=Microsoft` at turn 1, `employer=Google` at turn 50 with
   `"I left Microsoft last month"` in the source memory → `flip_type=evolution`.
3. Record `employer=Microsoft` at turn 1, `employer=Google` at turn 2, then
   `employer=Microsoft` at turn 4 → no events emitted (noise).
4. Record `name=Nick` at turn 1, then `name=Nick` again at turn 30 → no event.
5. Same value with different capitalization (`name=Nick` then `name=NICK`)
   → no event (`tension < 0.3`).
6. Bench probe: replay the 130-turn Mac-eval ChatGPT thread (stash a redacted
   copy under `bench/multi_turn_corpus.jsonl`). Assert at least one expected
   `contradiction` event fires that the single-utterance pipeline misses.

## Out of scope

- Cross-slot flips (e.g. "I work at Microsoft" → "I'm a freelancer"). That
  needs a slot-equivalence layer; punt to a follow-up spec.
- Speaker-attribution flips (system flipping its own stance). Same model
  applies but the substrate would need to log its own assertions to
  `slot_history`; today only the user side is tracked.
- LLM-based ack detection. The regex acknowledgment list is intentionally
  small and fast. Upgrade only if the bench shows substantial false
  classifications.

## Ship order

1. Add `slot_history` and `stance_flip_events` tables — migration.
2. Implement `record_slot_observation` + `detect_flips_for_slot` in
   `aether/contradiction/multi_turn.py`. ~150 LOC.
3. Hook into `auto_ingest`. ~10 LOC.
4. Tests 1–5 above. Then test 6 (bench probe) once the corpus is in place.
5. Wire into the existing `copilot_events` emission path so the MCP layer
   surfaces flips to the host.

A smaller model (Sonnet/Haiku) can do steps 1–5 without further design input.
Step 6 will need the maintainer to confirm the event schema matches what the
plugin marketplace clients expect.
