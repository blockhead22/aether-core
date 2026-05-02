"""Slot-coverage histogram on the GPT corpus.

What this measures
------------------
For every user turn in the 1,275-conversation export, run
``aether.memory.extract_fact_slots`` and record:

  * how many turns produced zero slot tags ("paraphrase-blind floor")
  * how many turns produced one or more slot tags
  * frequency of each slot category

The zero-slot percentage is the *floor* on paraphrase-blindness for the
contradiction layer: turns that produce no slot tag cannot have their
contradictions caught by the slot-template path, regardless of how
obviously contradictory they are at the prose level.

Run::

    python -m bench.slot_coverage_gpt_corpus

Reads the export from CORPUS_DIR (override via env). Prints a histogram
and writes JSON results to ``bench/slot_coverage_results.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from aether.memory import extract_fact_slots

DEFAULT_CORPUS = (
    "C:/Users/block/Downloads/"
    "fbf5a239c1af822f50241d4b5999b53954689723b9d4deb7ceb1e41a31847485-"
    "2026-03-27-22-39-55-cf0803cbc48c446cb8c3b317ea5f11ea"
)


def iter_user_turns(corpus_dir: Path):
    """Yield (conv_id, turn_text) for every non-empty user turn."""
    for f in sorted(corpus_dir.glob("conversations-*.json")):
        with open(f, encoding="utf-8") as fh:
            convos = json.load(fh)
        for c in convos:
            cid = c.get("conversation_id", "?")
            mapping = c.get("mapping") or {}
            for node in mapping.values():
                msg = node.get("message")
                if not msg:
                    continue
                if (msg.get("author") or {}).get("role") != "user":
                    continue
                content = msg.get("content") or {}
                parts = content.get("parts") or []
                text_parts = [p for p in parts if isinstance(p, str) and p.strip()]
                if not text_parts:
                    continue
                yield cid, " ".join(text_parts)


def main() -> int:
    corpus = Path(os.environ.get("CORPUS_DIR", DEFAULT_CORPUS))
    if not corpus.exists():
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 1

    n_total = 0
    n_zero = 0
    n_nonzero = 0
    slot_counter: Counter[str] = Counter()
    slots_per_turn: Counter[int] = Counter()
    convo_ids = set()

    t0 = time.time()
    for cid, text in iter_user_turns(corpus):
        n_total += 1
        convo_ids.add(cid)
        try:
            slots = extract_fact_slots(text) or {}
        except Exception:
            slots = {}
        if not slots:
            n_zero += 1
        else:
            n_nonzero += 1
            for k in slots.keys():
                slot_counter[k] += 1
        slots_per_turn[len(slots)] += 1
        if n_total % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  ... {n_total} turns processed in {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0

    pct_zero = 100.0 * n_zero / n_total if n_total else 0.0
    pct_nonzero = 100.0 * n_nonzero / n_total if n_total else 0.0

    print()
    print("=" * 70)
    print(f"GPT corpus slot-coverage histogram")
    print("=" * 70)
    print(f"  conversations:      {len(convo_ids):>8,}")
    print(f"  user turns total:   {n_total:>8,}")
    print(f"  zero-slot turns:    {n_zero:>8,}  ({pct_zero:5.2f}%)  <-- paraphrase-blind floor")
    print(f"  >=1 slot turns:     {n_nonzero:>8,}  ({pct_nonzero:5.2f}%)")
    print(f"  elapsed:            {elapsed:.1f}s")
    print()
    print("slots-per-turn distribution:")
    for k in sorted(slots_per_turn):
        print(f"  {k} slot(s): {slots_per_turn[k]:>7,}")
    print()
    print("top slot categories among >=1-slot turns:")
    for slot, count in slot_counter.most_common(30):
        print(f"  {slot:30s} {count:>6,}  ({100.0 * count / n_nonzero:5.2f}% of nonzero turns)")

    out = Path(__file__).parent / "slot_coverage_results.json"
    out.write_text(
        json.dumps(
            {
                "corpus_dir": str(corpus),
                "conversations": len(convo_ids),
                "user_turns_total": n_total,
                "zero_slot_turns": n_zero,
                "nonzero_slot_turns": n_nonzero,
                "zero_slot_pct": pct_zero,
                "elapsed_seconds": elapsed,
                "slots_per_turn": dict(slots_per_turn),
                "slot_frequencies": dict(slot_counter.most_common()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
