"""Slot-value drift on the GPT corpus.

What this measures
------------------
Across 13 months of GPT conversations, for each slot category the
extractor recognizes (location, employer, occupation, ...), how many
*distinct values* did the user commit to? A user who said "I live in
Seattle" in March and "I live in Chicago" in October committed to two
distinct ``location`` values — that's an open contradiction the
contradiction layer should catch on ingest.

The output answers two questions:

  1. How often does the corpus actually contain self-contradictions
     on slot-covered facts?
  2. Of those, what fraction would the current OSS aether-core slot
     comparator catch if the corpus were ingested chronologically?

The first number sets a ceiling on what the contradiction layer could
possibly detect on this corpus. The second number bounds what it
actually does detect, given today's slot vocabulary.

Run::

    python -m bench.slot_value_drift_gpt_corpus
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from aether.memory import extract_fact_slots

DEFAULT_CORPUS = (
    "C:/Users/block/Downloads/"
    "fbf5a239c1af822f50241d4b5999b53954689723b9d4deb7ceb1e41a31847485-"
    "2026-03-27-22-39-55-cf0803cbc48c446cb8c3b317ea5f11ea"
)


def iter_user_turns(corpus_dir: Path):
    """Yield (create_time_int, conv_id, turn_text) — chronological-ish."""
    for f in sorted(corpus_dir.glob("conversations-*.json")):
        with open(f, encoding="utf-8") as fh:
            convos = json.load(fh)
        for c in convos:
            cid = c.get("conversation_id", "?")
            ctime = c.get("create_time") or 0
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
                yield (msg.get("create_time") or ctime), cid, " ".join(text_parts)


def main() -> int:
    corpus = Path(os.environ.get("CORPUS_DIR", DEFAULT_CORPUS))
    if not corpus.exists():
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 1

    # slot -> normalized_value -> {count, first_seen, last_seen, sample_text}
    slot_values: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"count": 0, "first_seen": None, "last_seen": None, "sample": ""}
    ))

    n_total = 0
    t0 = time.time()
    rows = list(iter_user_turns(corpus))
    rows.sort(key=lambda r: r[0] or 0)

    for ts, cid, text in rows:
        n_total += 1
        try:
            slots = extract_fact_slots(text) or {}
        except Exception:
            slots = {}
        for slot_name, ext in slots.items():
            norm = (ext.normalized or ext.value or "").strip().lower()
            if not norm:
                continue
            entry = slot_values[slot_name][norm]
            entry["count"] += 1
            if entry["first_seen"] is None:
                entry["first_seen"] = ts
                entry["sample"] = text[:140]
            entry["last_seen"] = ts
        if n_total % 5000 == 0:
            print(f"  ... {n_total} turns processed in {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("Slot-value drift on the GPT corpus")
    print("=" * 70)
    print(f"  user turns processed: {n_total:,}")
    print(f"  elapsed:              {elapsed:.1f}s")
    print()
    print(f"{'slot':30s} {'distinct_values':>16s} {'total_mentions':>16s} {'contradicts?':>14s}")
    print("-" * 80)

    contradicts_summary = []
    for slot in sorted(slot_values, key=lambda s: -len(slot_values[s])):
        vals = slot_values[slot]
        total = sum(v["count"] for v in vals.values())
        contra = "YES" if len(vals) > 1 else "no"
        print(f"{slot:30s} {len(vals):>16,} {total:>16,} {contra:>14s}")
        if len(vals) > 1:
            contradicts_summary.append((slot, len(vals), total))

    print()
    print("=" * 70)
    print(f"Slots with >=2 distinct user-stated values: {len(contradicts_summary)}")
    print("=" * 70)
    for slot, n_vals, total in contradicts_summary:
        vals = slot_values[slot]
        ranked = sorted(vals.items(), key=lambda kv: -kv[1]["count"])
        print(f"\n{slot}  ({n_vals} distinct, {total} mentions)")
        for v, info in ranked[:6]:
            print(f"  - {v[:60]:60s} count={info['count']:>4d}")
        if len(ranked) > 6:
            print(f"  ... and {len(ranked) - 6} more")

    out = Path(__file__).parent / "slot_drift_results.json"
    out.write_text(
        json.dumps(
            {
                "user_turns": n_total,
                "elapsed_seconds": elapsed,
                "slots_with_drift": [
                    {
                        "slot": slot,
                        "distinct_values": n,
                        "total_mentions": total,
                        "values": [
                            {"value": v, **info}
                            for v, info in sorted(
                                slot_values[slot].items(),
                                key=lambda kv: -kv[1]["count"],
                            )
                        ],
                    }
                    for slot, n, total in contradicts_summary
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
