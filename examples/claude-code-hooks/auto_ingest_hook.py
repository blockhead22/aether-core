#!/usr/bin/env python3
"""Claude Code Stop hook: extract facts from the last turn into Aether.

Wire into `.claude/settings.json`:

    {
      "hooks": {
        "Stop": [
          {
            "matcher": "*",
            "hooks": [
              {
                "type": "command",
                "command": "python /path/to/auto_ingest_hook.py"
              }
            ]
          }
        ]
      }
    }

Claude Code passes the hook a JSON blob on stdin describing the last
turn. We read it, pull out user_message and the assistant text, then
hand both to `aether.memory.ingest_turn` against the local substrate
file at AETHER_STATE_PATH (or the default ~/.aether/mcp_state.json).

The extractor is conservative — explicit preferences, identity
statements, project facts, decisions, constraints, corrections only.
You're unlikely to get garbage, but you'll also miss many turns. Tune
the rules in aether/memory/auto_ingest.py.
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def _read_stdin_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


def _last_user_message(payload: dict) -> str:
    """Pull the most recent user message from the payload.

    The exact shape depends on the Claude Code version. We're defensive:
    look in `messages`, `transcript`, or `last_user_message` keys.
    """
    if isinstance(payload.get("last_user_message"), str):
        return payload["last_user_message"]

    for key in ("messages", "transcript", "history"):
        seq = payload.get(key)
        if isinstance(seq, list):
            for entry in reversed(seq):
                if not isinstance(entry, dict):
                    continue
                role = entry.get("role") or entry.get("type")
                if role in ("user", "human"):
                    content = entry.get("content") or entry.get("text") or ""
                    if isinstance(content, list):
                        # Some clients send segmented content
                        content = " ".join(
                            seg.get("text", "") if isinstance(seg, dict) else str(seg)
                            for seg in content
                        )
                    if content:
                        return str(content)
    return ""


def _last_assistant_message(payload: dict) -> str:
    for key in ("messages", "transcript", "history"):
        seq = payload.get(key)
        if isinstance(seq, list):
            for entry in reversed(seq):
                if not isinstance(entry, dict):
                    continue
                role = entry.get("role") or entry.get("type")
                if role in ("assistant", "claude", "ai"):
                    content = entry.get("content") or entry.get("text") or ""
                    if isinstance(content, list):
                        content = " ".join(
                            seg.get("text", "") if isinstance(seg, dict) else str(seg)
                            for seg in content
                        )
                    if content:
                        return str(content)
    return ""


def main() -> int:
    payload = _read_stdin_payload()
    user_msg = _last_user_message(payload)
    asst_msg = _last_assistant_message(payload)

    if not user_msg and not asst_msg:
        # Nothing to ingest. Silent success — don't block the hook chain.
        return 0

    try:
        # Lazy import so a missing aether-core install doesn't kill Claude.
        from aether.mcp.state import StateStore
        from aether.memory import ingest_turn
    except ImportError:
        print("aether-core not installed; skipping auto-ingest", file=sys.stderr)
        return 0

    try:
        store = StateStore()
        writes = ingest_turn(
            store,
            user_message=user_msg or None,
            assistant_response=asst_msg or None,
        )
        if writes:
            print(
                f"[aether auto-ingest] wrote {len(writes)} fact(s) "
                f"from the last turn",
                file=sys.stderr,
            )
            for w in writes:
                print(
                    f"  - ({w.get('signal')}) trust={w['trust']:.2f} "
                    f"-> {w['memory_id']}",
                    file=sys.stderr,
                )
    except Exception:
        # Never fail the hook because of an extractor error.
        print("[aether auto-ingest] error (suppressed):", file=sys.stderr)
        traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
