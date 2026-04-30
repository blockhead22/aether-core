#!/usr/bin/env python3
"""Stop hook: extract high-signal facts from the last turn into Aether.

Standalone copy of the plugin's auto_ingest_hook.py — wired at the
project level so this works whether or not the Claude Code plugin is
installed/enabled.

Reads the Stop-event JSON payload on stdin. Claude Code currently
sends:

    {"session_id": ..., "transcript_path": ..., "hook_event_name": "Stop", ...}

`transcript_path` points at a JSONL file with one entry per turn. We
parse that backward to find the most recent user + assistant messages,
skipping `isMeta` (system caveats) and `isSidechain` (sub-agent) entries,
then hand both to `aether.memory.ingest_turn`.

Older payload shapes (`messages` / `transcript` / `last_user_message`
inline in stdin) are still supported for compatibility with other
clients and the unit tests.

Never fails the hook chain — all errors are logged to stderr and the
script exits 0.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional


def _read_stdin_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


def _flatten_content(content: Any) -> str:
    """Pull a plain string out of a Claude Code content field.

    User content is usually a str. Assistant content (and tool-result
    user content) is a list of blocks like
    `[{"type": "text", "text": "..."}, {"type": "thinking", ...}, ...]`.
    We take the concatenation of `text` blocks; thinking is dropped
    because it's not meant to be persisted.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            # Skip "thinking", "tool_use", "tool_result", etc. — not
            # user-visible content. (Tool results show up as user-role
            # entries in the transcript but they're tool output, not
            # what the user said — handled by `_is_real_user_input`.)
        return "\n".join(p for p in parts if p)
    return ""


def _is_real_user_input(message: dict) -> bool:
    """Tool results are stored as `type: "user"` entries in the
    transcript with `content` = list-of-tool_result blocks. Distinguish
    those from actual user prompts.

    Real user input either has `content` as a string, or as a list that
    contains at least one `text` block. A list of pure tool_result
    blocks is tool output and should be skipped.
    """
    content = message.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return True
    return False


def _iter_transcript_entries(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a Claude Code transcript JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_from_transcript(
    transcript_path: str,
) -> tuple[str, str]:
    """Walk the transcript backward, return (user_msg, assistant_msg).

    "Last" means the most recent non-meta, non-sidechain message of
    each role. The last assistant text is typically the just-completed
    turn's reply; the last user text is the prompt that produced it.
    """
    p = Path(transcript_path)
    if not p.exists():
        return "", ""

    user_msg = ""
    asst_msg = ""

    # Read the whole file then walk backward — transcripts are small
    # enough (a few MB max) that this is fine, and we need backward
    # iteration anyway.
    entries = list(_iter_transcript_entries(p))
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        if entry.get("isMeta") or entry.get("isSidechain"):
            continue
        etype = entry.get("type")
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role") or etype
        if (
            role in ("user", "human")
            and not user_msg
            and etype == "user"
            and _is_real_user_input(message)
        ):
            content = _flatten_content(message.get("content", ""))
            if content.strip():
                user_msg = content
        elif (
            role in ("assistant", "claude", "ai")
            and not asst_msg
            and etype == "assistant"
        ):
            content = _flatten_content(message.get("content", ""))
            if content.strip():
                asst_msg = content
        if user_msg and asst_msg:
            break

    return user_msg, asst_msg


def _last_message_inline(payload: dict, *roles: str) -> str:
    """Legacy fallback: look for messages embedded in the payload itself."""
    for key in ("messages", "transcript", "history"):
        seq = payload.get(key)
        if isinstance(seq, list):
            for entry in reversed(seq):
                if not isinstance(entry, dict):
                    continue
                role = entry.get("role") or entry.get("type")
                if role in roles:
                    content = entry.get("content") or entry.get("text") or ""
                    flat = _flatten_content(content)
                    if flat.strip():
                        return flat
    if "user" in roles and isinstance(payload.get("last_user_message"), str):
        return payload["last_user_message"]
    if "assistant" in roles and isinstance(payload.get("last_assistant_message"), str):
        return payload["last_assistant_message"]
    return ""


def main() -> int:
    payload = _read_stdin_payload()

    user_msg = ""
    asst_msg = ""

    # Preferred path: Claude Code sends `transcript_path`. Read the JSONL.
    transcript_path = payload.get("transcript_path") or payload.get("transcriptPath")
    if isinstance(transcript_path, str) and transcript_path:
        try:
            user_msg, asst_msg = _extract_from_transcript(transcript_path)
        except Exception:
            print("[aether auto-ingest] transcript read error (suppressed):",
                  file=sys.stderr)
            traceback.print_exc()

    # Legacy fallback: messages embedded in the payload (other clients,
    # unit tests). Only use these if transcript-path extraction came up
    # empty.
    if not user_msg:
        user_msg = _last_message_inline(payload, "user", "human")
    if not asst_msg:
        asst_msg = _last_message_inline(payload, "assistant", "claude", "ai")

    if not user_msg and not asst_msg:
        return 0

    try:
        from aether.mcp.state import StateStore
        from aether.memory import ingest_turn
    except ImportError:
        # aether-core not installed — silent skip.
        return 0

    try:
        store = StateStore()
        writes = ingest_turn(
            store,
            user_message=user_msg or None,
            assistant_response=asst_msg or None,
        )
        if writes:
            print(f"[aether auto-ingest] wrote {len(writes)} fact(s)",
                  file=sys.stderr)
            for w in writes:
                print(f"  - ({w.get('signal')}) trust={w['trust']:.2f} "
                      f"-> {w['memory_id']}",
                      file=sys.stderr)
    except Exception:
        print("[aether auto-ingest] error (suppressed):", file=sys.stderr)
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
