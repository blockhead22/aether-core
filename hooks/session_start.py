#!/usr/bin/env python3
"""SessionStart hook: ensure aether is installed, warm, and ready.

Goal: a Claude Code user with `claude plugin install aether-core`
should not need a single manual step beyond that. This script runs
on every session start and:

  1. Ensures aether-core is importable (pip-installs if missing).
  2. Kicks off encoder warmup in the background (non-blocking — the
     first MCP query may run cold; subsequent queries warm).
  3. Ensures the user-global substrate exists with default policy
     beliefs (force-push, --no-verify, prod safety) — F#11 fix.
  4. On the first run after install (no marker file), emits an
     additionalContext welcome message so the user knows aether
     is active and how to interact with it.
  5. On subsequent runs, emits a brief status line (memory count,
     contradictions, encoder mode).

Output protocol (Claude Code SessionStart):
    JSON to stdout with shape
        {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                "additionalContext": "..."}}

Exit code:
    0 always — even on failure. Aether failures must never block a
    Claude session. Errors go to a log file at
    ~/.aether/session_start.log so install issues are debuggable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _log_path() -> Path:
    return Path.home() / ".aether" / "session_start.log"


def _first_run_marker() -> Path:
    return Path.home() / ".aether" / ".first_run_complete"


def _log(line: str) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
    except Exception:
        pass


def _emit(additional_context: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    }
    print(json.dumps(payload))


# Version floor: features the hook depends on. If the installed aether
# is older than this, the hook upgrades. Bump when the hook starts
# using a new symbol that older releases don't expose.
_REQUIRED_VERSION = "0.12.13"


def _version_tuple(v: str) -> tuple:
    """Parse 'X.Y.Z' (or with trailing alpha/beta) to a tuple for comparison.
    Forgiving — non-numeric components compare as 0 so we don't crash
    on dev-build version strings.
    """
    parts = []
    for chunk in v.split("."):
        n = ""
        for c in chunk:
            if c.isdigit():
                n += c
            else:
                break
        parts.append(int(n) if n else 0)
    return tuple(parts)


def _pip_install(upgrade: bool) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append("aether-core[mcp,graph,ml]")
    try:
        subprocess.run(cmd, check=True, timeout=180)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _log(f"INSTALL  pip failed: {type(e).__name__}: {e}")
        return False


def _ensure_installed() -> bool:
    """Pip-install or upgrade aether-core[mcp,graph,ml].

    Three cases:
      1. Not importable — fresh install via pip.
      2. Importable but version < _REQUIRED_VERSION — pip --upgrade.
      3. Already up to date — no-op.

    Returns True if aether is importable + at floor after this call.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("aether") is None:
            _log("INSTALL  aether not importable, running pip install")
            if not _pip_install(upgrade=False):
                return False
            return True
    except Exception:
        return False

    # Already importable — check the version.
    try:
        # Fresh import in case a previous import cached a stale module.
        import importlib
        import aether
        importlib.reload(aether)
        installed = getattr(aether, "__version__", "0.0.0")
    except Exception as e:
        _log(f"INSTALL  could not read aether.__version__: {e}")
        return True  # don't fail-stop on a cosmetic issue

    if _version_tuple(installed) < _version_tuple(_REQUIRED_VERSION):
        _log(
            f"INSTALL  aether {installed} < required {_REQUIRED_VERSION}, "
            f"running pip --upgrade"
        )
        if not _pip_install(upgrade=True):
            return False
        # After upgrade, force a re-import so the hook uses the new code.
        try:
            import importlib
            import aether
            importlib.reload(aether)
        except Exception:
            pass
    return True


def _start_warmup_async() -> None:
    """Kick off encoder warmup in the background. Non-blocking.

    The encoder takes 15-30s to load; blocking the session start
    on it produces a terrible first-impression UX. Instead we
    fire the background thread and return immediately. First
    MCP query may run cold; second is almost certainly warm.
    """
    try:
        from aether._lazy_encoder import LazyEncoder
        enc = LazyEncoder()
        enc.start_warmup()
        _log("WARMUP   started (background)")
    except Exception as e:
        _log(f"WARMUP   skipped: {type(e).__name__}: {e}")


def _ensure_user_substrate() -> dict:
    """Ensure the user-global ~/.aether/mcp_state.json exists with
    default policy beliefs seeded.

    Returns a dict with keys:
        existed     bool   — substrate file was present before this call
        seeded      int    — number of new default beliefs written
        memory_count int   — total memories after the call
        encoder_mode str   — 'warm' / 'warming' / 'cold'
    """
    info = {"existed": False, "seeded": 0, "memory_count": 0, "encoder_mode": "cold"}
    try:
        from aether.mcp.state import StateStore, _default_state_path
        from aether.cli import _seed_default_beliefs
    except Exception as e:
        _log(f"STATE    import failed: {type(e).__name__}: {e}")
        return info

    state_path = Path(_default_state_path())
    info["existed"] = state_path.exists()

    # Initialize the substrate file if missing (parallel of cmd_init for
    # the user-global path).
    if not state_path.exists():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2))
        info["seeded"] = _seed_default_beliefs(state_path)
        _log(f"STATE    initialized + seeded {info['seeded']} default beliefs")

    try:
        store = StateStore(state_path=str(state_path))
        stats = store.stats()
        info["memory_count"] = stats.get("memory_count", 0)
        if stats.get("embeddings_loaded"):
            info["encoder_mode"] = "warm"
        elif stats.get("embeddings_warming"):
            info["encoder_mode"] = "warming"
        else:
            info["encoder_mode"] = "cold"
    except Exception as e:
        _log(f"STATE    stats failed: {type(e).__name__}: {e}")

    return info


def _is_first_run() -> bool:
    return not _first_run_marker().exists()


def _mark_first_run_complete() -> None:
    try:
        marker = _first_run_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    except Exception:
        pass


def _welcome_message(info: dict) -> str:
    return (
        "<aether_substrate>\n"
        "Aether-core belief substrate is now active in this Claude session. "
        "It maintains a persistent belief state across sessions: facts "
        "you state, contradictions surfaced, trust scores that move under "
        "correction. The model is the mouth; the substrate is the self.\n"
        "\n"
        f"On install, {info['seeded']} default policy beliefs were seeded "
        "(force-push, --no-verify, production data safety, rm -rf). "
        "`aether_sanction` will gate proposed actions against them automatically. "
        "Add your own with `aether_remember` or just by stating facts in "
        "conversation — the auto-ingest hook captures high-signal facts every turn.\n"
        "\n"
        "Slash commands available: /aether-status, /aether-search, "
        "/aether-contradictions, /aether-check, /aether-correct, /aether-ingest.\n"
        "\n"
        "Run `aether doctor` in a terminal if anything feels off, or "
        "`aether doctor --report` for a one-paste GitHub-issue bundle.\n"
        "</aether_substrate>"
    )


def _status_message(info: dict) -> str:
    return (
        "<aether_substrate>\n"
        f"Aether: {info['memory_count']} memories, encoder {info['encoder_mode']}.\n"
        "</aether_substrate>"
    )


def main() -> int:
    # SessionStart payload on stdin — read it but don't require any field.
    try:
        sys.stdin.read()
    except Exception:
        pass

    # 1. Ensure aether is installed.
    if not _ensure_installed():
        _log("FATAL    aether could not be installed; emitting empty context")
        _emit("")
        return 0

    # 2. Kick off encoder warmup (non-blocking).
    _start_warmup_async()

    # 3. Ensure the substrate exists + seeded.
    info = _ensure_user_substrate()

    # 4. First-run vs returning user.
    first_run = _is_first_run()
    _log(
        f"FIRE     first_run={first_run} existed={info['existed']} "
        f"seeded={info['seeded']} memories={info['memory_count']}"
    )
    if first_run:
        _emit(_welcome_message(info))
        _mark_first_run_complete()
    else:
        _emit(_status_message(info))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        _log("CRASH    " + traceback.format_exc().replace("\n", " | "))
        # Never block Claude even on crash.
        sys.exit(0)
