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
from typing import List, Optional


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


_LAST_PIP_STDERR = ""  # populated by _pip_install on failure


def _pip_install(python_path: str, upgrade: bool) -> bool:
    """Run pip install of aether-core[mcp,graph,ml] under the given Python.

    Captures stderr so the caller can detect PEP 668 (Homebrew /
    Debian externally-managed Python) and fall back to a venv. Logs
    the LAST 800 chars of stderr on failure — previously the hook
    swallowed the actual pip error message and only logged the
    Python exception type, which made remote diagnosis impossible.
    """
    global _LAST_PIP_STDERR
    cmd = [python_path, "-m", "pip", "install", "--quiet"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append("aether-core[mcp,graph,ml]")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
        )
        _LAST_PIP_STDERR = result.stderr or ""
        if result.returncode == 0:
            return True
        _log(
            f"INSTALL  pip rc={result.returncode} stderr_tail={_LAST_PIP_STDERR[-800:]}"
        )
        return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _LAST_PIP_STDERR = str(e)
        _log(f"INSTALL  pip exception: {type(e).__name__}: {e}")
        return False


def _is_pep668_error() -> bool:
    """Heuristic: pip stderr indicates an externally-managed Python."""
    return "externally-managed-environment" in _LAST_PIP_STDERR.lower()


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _create_venv_and_install(venv_dir: Path) -> Optional[str]:
    """PEP 668 fallback: build a venv at venv_dir, install aether into it.

    Returns the venv's python path on success, None on failure.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _log(f"VENV     creation at {venv_dir} failed: {type(e).__name__}: {e}")
        return None
    venv_py = _venv_python(venv_dir)
    if not venv_py.exists():
        _log(f"VENV     created but {venv_py} not found")
        return None
    if not _pip_install(str(venv_py), upgrade=False):
        _log(f"VENV     pip install into {venv_dir} failed")
        return None
    _log(f"VENV     ready at {venv_dir}")
    return str(venv_py)


def _maybe_reexec_via_aether_python() -> None:
    """If aether is importable in a venv we recognize but NOT in the current
    Python, re-exec self via that venv's Python so the rest of the script
    sees aether. Returns silently when no re-exec is needed; never returns
    when re-exec succeeds.

    Discovery mirrors hooks/aether_launcher.py.
    """
    # If aether is already importable here, no re-exec needed.
    try:
        import importlib.util
        if importlib.util.find_spec("aether") is not None:
            return
    except Exception:
        pass

    # Try discovery in the same order as the launcher.
    candidates: List[Path] = []
    explicit = os.environ.get("AETHER_PYTHON", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    home = Path.home()
    candidates.append(_venv_python(home / ".aether-venv"))
    candidates.append(_venv_python(home / "aether-venv"))
    active = os.environ.get("VIRTUAL_ENV", "").strip()
    if active:
        candidates.append(_venv_python(Path(active)))

    for py in candidates:
        if not py.exists():
            continue
        try:
            r = subprocess.run(
                [str(py), "-c", "import aether"],
                capture_output=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            _log(f"REEXEC   self via {py}")
            try:
                os.execvp(str(py), [str(py), __file__])
            except OSError as e:
                _log(f"REEXEC   execvp failed: {e}; continuing in current Python")
                return
    # No working alternative — caller continues in current Python.


def _ensure_installed() -> bool:
    """Ensure aether-core is importable.

    Order of operations:
      1. If already importable + at version floor — done.
      2. If importable but old — pip --upgrade in current Python.
      3. If not importable — pip install in current Python.
      4. If pip install hits PEP 668 — create ~/.aether-venv and
         re-exec self via that Python (this call doesn't return).
      5. Any other failure — log loudly + return False so the caller
         can emit empty context (never block Claude).
    """
    try:
        import importlib.util
        already = importlib.util.find_spec("aether") is not None
    except Exception:
        already = False

    if not already:
        _log("INSTALL  aether not importable, running pip install")
        if _pip_install(sys.executable, upgrade=False):
            return True
        if _is_pep668_error():
            venv = Path.home() / ".aether-venv"
            _log(f"INSTALL  PEP 668 detected, falling back to venv at {venv}")
            venv_py = _create_venv_and_install(venv)
            if venv_py:
                _log(f"INSTALL  re-execing self via {venv_py}")
                try:
                    os.execvp(venv_py, [venv_py, __file__])
                except OSError as e:
                    _log(f"INSTALL  execvp into venv failed: {e}")
                    return False
            return False
        return False

    # Already importable — check the version.
    try:
        import importlib
        import aether
        importlib.reload(aether)
        installed = getattr(aether, "__version__", "0.0.0")
    except Exception as e:
        _log(f"INSTALL  could not read aether.__version__: {e}")
        return True  # cosmetic, don't fail

    if _version_tuple(installed) < _version_tuple(_REQUIRED_VERSION):
        _log(
            f"INSTALL  aether {installed} < required {_REQUIRED_VERSION}, "
            f"running pip --upgrade"
        )
        if not _pip_install(sys.executable, upgrade=True):
            # PEP 668 on upgrade is rare (user already had aether installed
            # somehow) but possible. Log and proceed with the older version.
            _log("INSTALL  upgrade failed; continuing with installed version")
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

    # 0. If we're running in a Python without aether but a known venv
    #    (~/.aether-venv, $VIRTUAL_ENV, $AETHER_PYTHON) does have it,
    #    re-exec there. Avoids re-running pip-install on every session
    #    start when the user already has a working venv from a prior
    #    bootstrap. Never returns when re-exec succeeds.
    _maybe_reexec_via_aether_python()

    # 1. Ensure aether is installed. Falls back to creating a venv if
    #    pip hits PEP 668 (Homebrew / managed Python).
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
