"""v0.13.2 regression: CLI commands must produce stdout immediately.

The Mac handoff (`HANDOFF_2026-05-01_to_windows_late.md`) caught a
universal-Python bug: `_LazyEncoder._load` was using
`contextlib.redirect_stdout` on a background thread. Since
`redirect_stdout` swaps `sys.stdout` process-globally (not
thread-locally), the main thread's `print()` calls during the ~3–8s
warmup window were silently discarded.

This made `aether status`, `aether check`, and `aether contradictions`
exit 0 with zero stdout. `aether doctor` worked because it called
`wait_until_ready()` synchronously and printed only after the redirect
released. The fix replaced the redirect with at-source HF env-var
suppression (`TRANSFORMERS_VERBOSITY`, `HF_HUB_DISABLE_PROGRESS_BARS`,
`TQDM_DISABLE`, `TOKENIZERS_PARALLELISM`) which is what the redirect
was trying to catch in the first place.

These tests are subprocess-based: run the actual CLI binary, measure
elapsed time, assert non-empty stdout. A unit test against `cmd_status`
would miss the bug entirely because the redirect leak only happens when
the warmup thread is racing the main thread's prints — and pytest's
own captured-output behavior masks the symptom.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest


def _venv_aether_or_skip():
    """Find the installed `aether` console script that pairs with THIS
    interpreter. Critical: do NOT use shutil.which() first — on systems
    with both a system Python and a venv, PATH usually puts the system
    aether.exe first, which is the OLD version we're trying to test
    against. The right binary is sibling to the running Python."""
    candidate = Path(sys.executable).parent / (
        "aether.exe" if sys.platform == "win32" else "aether"
    )
    if candidate.exists():
        return str(candidate)
    # Fallback: console_scripts on Linux can land in `bin/`, on Windows
    # they're always in Scripts/. If sibling lookup fails, try PATH last
    # — but expect it to find a different version.
    import shutil
    found = shutil.which("aether")
    if found:
        return found
    pytest.skip("aether console script not next to interpreter; install via `pip install -e .`")


def _run_cli(*args, timeout=30):
    """Invoke the installed aether CLI and capture stdout/stderr."""
    aether = _venv_aether_or_skip()
    start = time.monotonic()
    result = subprocess.run(
        [aether, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start
    return result, elapsed


class TestCLIProducesOutput:
    """The contract: every CLI command that takes a `_make_store()` call
    must produce stdout within a reasonable wall-clock window. The bug
    was: warmup thread eats stdout for 3–8s after StateStore() runs.
    Asserting `stdout != ""` within 5s catches the regression on every
    machine that has the encoder cached (which is every machine after
    the first warmup completes)."""

    def test_aether_status_produces_output(self):
        result, elapsed = _run_cli("status")
        assert result.returncode == 0, (
            f"aether status exited {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        assert result.stdout.strip(), (
            f"REGRESSION: aether status returned empty stdout in {elapsed:.2f}s. "
            f"This is the v0.12.6 redirect_stdout leak from "
            f"_LazyEncoder._load described in "
            f"HANDOFF_2026-05-01_to_windows_late.md.\n"
            f"stderr: {result.stderr!r}"
        )

    def test_aether_check_produces_output(self):
        result, elapsed = _run_cli("check", "--message", "test claim")
        # `aether check` may exit non-zero by design (CRITICAL on the
        # claim) — what we care about is that it produced SOME output.
        assert result.stdout.strip() or result.stderr.strip(), (
            f"REGRESSION: aether check produced no output in {elapsed:.2f}s "
            f"(exit={result.returncode})"
        )

    def test_aether_contradictions_produces_output(self):
        result, elapsed = _run_cli("contradictions")
        assert result.returncode == 0, (
            f"aether contradictions exited {result.returncode}\n"
            f"stderr: {result.stderr!r}"
        )
        # Either "no contradictions in substrate" or a list. Both are output.
        assert result.stdout.strip(), (
            f"REGRESSION: aether contradictions returned empty stdout in "
            f"{elapsed:.2f}s. stderr: {result.stderr!r}"
        )


class TestCLITimingSanity:
    """The fix shouldn't make CLI commands slower. They should still
    return promptly (within a few seconds — encoder warmup runs in
    background, doesn't block the print)."""

    def test_aether_status_returns_promptly(self):
        result, elapsed = _run_cli("status")
        assert elapsed < 10.0, (
            f"aether status took {elapsed:.2f}s — should return quickly "
            f"because encoder warmup is non-blocking. Either the encoder "
            f"is loading synchronously (regression) or the test machine "
            f"is unusually slow."
        )
