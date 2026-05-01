"""F#8 regression test: encoder warmup must complete in a subprocess.

Background — F#8 surfaced 2026-04-30 evening:
    The MCP server's `_LazyEncoder.start_warmup()` reliably set
    `_warmup_started=True`, but `_model` stayed None forever and
    `_unavailable` never flipped to True. State was stuck at
    `is_warming=True` indefinitely across multiple server restarts.

    Standalone Python in the same env loaded
    `sentence-transformers/all-MiniLM-L6-v2` cleanly in 0.3s — so the
    bug was specific to the MCP-subprocess context, not the model.

    Hypothesis (confirmed by fix): SentenceTransformer + transformers
    + tqdm write progress / warning messages to stdout & stderr by
    default. When the parent process expects MCP wire-protocol on
    stdout, those writes corrupt the stream and can stall the warmup
    thread on a backed-up pipe write. The thread didn't crash visibly
    so the existing `except Exception` never set `_unavailable`.

The fix (v0.12.6, in aether/_lazy_encoder.py):
    1. HuggingFace env vars at module load (TRANSFORMERS_VERBOSITY,
       HF_HUB_DISABLE_PROGRESS_BARS, TOKENIZERS_PARALLELISM).
    2. contextlib.redirect_stdout/stderr around the
       SentenceTransformer construction.
    3. BaseException (not just Exception) on the load except.
    4. Diagnostic log to ~/.aether/encoder_warmup.log so future hangs
       are debuggable in seconds.

This test reproduces the F#8 condition: spawn a subprocess with stdin
piped (mimicking MCP's stdio attachment), force-import the encoder
module, kick warmup, and verify it completes within a generous
timeout. Skips cleanly if [ml] isn't installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import sentence_transformers  # noqa: F401
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

needs_ml = pytest.mark.skipif(
    not _HAS_ML,
    reason="sentence-transformers required (install [ml] extra)",
)


@needs_ml
def test_encoder_warmup_completes_in_subprocess_with_pipes():
    """Spawn a child Python with stdin/stdout/stderr piped (the MCP-style
    context that triggered F#8) and confirm the encoder reaches
    `is_loaded=True` within 60s.

    Pre-fix this test would hang and timeout. Post-fix it should
    complete in 1-10s on a warm cache, 30-60s on a cold one.
    """
    code = (
        "import sys\n"
        "from aether._lazy_encoder import LazyEncoder\n"
        "enc = LazyEncoder()\n"
        # wait_until_ready triggers start_warmup if needed and joins the thread
        "ok = enc.wait_until_ready(timeout=60)\n"
        "sys.stdout.write('LOADED' if enc.is_loaded else 'NOT_LOADED')\n"
        "sys.stdout.flush()\n"
        "sys.exit(0 if enc.is_loaded else 1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"encoder warmup failed in subprocess. "
        f"stdout={result.stdout!r} stderr={result.stderr[-500:]!r}"
    )
    assert b"LOADED" in result.stdout, (
        f"encoder did not reach is_loaded=True. "
        f"stdout={result.stdout!r} stderr={result.stderr[-500:]!r}"
    )


@needs_ml
def test_warmup_log_file_records_load():
    """Confirm the diagnostic log gets written. Uses a tmp HOME so we
    don't pollute the real ~/.aether/encoder_warmup.log.
    """
    import tempfile
    home = Path(tempfile.mkdtemp())
    code = (
        "import sys\n"
        "from aether._lazy_encoder import LazyEncoder\n"
        "enc = LazyEncoder()\n"
        "enc.wait_until_ready(timeout=60)\n"
        "sys.exit(0 if enc.is_loaded else 1)\n"
    )
    env = {**os.environ, "USERPROFILE": str(home), "HOME": str(home)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    log_path = home / ".aether" / "encoder_warmup.log"
    assert log_path.exists(), (
        f"encoder_warmup.log not created. subprocess rc={result.returncode} "
        f"stderr={result.stderr[-500:]!r}"
    )
    contents = log_path.read_text(encoding="utf-8")
    assert "load_attempt" in contents, f"missing load_attempt line: {contents!r}"
    # Either load_ok or load_cache_hit is acceptable depending on cache state.
    assert (
        "load_ok" in contents or "load_cache_hit" in contents
    ), f"missing successful load line: {contents!r}"
