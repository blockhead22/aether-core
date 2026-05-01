"""Process-wide non-blocking sentence-transformer encoder.

Loading SentenceTransformer is slow (torch import, possibly model download,
device probing). Doing it synchronously inside a tool call blocks the entire
turn — on Windows with a cold cache, that can be 30s to several minutes.

This module provides a single non-blocking encoder pattern shared by every
component that needs embeddings: the MCP state store, the governance immune
agents (template detector, speech-leak detector, continuity auditor), and any
future caller. Sharing matters because each independent SentenceTransformer
load takes the same multi-second hit; serializing them all to one shared
instance is the difference between "warmup is annoying once" and "every
governance tool wedges separately."

Behavior summary:

    encoder = LazyEncoder()
    encoder.start_warmup()      # returns instantly, model loads behind
    encoder.encode("hello")     # may return None if not warm yet
    # ... 30 seconds later ...
    encoder.encode("hello")     # now returns a numpy vector

If the [ml] extra isn't installed, `start_warmup()` flags the encoder
unavailable nearly instantly. Same fallback as before.

Process-wide cache (`_MODEL_CACHE`) means multiple `LazyEncoder` instances
for the same model share one load. The lock serializes first-load attempts
to avoid duplicate torch imports.

F#8 fix (v0.12.6): in MCP-subprocess contexts, the warmup thread used to
silently hang. SentenceTransformer.__init__ + transformers writes progress
messages and warnings to stdout/stderr by default. When the host process is
Claude Code (which expects MCP wire-protocol messages on stdout), those
writes corrupt the protocol and can stall the load thread on a backed-up
pipe. Three defenses applied here:

    1. Quiet HuggingFace at module-load via env vars (no tqdm progress, no
       transformers warnings, no tokenizer parallelism noise).
    2. Redirect stdout + stderr inside the load itself, so any remaining
       output from torch / sentence-transformers / huggingface_hub is
       captured to a buffer and discarded.
    3. Broaden the warmup except clause to `BaseException` so any failure
       (including KeyboardInterrupt-shaped surprises) sets `_unavailable`
       rather than leaving the encoder in a permanent `is_warming` state.

Plus: every warmup attempt writes one line to `~/.aether/encoder_warmup.log`
so a future hang is debuggable in seconds without re-instrumenting.
"""

from __future__ import annotations

import os as _os
from typing import Any, Dict, List, Optional


# F#8 defense layer 1: silence HuggingFace before any HF import happens.
# `setdefault` so user env wins if they want progress output for debugging.
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# v0.13.2: stop tqdm globally. Without this, tqdm progress bars from
# torch / sentence-transformers can still emit even with the HF
# variables set, and they hit stdout from the warmup thread.
_os.environ.setdefault("TQDM_DISABLE", "1")

# F#8 layer 1.5 (v0.12.7): force offline mode if model is cached. By
# default HF Hub does an online "etag check" against huggingface.co
# even when the model is fully cached; if the network is slow/blocked
# this stalls SentenceTransformer.__init__ for 30s+ per load attempt.
# We set HF_HUB_OFFLINE=1 only when we can prove the model is locally
# cached (so a true cold-start install still works).
def _model_is_cached_locally(model_name: str) -> bool:
    """True if the HF cache has at least one snapshot for the model."""
    from pathlib import Path
    cache_root = Path(_os.environ.get("HF_HOME") or
                      Path.home() / ".cache" / "huggingface")
    hub = cache_root / "hub"
    # HF cache layout: hub/models--<org>--<name>/snapshots/<hash>/
    safe = "models--" + model_name.replace("/", "--")
    snapshots = hub / safe / "snapshots"
    try:
        return snapshots.exists() and any(snapshots.iterdir())
    except OSError:
        return False


if _model_is_cached_locally("sentence-transformers/all-MiniLM-L6-v2"):
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _log_warmup(line: str) -> None:
    """Append a single diagnostic line to ~/.aether/encoder_warmup.log.

    Best-effort; never raises. Used so future warmup hangs are debuggable
    by `cat ~/.aether/encoder_warmup.log` rather than re-instrumenting.
    """
    try:
        from pathlib import Path
        import time
        log = Path.home() / ".aether" / "encoder_warmup.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
    except Exception:
        pass


# Process-wide model cache. Shared across all LazyEncoder instances so
# the MCP state store, governance agents, and any other caller pay the
# cold-load cost exactly once. The lock serializes load attempts.
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_CACHE_LOCK = None  # initialized lazily — see _get_cache_lock


def _get_cache_lock():
    """Lazy-init a threading.Lock without forcing import at module load."""
    global _MODEL_CACHE_LOCK
    if _MODEL_CACHE_LOCK is None:
        import threading
        _MODEL_CACHE_LOCK = threading.Lock()
    return _MODEL_CACHE_LOCK


def _normalize_model_name(name: str) -> str:
    """SentenceTransformer accepts both 'all-MiniLM-L6-v2' and the
    'sentence-transformers/' prefixed form. They produce the same model.
    Normalize to the prefixed form so cache lookups don't double-load.
    """
    if "/" in name:
        return name
    return f"sentence-transformers/{name}"


class LazyEncoder:
    """Non-blocking sentence-transformer wrapper with process-wide cache.

    Public contract: tool handlers must NEVER block on a load. Use
    `encode()` / `encode_batch()` and accept None as "not ready, fall back."

    For tests / scripts that genuinely need the model, call
    `wait_until_ready(timeout)` or `_load()` (the latter is synchronous).
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = _normalize_model_name(model_name)
        self._model = None
        self._unavailable = False
        self._warmup_thread = None
        self._warmup_started = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self):
        """Synchronous load. Use start_warmup() for non-blocking init.

        Uses the process-wide cache so multiple LazyEncoder instances for
        the same model share one load. The lock serializes first-load
        attempts to avoid duplicate torch imports.

        F#8 (v0.12.6): wraps the SentenceTransformer construction in a
        stdout/stderr redirect so any HuggingFace / torch / tqdm output
        is captured to buffers rather than dumped to the parent process's
        pipe (which would corrupt the MCP wire protocol). The except is
        widened to BaseException so any failure flips `_unavailable`
        instead of leaving the warmup thread in a half-state forever.
        """
        if self._model is not None or self._unavailable:
            return
        cached = _MODEL_CACHE.get(self.model_name)
        if cached is not None:
            self._model = cached
            return
        lock = _get_cache_lock()
        _log_warmup(f"load_attempt model={self.model_name}")
        with lock:
            cached = _MODEL_CACHE.get(self.model_name)
            if cached is not None:
                self._model = cached
                _log_warmup("load_cache_hit")
                return
            try:
                # v0.13.2: do NOT use contextlib.redirect_stdout here.
                # That swap mutates sys.stdout process-globally — it's
                # not thread-local — so during the seconds the warmup
                # thread holds the redirect, the main thread's print()
                # calls get silently eaten. That's exactly what made
                # `aether status` / `check` / `contradictions` produce
                # zero stdout for ~3-8s after StateStore() construction.
                #
                # The HF env-var suppression at module load (above)
                # handles the noise the redirect was meant to catch:
                # TRANSFORMERS_VERBOSITY=error silences transformers,
                # HF_HUB_DISABLE_PROGRESS_BARS + TQDM_DISABLE kill the
                # download progress bars, TOKENIZERS_PARALLELISM=false
                # suppresses the parallelism warning.
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = model
                self._model = model
                _log_warmup("load_ok")
            except BaseException as e:
                # F#8 defense layer 3: BaseException catches everything.
                # Without this, an interpreter-level failure during
                # SentenceTransformer init left the encoder in a permanent
                # `is_warming=True` state with no error path.
                self._unavailable = True
                _log_warmup(
                    f"load_failed type={type(e).__name__} "
                    f"msg={str(e)[:300]}"
                )

    def start_warmup(self) -> None:
        """Kick off background model load. Idempotent; returns instantly."""
        if self._warmup_started or self._model is not None or self._unavailable:
            return
        self._warmup_started = True
        try:
            import threading
            t = threading.Thread(
                target=self._load,
                name="aether-encoder-warmup",
                daemon=True,
            )
            t.start()
            self._warmup_thread = t
            _log_warmup("warmup_thread_started")
        except BaseException as e:
            # Threading failure: degrade gracefully without blocking.
            self._unavailable = True
            _log_warmup(
                f"warmup_thread_failed type={type(e).__name__} "
                f"msg={str(e)[:200]}"
            )

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Block until the encoder is loaded or unavailable.

        Use ONLY in tests or one-off scripts. Tool handlers must never
        call this — they should let `encode()` return None and fall back.
        """
        if not self._warmup_started and not self._unavailable:
            self.start_warmup()
        if self._warmup_thread is not None:
            self._warmup_thread.join(timeout=timeout)
        return self.is_loaded

    # ------------------------------------------------------------------
    # State introspection
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Force-load (synchronous) and report whether the model is usable.

        WARNING: triggers a synchronous load on first access — can take
        30s-2min on a cold Windows cache. Almost no caller should use this.
        Prefer `is_loaded` for observation; `start_warmup()` + `is_loaded`
        for non-blocking polling.
        """
        if self._model is None and not self._unavailable:
            self._load()
        return not self._unavailable and self._model is not None

    @property
    def is_loaded(self) -> bool:
        """Whether the encoder is loaded. Never triggers loading."""
        return self._model is not None

    @property
    def is_unavailable(self) -> bool:
        """Whether a load attempt failed. Never triggers loading."""
        return self._unavailable

    @property
    def is_warming(self) -> bool:
        """Whether a background warmup is currently in flight."""
        return (
            self._warmup_started
            and not self.is_loaded
            and not self._unavailable
        )

    @property
    def model(self):
        """Raw SentenceTransformer model, or None if not loaded.

        Callers that need batch-encoding (multiple texts in one call) can
        use this directly. They MUST handle None — never block waiting.
        """
        return self._model

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str):
        """Encode a single text. Returns a normalized numpy vector or None.

        Crucially, this NEVER blocks waiting for the model. If warmup
        hasn't completed, the caller gets None and is expected to fall
        back to substring / structural / regex matching.
        """
        if not self.is_loaded:
            return None
        try:
            import numpy as np
            vec = self._model.encode([text], convert_to_numpy=True)[0]
            n = float(np.linalg.norm(vec))
            return vec / n if n > 0 else vec
        except Exception:
            return None

    def encode_batch(self, texts: List[str]):
        """Encode a list of texts. Returns a normalized numpy matrix or None.

        Same non-blocking contract as `encode()`. None means "not ready."
        Returned matrix is L2-normalized row-wise.
        """
        if not self.is_loaded or not texts:
            return None
        try:
            import numpy as np
            mat = self._model.encode(list(texts), convert_to_numpy=True)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return mat / norms
        except Exception:
            return None
