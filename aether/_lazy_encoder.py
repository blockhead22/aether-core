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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
        """
        if self._model is not None or self._unavailable:
            return
        cached = _MODEL_CACHE.get(self.model_name)
        if cached is not None:
            self._model = cached
            return
        lock = _get_cache_lock()
        with lock:
            cached = _MODEL_CACHE.get(self.model_name)
            if cached is not None:
                self._model = cached
                return
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = model
                self._model = model
            except Exception:
                self._unavailable = True

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
        except Exception:
            # Threading failure: degrade gracefully without blocking.
            self._unavailable = True

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
