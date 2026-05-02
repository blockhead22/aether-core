"""NLI cross-encoder contradiction detection.

Why this exists
---------------
The bench in ``bench/slot_coverage_gpt_corpus.py`` measured a 94.92%
paraphrase-blind floor on a real 1,275-conversation corpus: 95% of user
turns produce zero slot tags from the regex extractor, so the
slot-template contradiction layer cannot fire on them.

Aeteros's compression_lab (``D:/CRT/compression_lab/direction2_nli_detection.py``)
demonstrated independently that cross-encoder/nli-deberta-v3-small
achieves 100% contradiction detection accuracy on a fixed
contradiction/non-contradiction corpus, while cosine-based detection
maxed out at 68-82% on the same pairs. That research is what this
module ports.

The architecture finding from compression_lab: **use compressed
vectors for retrieval, NLI on raw text for confirmation.** This module
implements the second half of that pattern. It can run as the
contradiction-confirmation gate when the structural slot-template
detector either fired or did not fire on a candidate pair.

Opt-in via ``AETHER_NLI_CONTRADICTION=1``. Returns neutral scores when
disabled or when the underlying model cannot load. Never raises.

Usage
-----
::

    from aether.contradiction.nli import score_contradiction

    s = score_contradiction(
        "I live in Seattle",
        "I moved to Milwaukee last month",
    )
    print(s.contradiction_prob)  # high
    print(s.label)               # 'contradiction'

For a candidate-pair confirmation gate::

    from aether.contradiction.nli import is_contradiction

    if is_contradiction(memory_a.text, memory_b.text, threshold=0.5):
        # add CONTRADICTS edge

Backend
-------
Uses ``sentence-transformers``'s CrossEncoder API, which is already in
the ``[ml]`` extra. The model ``cross-encoder/nli-deberta-v3-small`` is
~140MB and runs on CPU at ~50-200ms per pair. Loaded lazily — first
call pays the cost, subsequent calls are warm.

If sentence-transformers is unavailable, the module returns neutral
scores rather than failing. The substrate's structural detector
remains the floor.
"""

from __future__ import annotations

import os
import logging
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Tuple

logger = logging.getLogger("aether.contradiction.nli")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_THRESHOLD = 0.5  # P(contradiction) above this → flag

# NLI model output ordering (deberta-v3-small follows MNLI convention):
#   index 0 = contradiction
#   index 1 = entailment
#   index 2 = neutral
LABEL_NAMES = ("contradiction", "entailment", "neutral")


# ---------------------------------------------------------------------------
# NLIScore dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NLIScore:
    """One pair's three-class probabilities plus the argmax label."""
    contradiction_prob: float
    entailment_prob: float
    neutral_prob: float
    label: str  # 'contradiction' | 'entailment' | 'neutral' | 'disabled'

    @property
    def is_contradiction(self) -> bool:
        return self.label == "contradiction"

    @property
    def is_disabled(self) -> bool:
        return self.label == "disabled"


_DISABLED = NLIScore(
    contradiction_prob=0.0,
    entailment_prob=0.0,
    neutral_prob=1.0,
    label="disabled",
)


# ---------------------------------------------------------------------------
# Lazy loader (mirrors aether/_lazy_encoder.py pattern)
# ---------------------------------------------------------------------------


class _LazyCrossEncoder:
    """Lazy-load the cross-encoder. Force HF offline if model is cached."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._load_failed = False

    def _try_load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            # Force HF offline mode if the model is in the local cache.
            # This avoids the connectivity check that causes hangs in
            # subprocess contexts (same fix as _lazy_encoder.py).
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            try:
                # Suppress sentence-transformers' chatty stdout/stderr.
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.model_name)
                logger.info("nli cross-encoder loaded: %s", self.model_name)
                return True
            except Exception as e:
                self._load_failed = True
                logger.warning("nli cross-encoder load failed: %s", e)
                return False

    def predict(self, pairs):
        """Return (N, 3) array of class probabilities, or None on failure."""
        if not self._try_load():
            return None
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                # CrossEncoder.predict returns logits; softmax to get probs.
                import numpy as np
                logits = self._model.predict(pairs, apply_softmax=True)
                return np.asarray(logits)
        except Exception as e:
            logger.warning("nli predict failed: %s", e)
            return None


# Module-level singleton — load once, reuse forever
_loader = _LazyCrossEncoder()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """True iff ``AETHER_NLI_CONTRADICTION`` is set to a truthy value."""
    raw = os.environ.get("AETHER_NLI_CONTRADICTION", "")
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def score_contradiction(text_a: str, text_b: str) -> NLIScore:
    """
    Score a single text pair for contradiction / entailment / neutral.

    Returns ``NLIScore`` with three softmax probabilities and a label.
    Returns the disabled sentinel if the env var is off or the model
    fails to load. Never raises.
    """
    if not is_enabled():
        return _DISABLED
    if not text_a or not text_b:
        return _DISABLED

    probs = _loader.predict([(text_a[:1024], text_b[:1024])])
    if probs is None or len(probs) == 0:
        return _DISABLED

    p = probs[0]
    contradiction, entailment, neutral = float(p[0]), float(p[1]), float(p[2])
    idx_max = max(range(3), key=lambda i: float(p[i]))
    return NLIScore(
        contradiction_prob=contradiction,
        entailment_prob=entailment,
        neutral_prob=neutral,
        label=LABEL_NAMES[idx_max],
    )


def is_contradiction(text_a: str, text_b: str, *, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """
    Convenience wrapper. True iff P(contradiction) > threshold.

    Pass a stricter threshold (0.7+) for high-precision gating.
    Default 0.5 matches the compression_lab harness.
    """
    score = score_contradiction(text_a, text_b)
    if score.is_disabled:
        return False
    return score.contradiction_prob > threshold


def score_pairs(pairs) -> Tuple[Optional[list], str]:
    """
    Batch-score a list of (text_a, text_b) tuples. Faster than calling
    ``score_contradiction`` in a loop because the cross-encoder amortizes
    the encode cost.

    Returns ``(scores, status)`` where status is 'ok', 'disabled', or
    'load_failed'. ``scores`` is a list of NLIScore or None on failure.
    """
    if not is_enabled():
        return None, "disabled"
    if not pairs:
        return [], "ok"

    truncated = [(a[:1024] if a else "", b[:1024] if b else "") for a, b in pairs]
    probs = _loader.predict(truncated)
    if probs is None:
        return None, "load_failed"

    out = []
    for p in probs:
        idx_max = max(range(3), key=lambda i: float(p[i]))
        out.append(
            NLIScore(
                contradiction_prob=float(p[0]),
                entailment_prob=float(p[1]),
                neutral_prob=float(p[2]),
                label=LABEL_NAMES[idx_max],
            )
        )
    return out, "ok"


def diagnostics() -> dict:
    """Report enable state, load state, and model name. For ``aether doctor``."""
    return {
        "enabled": is_enabled(),
        "model_name": _loader.model_name,
        "model_loaded": _loader._model is not None,
        "load_failed": _loader._load_failed,
        "default_threshold": DEFAULT_THRESHOLD,
    }


__all__ = [
    "NLIScore",
    "score_contradiction",
    "is_contradiction",
    "score_pairs",
    "is_enabled",
    "diagnostics",
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
]
