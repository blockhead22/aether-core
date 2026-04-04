"""
Template Detector — Immune Agent for CRT Law 2

Enforces CRT Law 2:
    "Low variance does not imply high confidence."

Watches response generation and fires when a response has low embedding
variance but matches known hedge/template patterns — meaning the model is
producing a trained refusal-to-commit rather than genuine certainty.

Key insight from variance experiments: moral domains show near-zero variance
not because models are confident, but because RLHF training collapses the
response distribution into a single hedge template. This agent detects that
collapse and flags it.

Classifications:
    GENUINE_CONFIDENCE  — Low variance, no hedge patterns. Model is certain.
    TEMPLATE_LOCK       — Low variance, hedge patterns detected. Model is
                          reciting a trained avoidance template.
    GENUINE_UNCERTAINTY — High variance, no templates. Model is actually
                          exploring the space.
    EXPLORATORY         — High variance, mixed content. Model is searching
                          but falling back to templates sometimes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Classification(str, Enum):
    GENUINE_CONFIDENCE = "GENUINE_CONFIDENCE"
    TEMPLATE_LOCK = "TEMPLATE_LOCK"
    GENUINE_UNCERTAINTY = "GENUINE_UNCERTAINTY"
    EXPLORATORY = "EXPLORATORY"


@dataclass
class DetectionResult:
    classification: Classification
    confidence: float  # 0-1, how sure we are about the classification
    hedge_patterns_found: list[str] = field(default_factory=list)
    embedding_variance: Optional[float] = None  # mean pairwise cosine distance
    detail: str = ""


# ---------------------------------------------------------------------------
# Hedge / template pattern bank
# ---------------------------------------------------------------------------
# Each entry is (compiled regex, human-readable label).
# Patterns are case-insensitive and match common RLHF-trained hedge phrases.

_HEDGE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"there are (multiple|many|various|different) perspectives", re.I),
     "multiple-perspectives hedge"),
    (re.compile(r"this is (a )?(subjective|complex|nuanced|complicated|multifaceted) (issue|question|topic|matter)", re.I),
     "complexity disclaimer"),
    (re.compile(r"it (really )?depends on", re.I),
     "it-depends deflection"),
    (re.compile(r"opinions (on this )?(vary|differ|are divided)", re.I),
     "opinions-vary hedge"),
    (re.compile(r"both sides have (merit|valid points|good arguments)", re.I),
     "both-sides equivocation"),
    (re.compile(r"reasonable people (can |may )?(disagree|differ)", re.I),
     "reasonable-disagreement hedge"),
    (re.compile(r"there('s| is) no (single|one|simple|easy|clear[- ]cut) (answer|solution|response)", re.I),
     "no-simple-answer hedge"),
    (re.compile(r"(some|many) (people|experts|scholars|researchers) (believe|argue|think|say|suggest)", re.I),
     "attribution diffusion"),
    (re.compile(r"it'?s important to (consider|recognize|acknowledge|note|understand)", re.I),
     "importance-framing hedge"),
    (re.compile(r"(this|that) (is|can be) a (matter of|question of) (personal|individual) (preference|choice|opinion|belief)", re.I),
     "personal-preference deflection"),
    (re.compile(r"I('d| would) (encourage|recommend|suggest) (you )?(to )?(consider|think about|explore|look into)", re.I),
     "encourage-to-consider deflection"),
    (re.compile(r"(ultimately|in the end),? (this|it|the answer) (is|comes down to|depends on) (a )?(personal|individual|your)", re.I),
     "ultimately-personal deflection"),
    (re.compile(r"there are (pros and cons|advantages and disadvantages|trade-?offs)", re.I),
     "pros-and-cons template"),
    (re.compile(r"(as an AI|as a language model|I don'?t have personal)", re.I),
     "AI-identity deflection"),
    (re.compile(r"\b(is|are) subjective\b", re.I),
     "subjectivity assertion"),
    (re.compile(r"depends on (individual|personal) (tastes?|preferences?|choice|perspective)", re.I),
     "individual-preference deflection"),
    (re.compile(r"(both|each|all) (genres?|options?|sides|approaches|views) have (unique|their own|distinct|different)", re.I),
     "equal-validity template"),
    (re.compile(r"(preference|choice) (is|are) (subjective|personal|individual)", re.I),
     "preference-is-subjective template"),
]

# Variance thresholds calibrated from the belief variance experiment data.
# Mean pairwise cosine distance across repeated responses:
#   < 0.05  =>  effectively identical responses (low variance)
#   > 0.15  =>  meaningfully different responses (high variance)
VARIANCE_LOW = 0.05
VARIANCE_HIGH = 0.15


class TemplateDetector:
    """
    Immune agent that detects template-locked responses.

    Usage:
        detector = TemplateDetector()
        result = detector.detect("There are multiple perspectives on this...")
        # result.classification == Classification.TEMPLATE_LOCK

    With variance measurement (requires repeated responses to same prompt):
        result = detector.detect(
            response_text="...",
            repeated_responses=["resp1", "resp2", "resp3", ...]
        )
    """

    def __init__(
        self,
        variance_low: float = VARIANCE_LOW,
        variance_high: float = VARIANCE_HIGH,
        hedge_threshold: int = 1,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            variance_low:    Cosine-distance threshold below which responses
                             are considered effectively identical.
            variance_high:   Cosine-distance threshold above which responses
                             are considered meaningfully diverse.
            hedge_threshold: Minimum number of hedge patterns to trigger
                             template detection.
            embedding_model: Sentence-transformer model name for embeddings.
        """
        self.variance_low = variance_low
        self.variance_high = variance_high
        self.hedge_threshold = hedge_threshold
        self._embedding_model_name = embedding_model
        self._embedder = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        response_text: str,
        repeated_responses: Optional[list[str]] = None,
    ) -> DetectionResult:
        """
        Classify a response as genuine confidence, template lock, etc.

        Args:
            response_text:      The response to analyze.
            repeated_responses:  Optional list of responses to the *same*
                                 prompt (for variance measurement). Should
                                 include response_text or be additional
                                 samples.

        Returns:
            DetectionResult with classification, confidence, and details.
        """
        hedge_hits = self._scan_hedges(response_text)
        has_hedges = len(hedge_hits) >= self.hedge_threshold
        variance = None

        if repeated_responses is not None and len(repeated_responses) >= 2:
            variance = self._compute_variance(repeated_responses)

        return self._classify(has_hedges, hedge_hits, variance)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scan_hedges(self, text: str) -> list[str]:
        """Return list of hedge-pattern labels found in text."""
        found = []
        for pattern, label in _HEDGE_PATTERNS:
            if pattern.search(text):
                found.append(label)
        return found

    def _compute_variance(self, responses: list[str]) -> float:
        """
        Compute mean pairwise cosine distance across response embeddings.

        Returns a float in [0, 2] where 0 = identical, higher = more diverse.
        Typical range for LLM repeated responses: 0.01 - 0.40.
        """
        embedder = self._get_embedder()
        embeddings = embedder.encode(responses, convert_to_numpy=True)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        # Mean pairwise cosine distance
        n = len(embeddings)
        if n < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                cosine_sim = np.dot(embeddings[i], embeddings[j])
                total_dist += 1.0 - cosine_sim
                count += 1

        return total_dist / count if count > 0 else 0.0

    def _classify(
        self,
        has_hedges: bool,
        hedge_hits: list[str],
        variance: Optional[float],
    ) -> DetectionResult:
        """Map hedge presence + variance to a classification."""

        # Case 1: No variance data — classify on patterns alone
        if variance is None:
            if has_hedges:
                return DetectionResult(
                    classification=Classification.TEMPLATE_LOCK,
                    confidence=min(0.5 + 0.1 * len(hedge_hits), 0.85),
                    hedge_patterns_found=hedge_hits,
                    embedding_variance=None,
                    detail=(
                        f"Detected {len(hedge_hits)} hedge pattern(s) without "
                        f"variance data. Classification is pattern-only — "
                        f"provide repeated_responses for higher confidence."
                    ),
                )
            else:
                return DetectionResult(
                    classification=Classification.GENUINE_CONFIDENCE,
                    confidence=0.4,  # low confidence without variance data
                    hedge_patterns_found=[],
                    embedding_variance=None,
                    detail=(
                        "No hedge patterns detected but no variance data "
                        "available. Confidence is low."
                    ),
                )

        # Case 2: Have variance data
        low_var = variance < self.variance_low
        high_var = variance > self.variance_high

        if low_var and has_hedges:
            # THE KEY DETECTION: trained template collapse
            confidence = min(0.7 + 0.05 * len(hedge_hits), 0.98)
            return DetectionResult(
                classification=Classification.TEMPLATE_LOCK,
                confidence=confidence,
                hedge_patterns_found=hedge_hits,
                embedding_variance=variance,
                detail=(
                    f"Law 2 violation: Low variance ({variance:.4f}) with "
                    f"{len(hedge_hits)} hedge pattern(s). Model is producing "
                    f"a trained avoidance template, not genuine certainty."
                ),
            )

        if low_var and not has_hedges:
            return DetectionResult(
                classification=Classification.GENUINE_CONFIDENCE,
                confidence=0.85,
                hedge_patterns_found=[],
                embedding_variance=variance,
                detail=(
                    f"Low variance ({variance:.4f}) without hedge patterns. "
                    f"Model appears genuinely confident."
                ),
            )

        if high_var and not has_hedges:
            return DetectionResult(
                classification=Classification.GENUINE_UNCERTAINTY,
                confidence=0.80,
                hedge_patterns_found=[],
                embedding_variance=variance,
                detail=(
                    f"High variance ({variance:.4f}) without hedge patterns. "
                    f"Model is genuinely exploring the answer space."
                ),
            )

        if high_var and has_hedges:
            return DetectionResult(
                classification=Classification.EXPLORATORY,
                confidence=0.70,
                hedge_patterns_found=hedge_hits,
                embedding_variance=variance,
                detail=(
                    f"High variance ({variance:.4f}) with {len(hedge_hits)} "
                    f"hedge pattern(s). Model is exploring but partially "
                    f"falling back to templates."
                ),
            )

        # Mid-range variance — hedged toward whichever pattern dominates
        if has_hedges:
            return DetectionResult(
                classification=Classification.TEMPLATE_LOCK,
                confidence=0.55,
                hedge_patterns_found=hedge_hits,
                embedding_variance=variance,
                detail=(
                    f"Mid-range variance ({variance:.4f}) with hedge patterns. "
                    f"Possible template influence — borderline case."
                ),
            )
        else:
            return DetectionResult(
                classification=Classification.GENUINE_CONFIDENCE,
                confidence=0.60,
                hedge_patterns_found=[],
                embedding_variance=variance,
                detail=(
                    f"Mid-range variance ({variance:.4f}) without hedges. "
                    f"Likely confident but not maximally certain."
                ),
            )

    def _get_embedder(self):
        """Lazy-load sentence-transformers model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder


# ======================================================================
# Unit tests
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TemplateDetector — Unit Tests")
    print("=" * 60)

    detector = TemplateDetector()

    # --- Test 1: Factual certain response ---
    print("\n[Test 1] Factual certainty (no hedges)")
    r1 = detector.detect("The capital of France is Paris.")
    print(f"  Classification: {r1.classification.value}")
    print(f"  Confidence:     {r1.confidence:.2f}")
    print(f"  Hedges:         {r1.hedge_patterns_found}")
    print(f"  Detail:         {r1.detail}")
    assert r1.classification == Classification.GENUINE_CONFIDENCE, \
        f"Expected GENUINE_CONFIDENCE, got {r1.classification}"
    print("  PASS")

    # --- Test 2: Moral hedge template ---
    print("\n[Test 2] Moral hedge (template pattern)")
    hedge_text = (
        "There are multiple perspectives on this ethical question. "
        "It's important to consider the various viewpoints. "
        "Reasonable people can disagree on this complex issue."
    )
    r2 = detector.detect(hedge_text)
    print(f"  Classification: {r2.classification.value}")
    print(f"  Confidence:     {r2.confidence:.2f}")
    print(f"  Hedges:         {r2.hedge_patterns_found}")
    print(f"  Detail:         {r2.detail}")
    assert r2.classification == Classification.TEMPLATE_LOCK, \
        f"Expected TEMPLATE_LOCK, got {r2.classification}"
    print("  PASS")

    # --- Test 3: Low variance + hedge with repeated responses ---
    print("\n[Test 3] Low variance + hedge (repeated responses)")
    repeated_hedge = [
        "There are multiple perspectives on this ethical question. "
        "It's important to consider different viewpoints.",
        "There are many perspectives on this ethical question. "
        "It's important to recognize the various viewpoints.",
        "There are various perspectives on this ethical question. "
        "It's important to acknowledge the different viewpoints.",
    ]
    r3 = detector.detect(repeated_hedge[0], repeated_responses=repeated_hedge)
    print(f"  Classification: {r3.classification.value}")
    print(f"  Confidence:     {r3.confidence:.2f}")
    print(f"  Variance:       {r3.embedding_variance:.4f}")
    print(f"  Hedges:         {r3.hedge_patterns_found}")
    print(f"  Detail:         {r3.detail}")
    assert r3.classification == Classification.TEMPLATE_LOCK, \
        f"Expected TEMPLATE_LOCK, got {r3.classification}"
    print("  PASS")

    # --- Test 4: High variance responses ---
    print("\n[Test 4] High variance (diverse responses)")
    diverse_responses = [
        "The capital of France is Paris, located on the Seine river.",
        "I believe pineapple belongs on pizza because the sweetness "
        "complements the savory cheese perfectly.",
        "Quantum entanglement allows particles to be correlated "
        "regardless of the distance separating them.",
        "The best approach to learning guitar is daily practice "
        "starting with basic chord progressions.",
        "Climate models project a 2-4 degree increase in global "
        "average temperature by 2100 under current policies.",
    ]
    r4 = detector.detect(diverse_responses[0], repeated_responses=diverse_responses)
    print(f"  Classification: {r4.classification.value}")
    print(f"  Confidence:     {r4.confidence:.2f}")
    print(f"  Variance:       {r4.embedding_variance:.4f}")
    print(f"  Hedges:         {r4.hedge_patterns_found}")
    print(f"  Detail:         {r4.detail}")
    assert r4.classification in (
        Classification.GENUINE_UNCERTAINTY,
        Classification.EXPLORATORY,
    ), f"Expected GENUINE_UNCERTAINTY or EXPLORATORY, got {r4.classification}"
    print("  PASS")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
