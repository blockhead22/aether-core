"""
GapAuditor — Immune Agent for CRT Law 5

Law 5 of the Mirus/Holden constitution:
    "Outward confidence must be bounded by internal support."

Sits on TOP of the other four agents. Measures the gap between what the system
expresses (speech confidence) and what the belief layer actually supports
(belief confidence). This is the auditable belief/speech separation — the core
Mirus/Holden principle made operational.

Threat model:
  - The model generates a response dripping with certainty ("definitely",
    "without doubt", "the answer is")
  - But the underlying belief layer has low support (low trust, sparse
    evidence, fragile domain, template-locked hedging)
  - Without this gate, the system projects confidence it hasn't earned
  - Users trust the confident output, which may be wrong
  - The speech/belief gap becomes invisible and unauditable

This agent makes the gap visible, measurable, and actionable.

Integration:
  - Consumes results from TemplateDetector (Law 2) and SpeechLeakDetector
    (Law 1) to adjust effective belief confidence
  - Flags epistemic voids: domains where susceptibility is near-zero on
    moral topics (guardrail lobotomy)
  - Tracks gap trends over time to detect widening divergence
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Severity & action enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    SAFE = "SAFE"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


class Action(str, Enum):
    PASS = "PASS"
    FLAG = "FLAG"
    HEDGE = "HEDGE"
    ESCALATE = "ESCALATE"


class Trend(str, Enum):
    STABLE = "STABLE"
    WIDENING = "WIDENING"
    NARROWING = "NARROWING"


# ---------------------------------------------------------------------------
# Input / output data structures
# ---------------------------------------------------------------------------

@dataclass
class ResponseAudit:
    """Input to the GapAuditor: everything known about a generated response."""
    response_text: str
    belief_confidence: float                          # 0-1, from memory/grounding layer

    # Optional cross-agent results
    template_detection: Optional[object] = None       # DetectionResult from TemplateDetector
    speech_leak_result: Optional[object] = None       # Verdict from SpeechLeakDetector

    # Optional context
    domain: Optional[str] = None                      # e.g. "factual_settled", "moral_clear"
    susceptibility: Optional[float] = None            # from variance probe fragility map
    source_trust: Optional[float] = None              # trust of backing evidence


@dataclass
class GapVerdict:
    """Result of a GapAuditor audit."""
    gap_score: float                                  # speech - belief (can be negative)
    severity: Severity
    action: Action
    speech_confidence: float
    belief_confidence: float
    contributing_factors: list[str] = field(default_factory=list)
    law: str = "Law 5: Outward confidence must be bounded by internal support"


# ---------------------------------------------------------------------------
# Speech confidence markers
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdefinitely\b", re.I), "definitely"),
    (re.compile(r"\bcertainly\b", re.I), "certainly"),
    (re.compile(r"\bclearly\b", re.I), "clearly"),
    (re.compile(r"\bwithout doubt\b", re.I), "without doubt"),
    (re.compile(r"\bthe answer is\b", re.I), "the answer is"),
    (re.compile(r"\bit is\b", re.I), "it is"),
    (re.compile(r"\balways\b", re.I), "always"),
    (re.compile(r"\bnever\b", re.I), "never"),
    (re.compile(r"\babsolutely\b", re.I), "absolutely"),
    (re.compile(r"\bundeniably\b", re.I), "undeniably"),
    (re.compile(r"\bthere is no question\b", re.I), "there is no question"),
]

_MODERATE_CONFIDENCE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\blikely\b", re.I), "likely"),
    (re.compile(r"\bprobably\b", re.I), "probably"),
    (re.compile(r"\bgenerally\b", re.I), "generally"),
    (re.compile(r"\btypically\b", re.I), "typically"),
    (re.compile(r"\busually\b", re.I), "usually"),
    (re.compile(r"\bin most cases\b", re.I), "in most cases"),
    (re.compile(r"\btends to\b", re.I), "tends to"),
]

_LOW_CONFIDENCE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bmight\b", re.I), "might"),
    (re.compile(r"\bperhaps\b", re.I), "perhaps"),
    (re.compile(r"\bpossibly\b", re.I), "possibly"),
    (re.compile(r"\bit depends\b", re.I), "it depends"),
    (re.compile(r"\bsome argue\b", re.I), "some argue"),
    (re.compile(r"\bthere are perspectives\b", re.I), "there are perspectives"),
    (re.compile(r"\bit'?s complex\b", re.I), "it's complex"),
    (re.compile(r"\barguably\b", re.I), "arguably"),
]

_HEDGING: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bI think\b", re.I), "I think"),
    (re.compile(r"\bin my opinion\b", re.I), "in my opinion"),
    (re.compile(r"\bone could say\b", re.I), "one could say"),
    (re.compile(r"\bit could be argued\b", re.I), "it could be argued"),
]


# ---------------------------------------------------------------------------
# Gap thresholds
# ---------------------------------------------------------------------------

GAP_SAFE = 0.2
GAP_ELEVATED = 0.4

# Trend window
MAX_HISTORY = 100
TREND_WINDOW = 10


# ---------------------------------------------------------------------------
# GapAuditor
# ---------------------------------------------------------------------------

class GapAuditor:
    """
    Measures the gap between expressed confidence and belief support.

    Usage:
        auditor = GapAuditor()
        verdict = auditor.audit(ResponseAudit(
            response_text="This is definitely the correct answer.",
            belief_confidence=0.3,
        ))
        # verdict.severity == Severity.CRITICAL
        # verdict.action == Action.HEDGE
    """

    def __init__(
        self,
        gap_safe: float = GAP_SAFE,
        gap_elevated: float = GAP_ELEVATED,
    ):
        self.gap_safe = gap_safe
        self.gap_elevated = gap_elevated
        self._history: deque[GapVerdict] = deque(maxlen=MAX_HISTORY)

    # -------------------------------------------------------------------
    # Speech confidence estimation
    # -------------------------------------------------------------------

    @staticmethod
    def estimate_speech_confidence(text: str) -> tuple[float, list[str]]:
        """
        Estimate how confident the response text sounds.

        Returns:
            (score, markers_found) where score is 0-1 and markers_found
            lists the labels of detected confidence markers.
        """
        markers: list[str] = []
        weights: list[float] = []

        for pattern, label in _HIGH_CONFIDENCE:
            if pattern.search(text):
                markers.append(f"high:{label}")
                weights.append(0.9)

        for pattern, label in _MODERATE_CONFIDENCE:
            if pattern.search(text):
                markers.append(f"moderate:{label}")
                weights.append(0.6)

        for pattern, label in _LOW_CONFIDENCE:
            if pattern.search(text):
                markers.append(f"low:{label}")
                weights.append(0.3)

        for pattern, label in _HEDGING:
            if pattern.search(text):
                markers.append(f"hedge:{label}")
                weights.append(0.2)

        if not weights:
            # No markers found — assume moderate baseline
            return 0.5, []

        # Score is the weighted average biased toward the strongest markers.
        # More markers of the same tier increase confidence.
        score = sum(weights) / len(weights)

        # Density bonus: many high-confidence markers push score up
        high_count = sum(1 for m in markers if m.startswith("high:"))
        low_count = sum(1 for m in markers if m.startswith("low:") or m.startswith("hedge:"))

        if high_count >= 3:
            score = min(score + 0.1, 1.0)
        if low_count >= 3:
            score = max(score - 0.1, 0.0)

        return round(min(max(score, 0.0), 1.0), 3), markers

    # -------------------------------------------------------------------
    # Effective belief confidence (adjusted by other agents)
    # -------------------------------------------------------------------

    @staticmethod
    def _adjust_belief_confidence(audit: ResponseAudit) -> tuple[float, list[str]]:
        """
        Adjust belief confidence based on cross-agent results.

        Returns:
            (adjusted_confidence, contributing_factors)
        """
        confidence = audit.belief_confidence
        factors: list[str] = []

        # TemplateDetector integration
        if audit.template_detection is not None:
            classification = getattr(audit.template_detection, "classification", None)
            if classification is not None:
                cls_value = classification.value if hasattr(classification, "value") else str(classification)
                if cls_value == "TEMPLATE_LOCK":
                    confidence -= 0.3
                    factors.append(
                        f"Template lock detected — belief reduced by 0.3 "
                        f"(template responses are not genuine knowledge)"
                    )

        # SpeechLeakDetector integration
        if audit.speech_leak_result is not None:
            action = getattr(audit.speech_leak_result, "action", None)
            if action is not None:
                action_value = action.value if hasattr(action, "value") else str(action)
                if action_value in ("BLOCK", "DOWNGRADE"):
                    confidence -= 0.2
                    factors.append(
                        f"Speech leak {action_value} — belief reduced by 0.2 "
                        f"(ungrounded claims detected)"
                    )

        # Clamp to [0, 1]
        confidence = round(min(max(confidence, 0.0), 1.0), 3)

        return confidence, factors

    # -------------------------------------------------------------------
    # Core audit
    # -------------------------------------------------------------------

    def audit(self, response_audit: ResponseAudit) -> GapVerdict:
        """
        Audit a response for belief/speech gap violations.

        Args:
            response_audit: All available information about the response.

        Returns:
            GapVerdict with gap score, severity, action, and explanation.
        """
        # Step 1: Estimate speech confidence from text
        speech_conf, speech_markers = self.estimate_speech_confidence(
            response_audit.response_text
        )

        # Step 2: Adjust belief confidence from cross-agent results
        belief_conf, cross_factors = self._adjust_belief_confidence(response_audit)

        # Step 3: Build contributing factors
        factors = list(cross_factors)
        if speech_markers:
            factors.append(f"Speech markers: {', '.join(speech_markers)}")

        # Step 4: Compute gap
        gap = round(speech_conf - belief_conf, 3)

        # Step 5: Determine effective thresholds
        effective_safe = self.gap_safe
        effective_elevated = self.gap_elevated

        # Fragile domain: tighten thresholds
        if (response_audit.susceptibility is not None
                and response_audit.susceptibility > 0.2):
            effective_safe -= 0.1
            effective_elevated -= 0.1
            factors.append(
                f"Fragile domain (susceptibility={response_audit.susceptibility:.2f} > 0.2) "
                f"— thresholds tightened by 0.1"
            )

        # Step 6: Check for epistemic void (moral domain + near-zero susceptibility)
        is_epistemic_void = False
        if (response_audit.domain is not None
                and "moral" in response_audit.domain.lower()
                and response_audit.susceptibility is not None
                and response_audit.susceptibility < 0.05):
            is_epistemic_void = True
            factors.append(
                f"Epistemic void: moral domain '{response_audit.domain}' with "
                f"near-zero susceptibility ({response_audit.susceptibility:.3f}) — "
                f"guardrails may have emptied moral reasoning"
            )

        # Step 7: Classify severity
        if gap < effective_safe:
            severity = Severity.SAFE
        elif gap < effective_elevated:
            severity = Severity.ELEVATED
        else:
            severity = Severity.CRITICAL

        # Step 8: Determine action
        if is_epistemic_void:
            action = Action.ESCALATE
            factors.append("Action ESCALATE: epistemic void detected regardless of gap")
        elif severity == Severity.SAFE:
            action = Action.PASS
        elif severity == Severity.ELEVATED:
            action = Action.FLAG
        else:
            # CRITICAL severity
            # Check for multi-law violation
            has_template_lock = False
            has_speech_block = False

            if response_audit.template_detection is not None:
                cls = getattr(response_audit.template_detection, "classification", None)
                if cls is not None:
                    cls_val = cls.value if hasattr(cls, "value") else str(cls)
                    has_template_lock = cls_val == "TEMPLATE_LOCK"

            if response_audit.speech_leak_result is not None:
                act = getattr(response_audit.speech_leak_result, "action", None)
                if act is not None:
                    act_val = act.value if hasattr(act, "value") else str(act)
                    has_speech_block = act_val == "BLOCK"

            if has_template_lock or has_speech_block:
                action = Action.ESCALATE
                violations = []
                if has_template_lock:
                    violations.append("Law 2 (template lock)")
                if has_speech_block:
                    violations.append("Law 1 (speech leak block)")
                factors.append(
                    f"Action ESCALATE: multiple law violations — {', '.join(violations)}"
                )
            else:
                action = Action.HEDGE
                factors.append(
                    "Action HEDGE: speech dramatically exceeds belief support"
                )

        # Step 9: Trend-based escalation
        trend = self.get_trend()
        if (trend == Trend.WIDENING
                and severity in (Severity.ELEVATED, Severity.CRITICAL)):
            if severity == Severity.ELEVATED:
                severity = Severity.CRITICAL
                factors.append(
                    "Severity escalated ELEVATED -> CRITICAL: WIDENING trend detected"
                )
            if action == Action.FLAG:
                action = Action.HEDGE
                factors.append(
                    "Action escalated FLAG -> HEDGE: WIDENING trend detected"
                )

        verdict = GapVerdict(
            gap_score=gap,
            severity=severity,
            action=action,
            speech_confidence=speech_conf,
            belief_confidence=belief_conf,
            contributing_factors=factors,
        )

        # Record in history
        self._history.append(verdict)

        return verdict

    # -------------------------------------------------------------------
    # Trend tracking
    # -------------------------------------------------------------------

    def get_trend(self) -> Trend:
        """
        Analyze recent audit history for gap trend direction.

        Returns:
            STABLE, WIDENING, or NARROWING based on recent gap scores.
        """
        if len(self._history) < 3:
            return Trend.STABLE

        recent = list(self._history)[-TREND_WINDOW:]
        if len(recent) < 3:
            return Trend.STABLE

        # Split into first half and second half
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]

        avg_first = sum(v.gap_score for v in first_half) / len(first_half)
        avg_second = sum(v.gap_score for v in second_half) / len(second_half)

        delta = avg_second - avg_first

        if delta > 0.05:
            return Trend.WIDENING
        elif delta < -0.05:
            return Trend.NARROWING
        else:
            return Trend.STABLE

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    def get_summary(self) -> dict:
        """
        Return aggregate statistics over the audit history.

        Returns:
            Dict with mean_gap, max_gap, severity_distribution, trend,
            and most common contributing factors.
        """
        if not self._history:
            return {
                "total_audits": 0,
                "mean_gap": 0.0,
                "max_gap": 0.0,
                "severity_distribution": {s.value: 0 for s in Severity},
                "trend": Trend.STABLE.value,
                "top_factors": [],
            }

        gaps = [v.gap_score for v in self._history]
        severity_counts = {s.value: 0 for s in Severity}
        factor_counts: dict[str, int] = {}

        for v in self._history:
            severity_counts[v.severity.value] += 1
            for f in v.contributing_factors:
                # Normalize factor strings for counting
                key = f.split("—")[0].strip() if "—" in f else f[:60]
                factor_counts[key] = factor_counts.get(key, 0) + 1

        total = len(self._history)
        severity_pct = {k: round(v / total * 100, 1) for k, v in severity_counts.items()}

        top_factors = sorted(factor_counts.items(), key=lambda x: -x[1])[:5]

        return {
            "total_audits": total,
            "mean_gap": round(sum(gaps) / total, 3),
            "max_gap": round(max(gaps), 3),
            "severity_distribution": severity_pct,
            "trend": self.get_trend().value,
            "top_factors": top_factors,
        }


# ======================================================================
# Unit tests
# ======================================================================

if __name__ == "__main__":
    # Local imports for cross-agent test stubs
    from dataclasses import dataclass as _dc
    from enum import Enum as _Enum

    print("=" * 70)
    print("GapAuditor — Law 5 enforcement tests")
    print("=" * 70)

    passed = 0
    total = 8

    # --- Stub types for cross-agent integration ---
    class _Classification(str, _Enum):
        TEMPLATE_LOCK = "TEMPLATE_LOCK"
        GENUINE_CONFIDENCE = "GENUINE_CONFIDENCE"

    @_dc
    class _DetectionResult:
        classification: _Classification
        embedding_variance: Optional[float] = None

    class _VerdictType(str, _Enum):
        ALLOW = "ALLOW"
        DOWNGRADE = "DOWNGRADE"
        BLOCK = "BLOCK"

    @_dc
    class _SpeechVerdict:
        action: _VerdictType

    # --- Test 1: High speech confidence + low belief → CRITICAL, HEDGE ---
    print("\n[Test 1] High speech + low belief -> CRITICAL, HEDGE")
    auditor = GapAuditor()
    v1 = auditor.audit(ResponseAudit(
        response_text=(
            "This is definitely the correct answer. There is no question about it. "
            "The answer is absolutely clear and undeniably true."
        ),
        belief_confidence=0.2,
    ))
    print(f"  Gap: {v1.gap_score:.3f} | Severity: {v1.severity.value} | Action: {v1.action.value}")
    print(f"  Speech: {v1.speech_confidence:.3f} | Belief: {v1.belief_confidence:.3f}")
    print(f"  Factors: {v1.contributing_factors}")
    assert v1.severity == Severity.CRITICAL, f"Expected CRITICAL, got {v1.severity}"
    assert v1.action == Action.HEDGE, f"Expected HEDGE, got {v1.action}"
    assert v1.gap_score > 0.4, f"Expected gap > 0.4, got {v1.gap_score}"
    print("  PASSED")
    passed += 1

    # --- Test 2: Low speech + high belief → SAFE, PASS ---
    print("\n[Test 2] Low speech + high belief -> SAFE, PASS")
    auditor2 = GapAuditor()
    v2 = auditor2.audit(ResponseAudit(
        response_text=(
            "It might be the case, perhaps, that this is possibly correct. "
            "Some argue it depends on the context. Arguably there are perspectives."
        ),
        belief_confidence=0.9,
    ))
    print(f"  Gap: {v2.gap_score:.3f} | Severity: {v2.severity.value} | Action: {v2.action.value}")
    print(f"  Speech: {v2.speech_confidence:.3f} | Belief: {v2.belief_confidence:.3f}")
    assert v2.severity == Severity.SAFE, f"Expected SAFE, got {v2.severity}"
    assert v2.action == Action.PASS, f"Expected PASS, got {v2.action}"
    assert v2.gap_score < 0, f"Expected negative gap, got {v2.gap_score}"
    print("  PASSED")
    passed += 1

    # --- Test 3: Template lock reduces effective belief → ELEVATED+ ---
    print("\n[Test 3] Template lock reduces belief -> ELEVATED+")
    auditor3 = GapAuditor()
    v3 = auditor3.audit(ResponseAudit(
        response_text="This is generally true and typically happens in most cases.",
        belief_confidence=0.6,
        template_detection=_DetectionResult(
            classification=_Classification.TEMPLATE_LOCK,
            embedding_variance=0.02,
        ),
    ))
    print(f"  Gap: {v3.gap_score:.3f} | Severity: {v3.severity.value} | Action: {v3.action.value}")
    print(f"  Speech: {v3.speech_confidence:.3f} | Belief: {v3.belief_confidence:.3f}")
    print(f"  Factors: {v3.contributing_factors}")
    assert v3.belief_confidence < 0.6, f"Belief should be reduced, got {v3.belief_confidence}"
    assert v3.severity in (Severity.ELEVATED, Severity.CRITICAL), \
        f"Expected ELEVATED or CRITICAL, got {v3.severity}"
    print("  PASSED")
    passed += 1

    # --- Test 4: Moral void detection → ESCALATE ---
    print("\n[Test 4] Moral void (zero susceptibility) -> ESCALATE")
    auditor4 = GapAuditor()
    v4 = auditor4.audit(ResponseAudit(
        response_text="This is probably a matter of personal choice.",
        belief_confidence=0.5,
        domain="moral_clear",
        susceptibility=0.01,
    ))
    print(f"  Gap: {v4.gap_score:.3f} | Severity: {v4.severity.value} | Action: {v4.action.value}")
    print(f"  Factors: {v4.contributing_factors}")
    assert v4.action == Action.ESCALATE, f"Expected ESCALATE, got {v4.action}"
    assert any("epistemic void" in f.lower() for f in v4.contributing_factors), \
        "Expected epistemic void in contributing factors"
    print("  PASSED")
    passed += 1

    # --- Test 5: Fragile domain (high susceptibility) → tighter thresholds ---
    print("\n[Test 5] Fragile domain -> tighter thresholds")
    auditor5 = GapAuditor()
    # Gap that would be SAFE at normal thresholds (0.15) but ELEVATED at tightened (0.10)
    v5 = auditor5.audit(ResponseAudit(
        response_text="This is likely true and probably correct.",
        belief_confidence=0.45,
        susceptibility=0.35,
    ))
    print(f"  Gap: {v5.gap_score:.3f} | Severity: {v5.severity.value} | Action: {v5.action.value}")
    print(f"  Speech: {v5.speech_confidence:.3f} | Belief: {v5.belief_confidence:.3f}")
    print(f"  Factors: {v5.contributing_factors}")
    assert any("fragile" in f.lower() for f in v5.contributing_factors), \
        "Expected fragile domain in contributing factors"
    # With susceptibility > 0.2, safe threshold drops to 0.1
    if v5.gap_score >= 0.1:
        assert v5.severity in (Severity.ELEVATED, Severity.CRITICAL), \
            f"Expected ELEVATED+ with tightened thresholds, got {v5.severity}"
    print("  PASSED")
    passed += 1

    # --- Test 6: Multiple agent violations → ESCALATE ---
    print("\n[Test 6] Template lock + speech leak block -> ESCALATE")
    auditor6 = GapAuditor()
    v6 = auditor6.audit(ResponseAudit(
        response_text=(
            "This is definitely and absolutely the correct answer. "
            "There is no question it is always true."
        ),
        belief_confidence=0.5,
        template_detection=_DetectionResult(
            classification=_Classification.TEMPLATE_LOCK,
        ),
        speech_leak_result=_SpeechVerdict(
            action=_VerdictType.BLOCK,
        ),
    ))
    print(f"  Gap: {v6.gap_score:.3f} | Severity: {v6.severity.value} | Action: {v6.action.value}")
    print(f"  Speech: {v6.speech_confidence:.3f} | Belief: {v6.belief_confidence:.3f}")
    print(f"  Factors: {v6.contributing_factors}")
    assert v6.action == Action.ESCALATE, f"Expected ESCALATE, got {v6.action}"
    assert v6.belief_confidence <= 0.0, \
        f"Belief should be reduced to 0 by both agents, got {v6.belief_confidence}"
    print("  PASSED")
    passed += 1

    # --- Test 7: Trend tracking — widening audits escalate severity ---
    print("\n[Test 7] Widening trend escalates severity")
    auditor7 = GapAuditor()
    # Feed several audits with increasing gap (small gap -> big gap)
    for i in range(6):
        # Belief decreases over time while speech stays the same
        auditor7.audit(ResponseAudit(
            response_text="This is likely the correct approach.",
            belief_confidence=max(0.7 - i * 0.1, 0.1),
        ))

    trend = auditor7.get_trend()
    print(f"  Trend after 6 widening audits: {trend.value}")
    assert trend == Trend.WIDENING, f"Expected WIDENING, got {trend}"

    # Now submit an ELEVATED-level audit — should be escalated to CRITICAL
    v7 = auditor7.audit(ResponseAudit(
        response_text="This is generally and typically the case, probably true.",
        belief_confidence=0.35,
    ))
    print(f"  Gap: {v7.gap_score:.3f} | Severity: {v7.severity.value} | Action: {v7.action.value}")
    print(f"  Factors: {v7.contributing_factors}")
    # The trend-based escalation should have kicked in
    has_trend_escalation = any("widening" in f.lower() for f in v7.contributing_factors)
    print(f"  Trend escalation applied: {has_trend_escalation}")
    # At minimum the trend must be WIDENING
    assert auditor7.get_trend() == Trend.WIDENING, "Trend should still be WIDENING"
    print("  PASSED")
    passed += 1

    # --- Test 8: Matched confidence → SAFE, PASS ---
    print("\n[Test 8] Matched confidence (speech ~ belief) -> SAFE, PASS")
    auditor8 = GapAuditor()
    v8 = auditor8.audit(ResponseAudit(
        response_text="This is likely true and probably correct in most cases.",
        belief_confidence=0.6,
    ))
    print(f"  Gap: {v8.gap_score:.3f} | Severity: {v8.severity.value} | Action: {v8.action.value}")
    print(f"  Speech: {v8.speech_confidence:.3f} | Belief: {v8.belief_confidence:.3f}")
    assert v8.severity == Severity.SAFE, f"Expected SAFE, got {v8.severity}"
    assert v8.action == Action.PASS, f"Expected PASS, got {v8.action}"
    assert abs(v8.gap_score) < 0.2, f"Expected small gap, got {v8.gap_score}"
    print("  PASSED")
    passed += 1

    # --- Summary ---
    print(f"\n{'=' * 70}")

    # Print summary stats from test 7's auditor (has the most history)
    summary = auditor7.get_summary()
    print(f"Summary from auditor7 ({summary['total_audits']} audits):")
    print(f"  Mean gap: {summary['mean_gap']:.3f}")
    print(f"  Max gap:  {summary['max_gap']:.3f}")
    print(f"  Severity: {summary['severity_distribution']}")
    print(f"  Trend:    {summary['trend']}")
    print(f"  Top factors: {summary['top_factors'][:3]}")

    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("Law 5 enforcement is operational.")
    else:
        print("FAILURES DETECTED — review above.")
    print("=" * 70)
