"""
Governance Layer — Dispatcher for CRT immune agents.

Wraps all six agents into a single boundary-inspection API.
Never touches pipeline internals. Observe and flag only.

Tiered response (never silent modification):
    SAFE     — pass through silently
    FLAG     — annotate + log, continue
    HEDGE    — reduce confidence in metadata, continue
    ESCALATE — block, surface conflict to user

Usage:
    gov = GovernanceLayer()

    # Before response reaches user:
    result = gov.govern_response("The answer is definitely X", belief_confidence=0.3)
    if result.should_block:
        # escalate to user
    elif result.tier == GovernanceTier.HEDGE:
        # reduce displayed confidence

    # Before memory write:
    result = gov.govern_memory_write("Paris is capital of France", 0.9, "generated", memories)

    # Before contradiction resolution:
    result = gov.govern_resolution(contradiction, ResolutionAction.RESOLVE_A)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .template_detector import TemplateDetector, Classification, DetectionResult
from .speech_leak_detector import (
    SpeechLeakDetector,
    Verdict as SpeechVerdict,
    VerdictType as SpeechVerdictType,
    MemoryRecord,
)
from .premature_resolution_guard import (
    PrematureResolutionGuard,
    Contradiction,
    ResolutionAction,
    VerdictAction as ResolutionVerdictAction,
)
from .memory_corruption_guard import (
    MemoryCorruptionGuard,
    Memory as ImmuneMemory,
    OverwriteAction,
    OverwriteReason,
    OverwriteVerdict,
)
from .gap_auditor import (
    GapAuditor,
    ResponseAudit,
    GapVerdict,
    Severity,
    Action as GapAction,
)


# ---------------------------------------------------------------------------
# Governance tiers
# ---------------------------------------------------------------------------

class GovernanceTier(str, Enum):
    SAFE = "safe"
    FLAG = "flag"
    HEDGE = "hedge"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Governance output types
# ---------------------------------------------------------------------------

@dataclass
class GovernanceAnnotation:
    """A single finding from one immune agent."""
    agent: str
    law: str
    finding: str
    severity: str
    details: dict = field(default_factory=dict)


@dataclass
class GovernedResponse:
    """The output of any governance check."""
    original_text: str
    tier: GovernanceTier
    annotations: list[GovernanceAnnotation] = field(default_factory=list)
    confidence_adjustment: float = 0.0  # negative = reduced
    should_block: bool = False
    audit_log: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.tier in (GovernanceTier.SAFE, GovernanceTier.FLAG)


# ---------------------------------------------------------------------------
# Governance Layer
# ---------------------------------------------------------------------------

class GovernanceLayer:
    """
    Wrapper that dispatches to immune agents at pipeline boundaries.
    Never touches pipeline internals. Observe and flag only.
    """

    def __init__(self, embedding_fn=None):
        self.template_detector = TemplateDetector()
        self.speech_leak_detector = SpeechLeakDetector(embedding_fn=embedding_fn)
        self.gap_auditor = GapAuditor()
        self.premature_resolution_guard = PrematureResolutionGuard()
        self.memory_corruption_guard = MemoryCorruptionGuard()

    # -------------------------------------------------------------------
    # Response governance (user-facing output)
    # -------------------------------------------------------------------

    def govern_response(
        self,
        text: str,
        belief_confidence: float,
        domain: Optional[str] = None,
        repeated_responses: Optional[list[str]] = None,
        susceptibility: Optional[float] = None,
        source_trust: Optional[float] = None,
    ) -> GovernedResponse:
        """
        Govern a response before it reaches the user.

        Phase 1 (always): TemplateDetector (cheap regex + pattern match).
        Phase 2 (lazy):   GapAuditor — only if Phase 1 flags something
                          OR belief_confidence < 0.5.
        """
        annotations: list[GovernanceAnnotation] = []
        audit_log: list[dict] = []
        t0 = time.monotonic()

        # --- Phase 1: TemplateDetector ---
        template_result = self.template_detector.detect(
            response_text=text,
            repeated_responses=repeated_responses,
        )
        audit_log.append({
            "agent": "template_detector",
            "elapsed_ms": _ms_since(t0),
            "classification": template_result.classification.value,
            "confidence": template_result.confidence,
        })

        template_flagged = template_result.classification == Classification.TEMPLATE_LOCK
        if template_flagged:
            annotations.append(GovernanceAnnotation(
                agent="template_detector",
                law="Law 2: Low variance does not imply high confidence",
                finding=f"Template lock detected ({template_result.confidence:.0%} confidence). "
                        f"Hedge patterns: {', '.join(template_result.hedge_patterns_found[:3])}",
                severity="flag",
                details={
                    "classification": template_result.classification.value,
                    "confidence": template_result.confidence,
                    "hedge_patterns": template_result.hedge_patterns_found,
                    "embedding_variance": template_result.embedding_variance,
                },
            ))

        # --- Phase 2: GapAuditor (lazy) ---
        gap_verdict: Optional[GapVerdict] = None
        run_gap = template_flagged or belief_confidence < 0.5
        if run_gap:
            t1 = time.monotonic()
            audit_input = ResponseAudit(
                response_text=text,
                belief_confidence=belief_confidence,
                template_detection=template_result if template_flagged else None,
                domain=domain,
                susceptibility=susceptibility,
                source_trust=source_trust,
            )
            gap_verdict = self.gap_auditor.audit(audit_input)
            audit_log.append({
                "agent": "gap_auditor",
                "elapsed_ms": _ms_since(t1),
                "gap_score": gap_verdict.gap_score,
                "severity": gap_verdict.severity.value,
                "action": gap_verdict.action.value,
            })

            if gap_verdict.severity != Severity.SAFE:
                annotations.append(GovernanceAnnotation(
                    agent="gap_auditor",
                    law=gap_verdict.law,
                    finding=f"Belief/speech gap: {gap_verdict.gap_score:.2f} "
                            f"(speech={gap_verdict.speech_confidence:.2f}, "
                            f"belief={gap_verdict.belief_confidence:.2f}). "
                            f"Factors: {', '.join(gap_verdict.contributing_factors[:3])}",
                    severity=gap_verdict.severity.value.lower(),
                    details={
                        "gap_score": gap_verdict.gap_score,
                        "speech_confidence": gap_verdict.speech_confidence,
                        "belief_confidence": gap_verdict.belief_confidence,
                        "factors": gap_verdict.contributing_factors,
                    },
                ))

        # --- Combine verdicts ---
        tier = _resolve_tier_response(template_result, gap_verdict)
        confidence_adj = 0.0
        if tier == GovernanceTier.HEDGE and gap_verdict:
            confidence_adj = -gap_verdict.gap_score
        elif tier == GovernanceTier.HEDGE and template_flagged:
            confidence_adj = -0.3

        return GovernedResponse(
            original_text=text,
            tier=tier,
            annotations=annotations,
            confidence_adjustment=confidence_adj,
            should_block=(tier == GovernanceTier.ESCALATE),
            audit_log=audit_log,
        )

    # -------------------------------------------------------------------
    # Memory write governance
    # -------------------------------------------------------------------

    def govern_memory_write(
        self,
        text: str,
        proposed_trust: float,
        source: str,
        existing_memories: list[MemoryRecord],
        existing_memory: Optional[ImmuneMemory] = None,
        proposed_replacement: Optional[ImmuneMemory] = None,
        overwrite_reason: Optional[OverwriteReason] = None,
    ) -> GovernedResponse:
        """
        Govern a memory write before it reaches storage.

        Always:  SpeechLeakDetector (is generated text trying to self-promote?).
        Lazy:    MemoryCorruptionGuard (only on overwrites).
        """
        annotations: list[GovernanceAnnotation] = []
        audit_log: list[dict] = []
        t0 = time.monotonic()

        # --- SpeechLeakDetector ---
        speech_verdict = self.speech_leak_detector.detect(
            candidate_text=text,
            proposed_trust=proposed_trust,
            source=source,
            existing_memories=existing_memories,
        )
        audit_log.append({
            "agent": "speech_leak_detector",
            "elapsed_ms": _ms_since(t0),
            "action": speech_verdict.action.value,
            "original_trust": speech_verdict.original_trust,
            "final_trust": speech_verdict.final_trust,
        })

        if speech_verdict.action != SpeechVerdictType.ALLOW:
            annotations.append(GovernanceAnnotation(
                agent="speech_leak_detector",
                law="Law 1: Speech cannot upgrade belief",
                finding=f"{speech_verdict.action.value}: {speech_verdict.reason}. "
                        f"Trust {speech_verdict.original_trust:.2f} -> {speech_verdict.final_trust:.2f}",
                severity="escalate" if speech_verdict.action == SpeechVerdictType.BLOCK else "hedge",
                details={
                    "action": speech_verdict.action.value,
                    "original_trust": speech_verdict.original_trust,
                    "final_trust": speech_verdict.final_trust,
                    "best_grounding": speech_verdict.best_grounding_similarity,
                    "reason": speech_verdict.reason,
                },
            ))

        # --- MemoryCorruptionGuard (lazy: only on overwrites) ---
        overwrite_verdict: Optional[OverwriteVerdict] = None
        if existing_memory is not None and proposed_replacement is not None:
            reason = overwrite_reason or OverwriteReason.NEW_EVIDENCE
            t1 = time.monotonic()
            overwrite_verdict = self.memory_corruption_guard.check_overwrite(
                existing_memory=existing_memory,
                proposed_replacement=proposed_replacement,
                reason=reason,
            )
            audit_log.append({
                "agent": "memory_corruption_guard",
                "elapsed_ms": _ms_since(t1),
                "action": overwrite_verdict.action.value,
                "trust_delta": overwrite_verdict.trust_delta,
                "fidelity_gap": overwrite_verdict.fidelity_gap,
            })

            if overwrite_verdict.action != OverwriteAction.ALLOW:
                annotations.append(GovernanceAnnotation(
                    agent="memory_corruption_guard",
                    law=overwrite_verdict.law,
                    finding=f"{overwrite_verdict.action.value}: {overwrite_verdict.reason}. "
                            f"Fidelity gap: {overwrite_verdict.fidelity_gap:.2f}",
                    severity="escalate" if overwrite_verdict.action == OverwriteAction.BLOCK else "hedge",
                    details={
                        "action": overwrite_verdict.action.value,
                        "trust_delta": overwrite_verdict.trust_delta,
                        "fidelity_gap": overwrite_verdict.fidelity_gap,
                        "content_similarity": overwrite_verdict.content_similarity,
                    },
                ))

        # --- Combine ---
        tier = _resolve_tier_memory(speech_verdict, overwrite_verdict)
        confidence_adj = 0.0
        if speech_verdict.action == SpeechVerdictType.DOWNGRADE:
            confidence_adj = speech_verdict.final_trust - speech_verdict.original_trust
        if overwrite_verdict and overwrite_verdict.action == OverwriteAction.DOWNGRADE:
            confidence_adj = min(confidence_adj, -overwrite_verdict.fidelity_gap)

        return GovernedResponse(
            original_text=text,
            tier=tier,
            annotations=annotations,
            confidence_adjustment=confidence_adj,
            should_block=(tier == GovernanceTier.ESCALATE),
            audit_log=audit_log,
        )

    # -------------------------------------------------------------------
    # Contradiction resolution governance
    # -------------------------------------------------------------------

    def govern_resolution(
        self,
        contradiction: Contradiction,
        proposed_action: ResolutionAction,
    ) -> GovernedResponse:
        """
        Govern a contradiction resolution attempt.

        Always runs PrematureResolutionGuard.
        """
        audit_log: list[dict] = []
        annotations: list[GovernanceAnnotation] = []
        t0 = time.monotonic()

        verdict = self.premature_resolution_guard.check_resolution(
            contradiction=contradiction,
            proposed_action=proposed_action,
        )
        audit_log.append({
            "agent": "premature_resolution_guard",
            "elapsed_ms": _ms_since(t0),
            "action": verdict.action.value,
            "disposition": verdict.disposition,
            "confidence": verdict.confidence,
        })

        if verdict.action != ResolutionVerdictAction.ALLOW:
            annotations.append(GovernanceAnnotation(
                agent="premature_resolution_guard",
                law=verdict.law,
                finding=f"{verdict.action.value}: {verdict.reason}",
                severity="escalate" if verdict.action == ResolutionVerdictAction.BLOCK else "flag",
                details={
                    "action": verdict.action.value,
                    "disposition": verdict.disposition,
                    "confidence": verdict.confidence,
                    "contradiction_id": contradiction.id,
                    "claim_a": contradiction.claim_a[:100],
                    "claim_b": contradiction.claim_b[:100],
                },
            ))

        tier_map = {
            ResolutionVerdictAction.ALLOW: GovernanceTier.SAFE,
            ResolutionVerdictAction.WARN: GovernanceTier.FLAG,
            ResolutionVerdictAction.BLOCK: GovernanceTier.ESCALATE,
        }
        tier = tier_map.get(verdict.action, GovernanceTier.FLAG)

        return GovernedResponse(
            original_text=f"Resolution({proposed_action.value}) on contradiction {contradiction.id}",
            tier=tier,
            annotations=annotations,
            should_block=(tier == GovernanceTier.ESCALATE),
            audit_log=audit_log,
        )


# ---------------------------------------------------------------------------
# Tier resolution helpers
# ---------------------------------------------------------------------------

def _resolve_tier_response(
    template: DetectionResult,
    gap: Optional[GapVerdict],
) -> GovernanceTier:
    """Combine template + gap verdicts into a single tier. Highest severity wins."""
    if gap is not None:
        if gap.action == GapAction.ESCALATE:
            return GovernanceTier.ESCALATE
        if gap.action == GapAction.HEDGE:
            return GovernanceTier.HEDGE
        if gap.action == GapAction.FLAG:
            return GovernanceTier.FLAG

    if template.classification == Classification.TEMPLATE_LOCK:
        return GovernanceTier.FLAG

    return GovernanceTier.SAFE


def _resolve_tier_memory(
    speech: SpeechVerdict,
    overwrite: Optional[OverwriteVerdict],
) -> GovernanceTier:
    """Combine speech leak + overwrite verdicts. Highest severity wins."""
    if speech.action == SpeechVerdictType.BLOCK:
        return GovernanceTier.ESCALATE
    if overwrite and overwrite.action == OverwriteAction.BLOCK:
        return GovernanceTier.ESCALATE
    if speech.action == SpeechVerdictType.DOWNGRADE:
        return GovernanceTier.HEDGE
    if overwrite and overwrite.action == OverwriteAction.DOWNGRADE:
        return GovernanceTier.HEDGE
    return GovernanceTier.SAFE


def _ms_since(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000, 2)
