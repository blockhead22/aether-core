"""Tests for the Aether governance layer."""

from aether.governance import (
    GovernanceLayer,
    GovernanceTier,
    TemplateDetector,
    Classification,
    GapAuditor,
    ResponseAudit,
    Severity,
    PrematureResolutionGuard,
    Contradiction,
    Disposition,
    ResolutionAction,
    VerdictAction,
    MemoryCorruptionGuard,
    Memory,
    OverwriteAction,
    OverwriteReason,
)


class TestTemplateDetector:
    def test_genuine_confidence(self):
        td = TemplateDetector()
        result = td.detect("Paris is the capital of France.")
        assert result.classification == Classification.GENUINE_CONFIDENCE

    def test_template_lock_detection(self):
        td = TemplateDetector()
        hedgy = (
            "I understand this is a complex topic with many perspectives. "
            "It's important to consider multiple viewpoints and I want to "
            "be balanced in my response. There are valid arguments on both sides."
        )
        result = td.detect(hedgy)
        assert result.classification == Classification.TEMPLATE_LOCK
        assert len(result.hedge_patterns_found) > 0


class TestGapAuditor:
    def test_safe_when_aligned(self):
        ga = GapAuditor()
        audit = ResponseAudit(
            response_text="I think it might be X.",
            belief_confidence=0.6,
        )
        verdict = ga.audit(audit)
        assert verdict.severity == Severity.SAFE

    def test_elevated_when_overconfident(self):
        ga = GapAuditor()
        audit = ResponseAudit(
            response_text="The answer is definitely and absolutely X without question.",
            belief_confidence=0.2,
        )
        verdict = ga.audit(audit)
        assert verdict.severity in (Severity.ELEVATED, Severity.CRITICAL)
        assert verdict.gap_score > 0.3


class TestPrematureResolutionGuard:
    def test_blocks_premature_resolution(self):
        guard = PrematureResolutionGuard()
        c = Contradiction(
            id="c1",
            claim_a="User prefers Python",
            claim_b="User prefers Rust",
            disposition=Disposition.HELD,
            evidence_count_a=3,
            evidence_count_b=3,
        )
        verdict = guard.check_resolution(c, ResolutionAction.RESOLVE_A)
        # Held contradictions with balanced evidence should not be resolved
        assert verdict.action in (VerdictAction.BLOCK, VerdictAction.WARN)

    def test_allows_resolvable(self):
        guard = PrematureResolutionGuard()
        c = Contradiction(
            id="c2",
            claim_a="User is 34",
            claim_b="User is 32",
            disposition=Disposition.RESOLVABLE,
            evidence_count_a=5,
            evidence_count_b=1,
            disposition_confidence=0.9,
        )
        verdict = guard.check_resolution(c, ResolutionAction.RESOLVE_A)
        # Resolvable with strong evidence imbalance should allow or warn (not block)
        assert verdict.action in (VerdictAction.ALLOW, VerdictAction.WARN)


class TestMemoryCorruptionGuard:
    def test_blocks_trust_downgrade(self):
        guard = MemoryCorruptionGuard()
        existing = Memory(
            id="m1", content="User works at Google", trust=0.9,
            source="user", fidelity=0.95,
        )
        replacement = Memory(
            id="m2", content="User works at startup", trust=0.3,
            source="generated", fidelity=0.4,
        )
        verdict = guard.check_overwrite(existing, replacement, OverwriteReason.RECONSTRUCTION)
        assert verdict.action in (OverwriteAction.BLOCK, OverwriteAction.DOWNGRADE)


class TestGovernanceLayer:
    def test_safe_response(self):
        gov = GovernanceLayer()
        result = gov.govern_response(
            "I think the answer might be X, but I'm not fully sure.",
            belief_confidence=0.6,
        )
        assert result.tier == GovernanceTier.SAFE
        assert result.passed

    def test_overconfident_response_hedged(self):
        gov = GovernanceLayer()
        result = gov.govern_response(
            "The answer is absolutely and definitively X. I am 100% certain.",
            belief_confidence=0.2,
        )
        assert result.tier in (GovernanceTier.HEDGE, GovernanceTier.ESCALATE)
        assert len(result.annotations) > 0
        assert result.confidence_adjustment < 0
