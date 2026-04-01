"""CRT Governance — Immune agents enforcing constitutional laws at runtime.

Six autonomous monitors that inspect inputs and outputs at pipeline boundaries.
They never touch pipeline internals — they observe and intervene at boundaries.

Agents:
    TemplateDetector          — Law 2: Low variance does not imply high confidence
    SpeechLeakDetector        — Law 1: Speech cannot upgrade belief
    PrematureResolutionGuard  — Law 3: Contradiction must be preserved before resolution
    MemoryCorruptionGuard     — Law 4: Degraded reconstruction cannot silently overwrite trusted memory
    GapAuditor                — Law 5: Outward confidence must be bounded by internal support
    ContinuityAuditor         — Law 6: Confidence must not exceed continuity

Usage:
    from crt.governance import GovernanceLayer, GovernanceTier

    gov = GovernanceLayer()
    result = gov.govern_response("The answer is definitely X", belief_confidence=0.3)
    if result.should_block:
        # escalate to user
    elif result.tier == GovernanceTier.HEDGE:
        # reduce displayed confidence
"""

from .template_detector import TemplateDetector, Classification, DetectionResult
from .speech_leak_detector import SpeechLeakDetector, Verdict, VerdictType, MemoryRecord
from .premature_resolution_guard import (
    PrematureResolutionGuard, Contradiction, Disposition,
    ResolutionAction, VerdictAction,
)
from .memory_corruption_guard import (
    MemoryCorruptionGuard, Memory, OverwriteAction,
    OverwriteReason, OverwriteVerdict,
)
from .gap_auditor import (
    GapAuditor, ResponseAudit, GapVerdict, Severity, Action, Trend,
)
from .continuity_auditor import (
    ContinuityAuditor, ContinuityCheck, ContinuityVerdict,
    ContinuityAction, PriorResponse,
)
from .layer import GovernanceLayer, GovernanceTier, GovernanceAnnotation, GovernedResponse

__all__ = [
    # Layer
    "GovernanceLayer", "GovernanceTier", "GovernanceAnnotation", "GovernedResponse",
    # Template Detector
    "TemplateDetector", "Classification", "DetectionResult",
    # Speech Leak Detector
    "SpeechLeakDetector", "Verdict", "VerdictType", "MemoryRecord",
    # Premature Resolution Guard
    "PrematureResolutionGuard", "Contradiction", "Disposition",
    "ResolutionAction", "VerdictAction",
    # Memory Corruption Guard
    "MemoryCorruptionGuard", "Memory", "OverwriteAction",
    "OverwriteReason", "OverwriteVerdict",
    # Gap Auditor
    "GapAuditor", "ResponseAudit", "GapVerdict", "Severity", "Action", "Trend",
    # Continuity Auditor
    "ContinuityAuditor", "ContinuityCheck", "ContinuityVerdict",
    "ContinuityAction", "PriorResponse",
]
