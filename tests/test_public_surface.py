"""Tests for the public package surface — all namespaces export real classes."""

import aether
from aether import contradiction, epistemics, governance, memory
from aether.governance import GovernanceLayer, GovernanceTier
from aether.contradiction import StructuralTensionMeter, TensionRelationship
from aether.epistemics import EpistemicLoss, CorrectionEvent, DomainVolatility
from aether.memory import extract_fact_slots, ExtractedFact, MemoryNode, BeliefDependencyGraph


def test_package_version():
    assert aether.__version__ == "0.9.1"


def test_all_namespaces_have_real_exports():
    # Governance
    assert hasattr(governance, "GovernanceLayer")
    assert hasattr(governance, "GovernanceTier")
    # Contradiction
    assert hasattr(contradiction, "StructuralTensionMeter")
    assert hasattr(contradiction, "TensionRelationship")
    # Epistemics
    assert hasattr(epistemics, "EpistemicLoss")
    assert hasattr(epistemics, "DomainVolatility")
    # Memory
    assert hasattr(memory, "extract_fact_slots")
    assert hasattr(memory, "MemoryGraph")
    assert hasattr(memory, "BeliefDependencyGraph")


def test_governance_layer_still_works():
    gov = GovernanceLayer()
    result = gov.govern_response(
        "I think the answer might be X.",
        belief_confidence=0.6,
    )
    assert result.tier in {GovernanceTier.SAFE, GovernanceTier.FLAG, GovernanceTier.HEDGE, GovernanceTier.ESCALATE}


def test_cross_namespace_integration():
    """Tension meter uses memory slots — verify cross-namespace works."""
    meter = StructuralTensionMeter()
    result = meter.measure("I live in Seattle", "I live in Portland")
    assert result.tension_score >= 0
