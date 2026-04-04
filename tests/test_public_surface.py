"""Lightweight tests for the public package surface."""

import crt
from crt import contradiction, epistemics, governance, memory
from crt.governance import GovernanceLayer, GovernanceTier


def test_package_version_exposed():
    assert isinstance(crt.__version__, str)
    assert crt.__version__


def test_placeholder_namespaces_import():
    assert memory is not None
    assert contradiction is not None
    assert epistemics is not None


def test_governance_layer_still_works():
    gov = GovernanceLayer()
    result = gov.govern_response(
        "I think the answer might be X.",
        belief_confidence=0.6,
    )
    assert result.tier in {GovernanceTier.SAFE, GovernanceTier.FLAG, GovernanceTier.HEDGE, GovernanceTier.ESCALATE}
