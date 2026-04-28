"""Calibration regression test backed by bench/fidelity_corpus.json.

Layer 3 of the governance work: turn "fidelity catches the right things"
from anecdote into a measurable property. This test runs the full
benchmark corpus and asserts that every blocker category still passes.
Known-gap categories (e.g. quantitative-factual conflicts) are tracked
in the report but don't fail the suite — they're tracked limitations.

If you regress fidelity, this test fails with a per-category breakdown
of which case (or category) broke. The benchmark report itself is in
bench/fidelity_corpus.json; add cases to expand coverage.
"""

from __future__ import annotations

import pytest

try:
    import networkx  # noqa: F401
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx required (install [graph] extra)",
)

pytest.importorskip("mcp")

from bench.run_fidelity_bench import run_corpus, render_markdown


@needs_networkx
class TestFidelityCalibration:
    """Run the corpus, assert every blocker category passes."""

    @pytest.fixture(scope="class")
    def summary(self):
        # Run the corpus once for the whole class.
        return run_corpus()

    def test_every_blocker_category_passes(self, summary):
        """If a non-known-gap category regresses, this test fails with a
        full breakdown so the regression is easy to localize."""
        blocker_results = [
            r for r in summary["results"]
            if not r["category"].startswith("known_gap_")
        ]
        failing = [r for r in blocker_results if not r["passed"]]
        if failing:
            # Render the markdown report to give the maintainer
            # everything they need to debug the regression.
            report = render_markdown(summary)
            pytest.fail(
                f"{len(failing)} blocker case(s) regressed:\n\n{report}"
            )

    def test_blocker_pass_rate_is_100_percent(self, summary):
        """Numeric guarantee: blocker categories collectively pass 100%."""
        assert summary["blocker_pass_rate"] == 1.0, (
            f"blocker pass rate {summary['blocker_pass_rate']*100:.1f}% "
            f"(expected 100%); see report:\n{render_markdown(summary)}"
        )

    def test_methodological_recall_is_100_percent(self, summary):
        """The Layer 2 fix must hold across all methodological cases.
        If this regresses, fidelity has lost methodological-overclaim
        detection."""
        cat = summary["per_category"].get("methodological_overclaim", {})
        total = cat.get("pass", 0) + cat.get("fail", 0)
        assert cat.get("fail", 0) == 0, (
            f"methodological_overclaim failed {cat['fail']}/{total} cases — "
            f"Layer 2 regression"
        )

    def test_false_positive_guards_hold(self, summary):
        """Methodological / contradiction detection must not fire on
        unrelated topics or substrings ('son' shouldn't match 'so' marker)."""
        cat = summary["per_category"].get("false_positive_guard", {})
        total = cat.get("pass", 0) + cat.get("fail", 0)
        assert cat.get("fail", 0) == 0, (
            f"false_positive_guard failed {cat['fail']}/{total} cases — "
            f"specificity regression (the bench is firing on cases that "
            f"should NOT trigger)"
        )

    def test_corpus_has_at_least_one_case_per_blocker_category(self, summary):
        """Defensive: if someone deletes cases, the corpus shouldn't
        accidentally lose category coverage."""
        required_categories = {
            "factual_contradiction",
            "mutex_contradiction",
            "methodological_overclaim",
            "policy_violation",
            "negation_asymmetry",
            "no_issue_grounded",
            "no_issue_unrelated",
            "false_positive_guard",
        }
        present = set(summary["per_category"].keys())
        missing = required_categories - present
        assert not missing, f"corpus is missing categories: {missing}"
