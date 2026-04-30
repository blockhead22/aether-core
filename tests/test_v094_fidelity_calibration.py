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

try:
    import sentence_transformers  # noqa: F401
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

needs_networkx = pytest.mark.skipif(
    not _HAS_NETWORKX, reason="networkx required (install [graph] extra)",
)
needs_sentence_transformers = pytest.mark.skipif(
    not _HAS_SENTENCE_TRANSFORMERS,
    reason=(
        "sentence-transformers required for warm-mode calibration "
        "(install [ml] extra). Cold-mode tests below run regardless."
    ),
)

pytest.importorskip("mcp")

from bench.run_fidelity_bench import run_corpus, render_markdown


@needs_networkx
@needs_sentence_transformers
class TestFidelityCalibration:
    """Run the corpus, assert every blocker category passes.

    Skipped when sentence-transformers isn't installed (cold-mode
    fallback can't satisfy the warm-mode hit-rate guarantees). The
    cold-mode class below still runs and pins the substring-only path.
    """

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


@needs_networkx
class TestFidelityCalibrationColdMode:
    """v0.9.5: same corpus, cold encoder. Runs the production cold-start
    code path that the v0.9.4 warm-only bench missed.

    Cold-mode rates are inherently lower than warm-mode for some
    categories — slot extraction and embedding-similarity-gated checks
    (policy, negation_asymmetry, factual slot conflicts) need
    embeddings to function. Categories that work cold:
      - mutex_contradiction (regex-based)
      - methodological_overclaim (text+source-tag based)
      - false_positive_guard (negative tests)
      - no_issue_unrelated (negative tests)

    These assertions establish the v0.9.5 cold-mode baseline. Future
    fixes that expand cold-mode coverage will need to update the
    expected rates upward (good problem to have)."""

    @pytest.fixture(scope="class")
    def cold_summary(self):
        return run_corpus(cold_encoder=True)

    def test_methodological_cold_mode_at_least_80_percent(self, cold_summary):
        """v0.9.5 baseline: methodological detection works in cold mode
        for 4/5 cases. The 5th requires embedding retrieval (substring
        score below threshold even after the 0.10 floor)."""
        cat = cold_summary["per_category"].get("methodological_overclaim", {})
        total = cat.get("pass", 0) + cat.get("fail", 0)
        rate = cat.get("pass", 0) / total if total else 0.0
        assert rate >= 0.8, (
            f"methodological cold-mode rate {rate*100:.1f}% < 80% baseline. "
            f"Regression in v0.9.5 cold-encoder support."
        )

    def test_mutex_cold_mode_is_100_percent(self, cold_summary):
        """Mutex detection is regex-based and must work without any
        embeddings. If this regresses, the cold-mode path is broken."""
        cat = cold_summary["per_category"].get("mutex_contradiction", {})
        assert cat.get("fail", 0) == 0, (
            f"mutex cold-mode failed {cat['fail']} cases — regex-based "
            f"contradiction detection regressed"
        )

    def test_false_positive_guards_hold_in_cold_mode(self, cold_summary):
        """Specificity must hold in BOTH modes — methodological /
        contradiction detection must not spuriously fire on unrelated
        text just because we're in cold mode with looser thresholds."""
        cat = cold_summary["per_category"].get("false_positive_guard", {})
        assert cat.get("fail", 0) == 0, (
            f"false_positive_guard cold-mode failed {cat['fail']} cases "
            f"— v0.9.5 lower thresholds caused spurious firing"
        )
